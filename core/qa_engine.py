"""Source-grounded retrieval across expert-call transcripts."""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.answer_generator import VerificationResult, build_prompts, parse_model_reply, verify_model_reply
from core.extractor import verify_quote
from core.llm_service import OpenAICompatibleModelService
from core.models import ExpertTranscript, QAResponse, QuoteCitation
from core.textutil import market_terms as _market_terms, stem_token, tokenize as _tokens


STOP_WORDS = {
    "a", "about", "across", "affect", "against", "all", "also", "an", "and", "any", "are", "as", "at", "be", "between",
    "by", "can", "compare", "did", "do", "does", "each", "for", "from", "has", "have", "how", "i",
    "in", "is", "it", "me", "mention", "more", "most", "of", "on", "or", "rather", "say", "said",
    "tell", "than", "that", "the", "their", "them", "there", "these", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
    "europe", "european", "expert", "experts", "call", "calls", "transcript", "transcripts",
}

# Query-stem -> alternative stems that express the same idea in these calls.
# Small, explicit and reviewable: it widens recall without inventing content,
# because retrieval still returns only source turns.
SYNONYMS = {
    "timelin": {"month", "cycle"},
    "barrier": {"hold", "stall", "issue"},
    "growth": {"grow", "increas", "accelerat"},
    "trend": {"grow", "increas", "accelerat"},
    "optimistic": {"positive"},
    "roi": {"economic", "financial", "financ", "utilisation"},
    "budget": {"capital", "fund", "financ"},
    "outcome": {"clinical"},
    "expect": {"expectation", "outlook"},
    "purchase": {"buy", "procurement", "approval"},
    "decision": {"approval", "committee"},
    "decid": {"decision", "approval", "committee"},
}

MIN_COVERAGE = 0.3  # below this the best turn is not evidence for the question
HIGH_COVERAGE = 0.7
MEDIUM_COVERAGE = 0.45


def _name_terms(name: str) -> Set[str]:
    titles = {"dr", "prof", "mr", "ms", "mrs"}
    return {token for token in _tokens(name) if token not in titles and len(token) > 2}


@dataclass
class _ModelOutcome:
    """Either a finished response, or why the caller should fall back to source passages."""

    response: Optional[QAResponse] = None
    mode_used: str = ""
    note: str = ""


class CrossTranscriptQA:
    """Retrieves source turns with coverage-based scoring, then optionally re-ranks them."""

    def __init__(
        self,
        transcripts: List[ExpertTranscript],
        llm_service: Optional[OpenAICompatibleModelService] = None,
    ):
        self.transcripts = transcripts
        self.llm_service = llm_service
        self.indexed_turns: List[Dict[str, Any]] = []
        self._doc_freq: Dict[str, int] = {}
        self._answer_count = 0
        self._build_index()

    def _build_index(self) -> None:
        self.indexed_turns.clear()
        for transcript in self.transcripts:
            for index, turn in enumerate(transcript.turns):
                prompt_context = ""
                if not turn.is_interviewer and index and transcript.turns[index - 1].is_interviewer:
                    prompt_context = transcript.turns[index - 1].text
                self.indexed_turns.append(
                    {
                        "expert_id": transcript.profile.id,
                        "expert_name": transcript.profile.name,
                        "role": transcript.profile.role,
                        "market": transcript.profile.market,
                        "flag": transcript.profile.country_flag,
                        "turn_id": turn.turn_id,
                        "timestamp": turn.timestamp,
                        "timestamp_seconds": turn.timestamp_seconds,
                        "speaker": turn.speaker,
                        "is_interviewer": turn.is_interviewer,
                        "prompt_context": prompt_context,
                        "text": turn.text,
                        "text_stems": {stem_token(t) for t in _tokens(turn.text)},
                        "prompt_stems": {stem_token(t) for t in _tokens(prompt_context)},
                    }
                )

        # Document frequency over expert answers (answer text plus the question
        # that prompted it): rare terms carry more weight than common ones.
        self._doc_freq = {}
        answers = [item for item in self.indexed_turns if not item["is_interviewer"]]
        for item in answers:
            for stem in item["text_stems"] | item["prompt_stems"]:
                self._doc_freq[stem] = self._doc_freq.get(stem, 0) + 1
        self._answer_count = len(answers)

    def _idf(self, stem: str) -> float:
        return math.log(1 + (self._answer_count + 1) / (self._doc_freq.get(stem, 0) + 1))

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Split a query into topic stems plus market and expert filters."""
        raw_tokens = _tokens(query)
        query_lower = " ".join(raw_tokens)

        target_markets: List[str] = []
        expert_ids: List[str] = []
        market_words: Set[str] = set()
        name_words: Set[str] = set()
        for transcript in self.transcripts:
            market = transcript.profile.market
            terms = _market_terms(market)
            market_words |= terms
            mentioned = market.lower() in query_lower or any(term in raw_tokens for term in terms)
            if mentioned and market not in target_markets:
                target_markets.append(market)
            names = _name_terms(transcript.profile.name)
            name_words |= names
            if any(name in raw_tokens for name in names):
                expert_ids.append(transcript.profile.id)

        topic_tokens = [
            token
            for token in raw_tokens
            if token not in STOP_WORDS
            and token not in market_words
            and token not in name_words
            and len(token) > 2
        ]
        stems: List[str] = []
        surface: Dict[str, str] = {}
        for token in topic_tokens:
            stem = stem_token(token)
            if stem not in surface:
                stems.append(stem)
                surface[stem] = token
        return {"stems": stems, "surface": surface, "markets": target_markets, "expert_ids": expert_ids}

    def _score_turn(self, item: Dict[str, Any], stems: List[str]) -> Tuple[float, int]:
        """Return (IDF-weighted coverage of the query, number of matched terms)."""
        total = sum(self._idf(stem) for stem in stems)
        if not total:
            return 0.0, 0
        gained, matched = 0.0, 0
        for stem in stems:
            alternatives = SYNONYMS.get(stem, set())
            # Best of: the expert's own words, or the question they were answering.
            weight = max(
                1.0 if stem in item["text_stems"] else 0.0,
                0.85 if stem in item["prompt_stems"] else 0.0,
                0.8 if alternatives & item["text_stems"] else 0.0,
                0.6 if alternatives & item["prompt_stems"] else 0.0,
            )
            if not weight:
                continue
            gained += self._idf(stem) * weight
            matched += 1
        return gained / total, matched

    def _retrieve(self, query: str, top_k: int = 6, relative_cutoff: float = 0.6) -> Dict[str, Any]:
        parsed = self._parse_query(query)
        stems = parsed["stems"]
        known_stems = set(self._doc_freq)
        result: Dict[str, Any] = {
            "turns": [],
            "coverage": 0.0,
            "matched": 0,
            "stems": stems,
            "unsupported": [
                parsed["surface"][stem]
                for stem in stems
                if stem not in known_stems and not (SYNONYMS.get(stem, set()) & known_stems)
            ],
        }
        if not stems:
            return result

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for item in self.indexed_turns:
            if item["is_interviewer"]:
                continue
            if parsed["markets"] and item["market"] not in parsed["markets"]:
                continue
            if parsed["expert_ids"] and item["expert_id"] not in parsed["expert_ids"]:
                continue
            coverage, matched = self._score_turn(item, stems)
            if matched and coverage >= MIN_COVERAGE:
                scored.append((coverage, matched, item))
        if not scored:
            return result

        scored.sort(key=lambda entry: (-entry[0], entry[2]["timestamp_seconds"]))
        best_coverage, best_matched, _ = scored[0]

        if len(parsed["markets"]) > 1:
            # Comparison question: the strongest turn from each named market.
            selected = []
            for market in parsed["markets"]:
                match = next((entry for entry in scored if entry[2]["market"] == market), None)
                if match:
                    selected.append(match)
        else:
            selected = [entry for entry in scored if entry[0] >= relative_cutoff * best_coverage][:top_k]

        result.update(turns=[entry[2] for entry in selected], coverage=best_coverage, matched=best_matched)
        return result

    def search_relevant_turns(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        return self._retrieve(query, top_k)["turns"]

    @staticmethod
    def _confidence(coverage: float, matched: int) -> str:
        if matched >= 2 and coverage >= HIGH_COVERAGE:
            return "High"
        if matched >= 2 and coverage >= MEDIUM_COVERAGE:
            return "Medium"
        return "Low"

    def answer_query(self, query: str) -> QAResponse:
        retrieval = self._retrieve(query)
        relevant_turns = retrieval["turns"]
        if not relevant_turns:
            missing = retrieval["unsupported"]
            reason = (
                f"These terms do not appear anywhere in the transcripts: {', '.join(missing)}."
                if missing
                else "No transcript passage covers enough of the question's key terms."
            )
            return QAResponse(
                query=query,
                answer=(
                    "No directly relevant discussion was found in the transcripts for this query. "
                    "The app refuses to extrapolate beyond the supplied evidence."
                ),
                citations=[],
                referenced_experts=[],
                confidence="None",
                mode_used="Guardrail refusal",
                notes=reason,
            )

        confidence = self._confidence(retrieval["coverage"], retrieval["matched"])
        notes = f"Best passage covers {retrieval['coverage']:.0%} of the question's key terms ({retrieval['matched']} matched)."
        if retrieval["matched"] < 2:
            lead = (
                "The question has only one searchable keyword"
                if len(retrieval["stems"]) == 1
                else "Only one keyword matched"
            )
            notes = f"{lead}, so these passages may not answer it. Treat them as keyword hits, not findings. " + notes
        elif retrieval["unsupported"]:
            notes += f" Not found in any transcript: {', '.join(retrieval['unsupported'])}."

        mode_used = "Deterministic grounded retrieval"
        if self.llm_service and self.llm_service.is_configured():
            outcome = self._answer_with_model(query, confidence, notes)
            if outcome.response is not None:
                return outcome.response
            mode_used, notes = outcome.mode_used, f"{outcome.note} {notes}"
        response = self._render_source_evidence(query, relevant_turns, mode_used)
        response.confidence = confidence
        response.notes = notes
        return response

    def _answer_with_model(self, query: str, confidence: str, notes: str) -> _ModelOutcome:
        """Ask the model for a cited answer and keep only the claims that verify."""
        model = self.llm_service.model
        candidates = self._retrieve(query, top_k=8, relative_cutoff=0.4)["turns"]
        candidate_map = {f"S{index}": turn for index, turn in enumerate(candidates, start=1)}
        reply = self.llm_service.query(**build_prompts(query, candidate_map))
        if not reply.get("success"):
            return _ModelOutcome(
                mode_used="Deterministic grounded retrieval (model unavailable)",
                note="The model call failed, so only source passages are shown.",
            )

        result = verify_model_reply(
            parse_model_reply(reply.get("content", "")),
            candidate_map,
            {transcript.profile.id: transcript.raw_text for transcript in self.transcripts},
            {transcript.profile.market: _market_terms(transcript.profile.market) for transcript in self.transcripts},
        )
        if not result.supported:
            reason = result.reason or "The retrieved passages do not answer the question."
            return _ModelOutcome(
                response=QAResponse(
                    query=query,
                    answer=(
                        "The transcripts do not contain enough to answer this question. "
                        "The app refuses to extrapolate beyond the supplied evidence."
                    ),
                    citations=[],
                    referenced_experts=[],
                    confidence="None",
                    mode_used=f"Model refusal ({model})",
                    notes=f"The model judged the retrieved passages insufficient: {reason}",
                )
            )
        if not result.claims:
            return _ModelOutcome(
                mode_used="Source passages only (model output failed verification)",
                note="The model's answer could not be verified against the transcripts, so only source passages are shown.",
            )
        return _ModelOutcome(
            response=self._render_model_answer(query, result, candidate_map, confidence, notes, model)
        )

    def _render_model_answer(
        self,
        query: str,
        result: VerificationResult,
        candidate_map: Dict[str, Dict[str, Any]],
        confidence: str,
        notes: str,
        model: str,
    ) -> QAResponse:
        def tag(turn: Dict[str, Any]) -> str:
            return f"**[{turn['market']} · {turn['timestamp']}]**"

        lines = ["### Answer", "", "_Written by the model; every claim below is backed by a verified transcript quote._", ""]
        for claim in result.claims:
            tags = " ".join(dict.fromkeys(tag(candidate_map[quote.source_id]) for quote in claim.quotes))
            lines.append(f"- {claim.text} {tags}")
        lines += ["", "### Supporting quotes", ""]

        citations: List[QuoteCitation] = []
        experts: List[str] = []
        seen = set()
        for claim in result.claims:
            for quote in claim.quotes:
                turn = candidate_map[quote.source_id]
                key = (turn["expert_id"], turn["timestamp"], quote.quote)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {tag(turn)} \"{quote.quote}\"")
                citations.append(
                    QuoteCitation(
                        quote_text=quote.quote,
                        timestamp=turn["timestamp"],
                        timestamp_seconds=turn["timestamp_seconds"],
                        speaker=turn["speaker"],
                        is_verified=True,
                        verification_score=quote.score,
                        char_start=quote.char_start,
                        char_end=quote.char_end,
                        market=turn["market"],
                        transcript_id=turn["expert_id"],
                    )
                )
                header = f"{turn['expert_name']} ({turn['role']}, {turn['market']})"
                if header not in experts:
                    experts.append(header)

        if result.dropped:
            order = ["None", "Low", "Medium", "High"]
            confidence = order[max(order.index(confidence) - 1, 1)]
            plural = "s" if result.dropped != 1 else ""
            notes = (
                f"{result.dropped} claim{plural} removed because the quote could not be verified "
                f"against the transcript. {notes}"
            )
        return QAResponse(
            query=query,
            answer="\n".join(lines),
            citations=citations,
            referenced_experts=experts,
            confidence=confidence,
            mode_used=f"Grounded retrieval + {model} cited answer (claims verified)",
            notes=notes,
        )

    def _render_source_evidence(
        self, query: str, relevant_turns: List[Dict[str, Any]], mode_used: str
    ) -> QAResponse:
        by_expert: Dict[str, List[Dict[str, Any]]] = {}
        for turn in relevant_turns:
            if not turn["is_interviewer"]:
                header = f"{turn['expert_name']} ({turn['role']}, {turn['market']})"
                by_expert.setdefault(header, []).append(turn)

        answer_lines = ["### Evidence found in the transcripts", ""]
        citations: List[QuoteCitation] = []
        for header, turns in by_expert.items():
            answer_lines.append(f"#### {header}")
            for turn in turns:
                answer_lines.append(f"- **[{turn['market']} · {turn['timestamp']}]** \"{turn['text']}\"")
                raw_text = next(
                    (
                        transcript.raw_text
                        for transcript in self.transcripts
                        if transcript.profile.id == turn["expert_id"]
                    ),
                    "",
                )
                verified, score, start, end = verify_quote(turn["text"], raw_text)
                citations.append(
                    QuoteCitation(
                        quote_text=turn["text"],
                        timestamp=turn["timestamp"],
                        timestamp_seconds=turn["timestamp_seconds"],
                        speaker=turn["speaker"],
                        is_verified=verified,
                        verification_score=score,
                        char_start=start,
                        char_end=end,
                        market=turn["market"],
                        transcript_id=turn["expert_id"],
                    )
                )
            answer_lines.append("")

        return QAResponse(
            query=query,
            answer="\n".join(answer_lines),
            citations=citations,
            referenced_experts=list(by_expert),
            confidence="High",
            mode_used=mode_used,
        )

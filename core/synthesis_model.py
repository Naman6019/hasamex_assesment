"""Optional model-drafted themes and disagreements, with every quote verified.

The model proposes; code disposes. An item is shown only if each quote it cites
appears verbatim in the source turn it names, the figures and market names in
its text are backed by those quotes, and it cites at least two different calls.
The number of supporting calls is counted from the verified quotes, never taken
from the model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.answer_generator import (
    format_sources,
    parse_model_reply,
    text_is_grounded,
    verify_evidence,
)
from core.models import CrossExpertDisagreement, CrossExpertTheme, ExpertTranscript
from core.textutil import market_terms

MAX_SOURCE_TURNS = 120  # beyond this, only the turns cited as guide answers are sent

SYSTEM_PROMPT = """You compare expert-call transcripts. You are given numbered sources; each is one expert's answer from a call. Use ONLY these sources.

Find:
1. themes: points made by at least two DIFFERENT calls.
2. disagreements: points where experts from different calls take genuinely different positions (different conclusions, priorities or figures), not merely different wording.

Rules:
- Every theme and disagreement needs evidence: [{"source": "<S-label>", "quote": "<words copied verbatim from that source's Answer>"}].
  Copy the words exactly; do not paraphrase, join fragments, or add ellipses. Quote only from "Answer", never from "Question asked".
- Each item must cite sources from at least two different calls.
- Titles and descriptions are one or two plain sentences. Name a market only if you cite a source from it.
  Write numbers exactly as they appear in your quotes; do not compute, convert, or count experts.
- Never use outside knowledge. Return at most 6 themes and 5 disagreements, and fewer if the sources support fewer.

Reply with JSON only, in this shape:
{"themes": [{"title": "...", "summary": "...", "evidence": [{"source": "S1", "quote": "..."}]}],
 "disagreements": [{"topic": "...", "type": "<2-3 word label>", "description": "...", "evidence": [{"source": "S1", "quote": "..."}]}]}"""


@dataclass
class ModelDraft:
    themes: List[CrossExpertTheme] = field(default_factory=list)
    disagreements: List[CrossExpertDisagreement] = field(default_factory=list)
    dropped: int = 0
    notes: str = ""
    model: str = ""


def collect_sources(
    transcripts: List[ExpertTranscript], priority: Set[Tuple[int, str]]
) -> Dict[str, Dict[str, Any]]:
    """Number every expert turn (or, for large corpora, just the priority turns) as S1, S2, ..."""
    turns: List[Dict[str, Any]] = []
    for call, transcript in enumerate(transcripts):
        for index, turn in enumerate(transcript.turns):
            if turn.is_interviewer:
                continue
            previous = transcript.turns[index - 1] if index and transcript.turns[index - 1].is_interviewer else None
            turns.append(
                {
                    "call": call,
                    "expert_id": transcript.profile.id,
                    "expert_name": transcript.profile.name,
                    "market": transcript.profile.market,
                    "timestamp": turn.timestamp,
                    "speaker": turn.speaker,
                    "prompt_context": previous.text if previous else "",
                    "text": turn.text,
                }
            )
    if len(turns) > MAX_SOURCE_TURNS:
        turns = [turn for turn in turns if (turn["call"], turn["timestamp"]) in priority]
    return {f"S{number}": turn for number, turn in enumerate(turns, start=1)}


def draft_synthesis(
    transcripts: List[ExpertTranscript], llm_service: Any, priority: Set[Tuple[int, str]]
) -> ModelDraft:
    sources = collect_sources(transcripts, priority)
    draft = ModelDraft(model=llm_service.model)
    reply = llm_service.query(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Sources:\n\n{format_sources(sources)}",
    )
    if not reply.get("success"):
        draft.notes = "The model call failed, so the computed synthesis is shown."
        return draft
    payload = parse_model_reply(reply.get("content", ""))
    if payload is None or not isinstance(payload.get("themes", []), list) or not isinstance(
        payload.get("disagreements", []), list
    ):
        draft.notes = "The model's reply could not be parsed, so the computed synthesis is shown."
        return draft

    raw_texts = {t.profile.id: t.raw_text for t in transcripts}
    terms = {t.profile.market: market_terms(t.profile.market) for t in transcripts}
    call_total = len(transcripts)

    for item in payload.get("themes", []):
        checked = _verify_item(item, "title", "summary", sources, raw_texts, terms)
        if checked is None:
            draft.dropped += 1
            continue
        title, summary, evidence, calls = checked
        draft.themes.append(
            CrossExpertTheme(
                theme_id=f"model-theme-{len(draft.themes) + 1}",
                title=title,
                summary=summary,
                consensus_level=f"Supported by {calls} of {call_total} calls",
                supporting_evidence=evidence,
            )
        )
    for item in payload.get("disagreements", []):
        checked = _verify_item(item, "topic", "description", sources, raw_texts, terms)
        if checked is None:
            draft.dropped += 1
            continue
        topic, description, evidence, _ = checked
        label = str(item.get("type") or "").strip()[:40] if isinstance(item, dict) else ""
        draft.disagreements.append(
            CrossExpertDisagreement(
                topic=topic,
                description=description,
                divergence_type=label or "Model-identified difference",
                expert_positions=evidence,
            )
        )

    notes = []
    if draft.dropped:
        plural = "s" if draft.dropped != 1 else ""
        notes.append(
            f"{draft.dropped} model-drafted item{plural} removed because a quote, figure or market "
            "attribution could not be verified."
        )
    if not (draft.themes or draft.disagreements):
        notes.append("Nothing from the model survived verification, so the computed synthesis is shown.")
    elif not draft.themes:
        notes.append("No model theme survived verification, so computed themes are shown.")
    elif not draft.disagreements:
        notes.append("No model disagreement survived verification, so computed disagreements are shown.")
    draft.notes = " ".join(notes)
    return draft


def _verify_item(
    item: Any,
    title_key: str,
    text_key: str,
    sources: Dict[str, Dict[str, Any]],
    raw_texts: Dict[str, str],
    terms: Dict[str, Set[str]],
) -> Optional[Tuple[str, str, Dict[str, str], int]]:
    """Return (title, text, evidence, distinct calls) if the whole item verifies."""
    if not isinstance(item, dict):
        return None
    title = str(item.get(title_key) or "").strip()
    text = str(item.get(text_key) or "").strip()
    raw_evidence = item.get("evidence")
    if not title or not text or not isinstance(raw_evidence, list) or not raw_evidence:
        return None

    quotes = verify_evidence(raw_evidence, sources, raw_texts)
    if len(quotes) != len(raw_evidence):
        return None  # one unverifiable quote invalidates the item
    cited = [sources[quote.source_id] for quote in quotes]
    if len({turn["call"] for turn in cited}) < 2:
        return None
    markets = {turn["market"] for turn in cited}
    type_text = str(item.get("type") or "")
    if not text_is_grounded(f"{title} {text} {type_text}", (q.quote for q in quotes), markets, terms):
        return None

    evidence: Dict[str, str] = {}
    for quote, turn in zip(quotes, cited):
        key = f"{turn['market']} · {turn['expert_name']}"
        if key in evidence:
            key = f"{key} ({turn['timestamp']})"
        evidence[key] = f"[{turn['timestamp']}] {quote.quote}"
    return title, text, evidence, len({turn["call"] for turn in cited})

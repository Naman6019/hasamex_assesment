"""Optional LLM answer step with machine-verified citations.

The model may write a short answer, but it is never trusted. It must return
claims, each carrying verbatim quotes copied from numbered source turns. Code
then re-checks every claim against the transcripts and drops any that fail, so
nothing the model made up reaches the user.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.extractor import verify_quote


SYSTEM_PROMPT = """You answer questions about expert-call transcripts using ONLY the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never use outside knowledge, and never guess.
- Break your answer into short claims. Every claim needs one or more evidence items.
- Each evidence item is {"source": "<S-label>", "quote": "<words copied verbatim from that source's Answer>"}.
  Copy the words exactly, including spelling; do not paraphrase, join fragments, or add ellipses.
- Quote only from the "Answer" text of the source you cite, never from the "Question asked" line.
- Name the market or expert a claim is about only if you cite a source from that market.
- Write numbers exactly as they appear in your quotes. Do not compute, convert, or count experts.
- If the sources do not answer the question, return {"supported": false, "reason": "<one sentence>"}.

Reply with JSON only, in this shape:
{"supported": true, "claims": [{"text": "<one sentence>", "evidence": [{"source": "S1", "quote": "<verbatim words>"}]}]}"""

MIN_QUOTE_WORDS = 3
NUMBER_WORDS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    "million", "billion",
}


@dataclass
class VerifiedQuote:
    source_id: str
    quote: str
    char_start: Optional[int]
    char_end: Optional[int]
    score: float


@dataclass
class VerifiedClaim:
    text: str
    quotes: List[VerifiedQuote]
    markets: Set[str] = field(default_factory=set)


@dataclass
class VerificationResult:
    claims: List[VerifiedClaim] = field(default_factory=list)
    dropped: int = 0
    supported: bool = True
    reason: str = ""
    malformed: bool = False


def build_prompts(query: str, candidates: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    sources = "\n\n".join(
        f"{source_id} | {turn['market']} | {turn['timestamp']} | {turn['speaker']}\n"
        f"Question asked: {turn['prompt_context'] or '(not recorded)'}\n"
        f"Answer: {turn['text']}"
        for source_id, turn in candidates.items()
    )
    return {"system_prompt": SYSTEM_PROMPT, "user_prompt": f"Question: {query}\n\nSources:\n\n{sources}"}


def parse_model_reply(content: str) -> Optional[Dict[str, Any]]:
    """Parse the model's JSON, tolerating code fences and surrounding chatter."""
    text = (content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _figures(text: str) -> Set[str]:
    """Digits and spelled-out numbers: the details most often invented."""
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower())
    return {token for token in tokens if token[0].isdigit() or token in NUMBER_WORDS}


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _verify_quote_in_turn(quote: str, turn: Dict[str, Any], raw_text: str) -> Optional[VerifiedQuote]:
    clean = quote.strip(" \"'“”‘’")
    if len(clean.split()) < MIN_QUOTE_WORDS:
        return None
    found, score, offset, _ = verify_quote(clean, turn["text"])
    if not found:
        return None
    start = end = None
    turn_start = raw_text.find(turn["text"])
    if offset is not None and turn_start != -1:
        start = turn_start + offset
        end = start + len(clean)
    return VerifiedQuote(source_id="", quote=clean, char_start=start, char_end=end, score=score)


def verify_model_reply(
    payload: Any,
    candidates: Dict[str, Dict[str, Any]],
    raw_texts: Dict[str, str],
    market_terms: Dict[str, Set[str]],
) -> VerificationResult:
    """Keep only claims whose quotes, figures and market attributions check out."""
    result = VerificationResult()
    if not isinstance(payload, dict):
        result.malformed = True
        return result
    if payload.get("supported") is False:
        result.supported = False
        result.reason = str(payload.get("reason") or "").strip()
        return result

    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        result.malformed = True
        return result

    for claim in claims:
        verified = _verify_claim(claim, candidates, raw_texts, market_terms)
        if verified:
            result.claims.append(verified)
        else:
            result.dropped += 1
    return result


def _verify_claim(
    claim: Any,
    candidates: Dict[str, Dict[str, Any]],
    raw_texts: Dict[str, str],
    market_terms: Dict[str, Set[str]],
) -> Optional[VerifiedClaim]:
    if not isinstance(claim, dict):
        return None
    text = str(claim.get("text") or "").strip()
    evidence = claim.get("evidence")
    if not text or not isinstance(evidence, list) or not evidence:
        return None

    quotes: List[VerifiedQuote] = []
    markets: Set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            return None
        source_id = str(item.get("source") or "").strip().upper()
        turn = candidates.get(source_id)
        if turn is None:
            return None
        verified = _verify_quote_in_turn(str(item.get("quote") or ""), turn, raw_texts.get(turn["expert_id"], ""))
        if verified is None:
            return None
        verified.source_id = source_id
        quotes.append(verified)
        markets.add(turn["market"])

    # Any figure in the claim must appear in the quotes that back it.
    quoted_figures = _figures(" ".join(quote.quote for quote in quotes))
    if not _figures(text) <= quoted_figures:
        return None

    # A claim about a market must cite that market.
    claim_tokens = _tokens(text)
    for market, terms in market_terms.items():
        if terms & claim_tokens and market not in markets:
            return None
    return VerifiedClaim(text=text, quotes=quotes, markets=markets)

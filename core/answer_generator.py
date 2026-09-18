"""Optional LLM answer step with machine-verified citations.

The model may write a short answer, but it is never trusted. It must return
claims, each carrying verbatim quotes copied from numbered source turns. Code
then re-checks every claim against the transcripts and drops any that fail, so
nothing the model made up reaches the user.

The helpers here (quote, figure and market-attribution checks) are shared with
the model-drafted synthesis in ``core/synthesis_model.py``.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

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

_UNITS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_COMPOUND = re.compile(rf"\b({'|'.join(_TENS)})[- ]({'|'.join(_ONES)})\b")
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


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
    return {"system_prompt": SYSTEM_PROMPT, "user_prompt": f"Question: {query}\n\nSources:\n\n{format_sources(candidates)}"}


def format_sources(candidates: Dict[str, Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{source_id} | {turn['market']} | {turn['timestamp']} | {turn['speaker']}\n"
        f"Question asked: {turn['prompt_context'] or '(not recorded)'}\n"
        f"Answer: {turn['text']}"
        for source_id, turn in candidates.items()
    )


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


def figures(text: str) -> Set[str]:
    """Numbers in ``text`` as digit strings, so "six", "6" and "twenty-four"/"24" compare equal.

    Figures are the details most often invented, so claims are checked against them.
    "one" is ignored because it is usually a pronoun ("one surgeon").
    """
    lowered = _COMPOUND.sub(lambda m: str(_TENS[m.group(1)] + _ONES[m.group(2)]), text.lower())
    found: Set[str] = set()
    for token in _NUMBER_TOKEN.findall(lowered):
        if token[0].isdigit():
            found.add(str(int(float(token))) if float(token) == int(float(token)) else token)
        elif token in _UNITS:
            found.add(str(_UNITS[token]))
        elif token in _TENS:
            found.add(str(_TENS[token]))
        elif token in _SCALES:
            found.add(str(_SCALES[token]))
    return found


def word_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def verify_quote_in_turn(quote: str, turn: Dict[str, Any], raw_text: str) -> Optional[VerifiedQuote]:
    """Return the quote (with transcript offsets) only if it is verbatim in ``turn``."""
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


def text_is_grounded(
    text: str,
    quotes: Iterable[str],
    cited_markets: Set[str],
    market_terms: Dict[str, Set[str]],
) -> bool:
    """Model-written text may only state figures its quotes contain and name markets it cites."""
    if not figures(text) <= figures(" ".join(quotes)):
        return False
    words = word_set(text)
    return all(not (terms & words) or market in cited_markets for market, terms in market_terms.items())


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


def verify_evidence(
    evidence: Any,
    candidates: Dict[str, Dict[str, Any]],
    raw_texts: Dict[str, str],
) -> List[VerifiedQuote]:
    """Verified quotes for every evidence item that checks out; invalid items are skipped."""
    verified: List[VerifiedQuote] = []
    if not isinstance(evidence, list):
        return verified
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source") or "").strip().upper()
        turn = candidates.get(source_id)
        if turn is None:
            continue
        quote = verify_quote_in_turn(str(item.get("quote") or ""), turn, raw_texts.get(turn["expert_id"], ""))
        if quote is not None:
            quote.source_id = source_id
            verified.append(quote)
    return verified


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

    # Every cited quote must verify; one bad quote invalidates the claim.
    quotes = verify_evidence(evidence, candidates, raw_texts)
    if len(quotes) != len(evidence):
        return None
    markets = {candidates[quote.source_id]["market"] for quote in quotes}
    if not text_is_grounded(text, (quote.quote for quote in quotes), markets, market_terms):
        return None
    return VerifiedClaim(text=text, quotes=quotes, markets=markets)

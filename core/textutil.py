"""Shared word-level text helpers for retrieval and guide-question extraction."""

import re
from typing import List, Set


def stem_token(word: str) -> str:
    """Small normaliser that keeps lexical retrieval predictable and explainable."""
    word = word.lower().strip()
    if word.startswith("purchas"):
        return "purchase"
    if word.startswith("utilis"):
        return "utilisation"
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Known market adjectives; the market name itself is always included.
MARKET_ALIASES = {
    "france": ("french",),
    "germany": ("german",),
    "united kingdom": ("uk", "british", "britain"),
    "italy": ("italian",),
    "spain": ("spanish",),
}


def market_terms(market: str) -> Set[str]:
    """Words that refer to ``market`` in a query or in model-written text."""
    key = market.lower()
    terms = set(tokenize(key))
    terms.update(MARKET_ALIASES.get(key, ()))
    return terms

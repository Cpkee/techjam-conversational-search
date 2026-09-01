"""Low-level text helpers with no dependency on any other starter module -- shared by
catalog indexing, slot extraction, Stage 2 query building, and CLARIFY's catalog-side
signal matching."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _first_word_match(text: str, words: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return word
    return None

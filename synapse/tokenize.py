from __future__ import annotations

import re
from collections.abc import Iterable

_TERM_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]+", re.UNICODE)


def extract_terms(text: str) -> list[str]:
    terms = (t.lower() for t in _TERM_RE.findall(text))
    return [t for t in terms if len(t) >= 2]


def term_frequencies(text: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    for term in extract_terms(text):
        freq[term] = freq.get(term, 0) + 1
    return freq


def count_tokens(text: str) -> int:
    # Definição de token do protótipo: termos alfanuméricos (normalizados).
    return len(extract_terms(text))


def split_terms(terms: list[str], max_tokens: int) -> list[list[str]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    return [terms[i : i + max_tokens] for i in range(0, len(terms), max_tokens)]


def join_terms(terms: Iterable[str]) -> str:
    return " ".join(terms)


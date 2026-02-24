from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .tokenize import term_frequencies
from .util import parse_dt


@dataclass(slots=True)
class MemoryMeta:
    id: str
    path: str
    title: str | None
    token_count: int
    status: str
    last_used_at: datetime


@dataclass(slots=True)
class InMemoryIndex:
    # termo -> {memory_id: tf}
    postings: dict[str, dict[str, int]]
    meta: dict[str, MemoryMeta]
    num_docs: int
    avg_len: float

    @classmethod
    def empty(cls) -> "InMemoryIndex":
        return cls(postings={}, meta={}, num_docs=0, avg_len=0.0)

    @classmethod
    def build_from_rows(cls, rows: list[dict]) -> "InMemoryIndex":
        postings: dict[str, dict[str, int]] = {}
        meta: dict[str, MemoryMeta] = {}

        total_len = 0
        for row in rows:
            mem_id = row["id"]
            content = row["content"]
            freq = term_frequencies(content)
            for term, tf in freq.items():
                by_doc = postings.get(term)
                if by_doc is None:
                    by_doc = {}
                    postings[term] = by_doc
                by_doc[mem_id] = tf

            token_count = int(row["token_count"])
            total_len += token_count
            meta[mem_id] = MemoryMeta(
                id=mem_id,
                path=row["path"],
                title=row["title"],
                token_count=token_count,
                status=row["status"],
                last_used_at=parse_dt(row["last_used_at"]),
            )

        num_docs = len(meta)
        avg_len = (total_len / num_docs) if num_docs else 0.0
        return cls(postings=postings, meta=meta, num_docs=num_docs, avg_len=avg_len)


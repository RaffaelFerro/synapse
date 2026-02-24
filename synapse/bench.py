from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any

from .engine import SynapseEngine
from .tokenize import join_terms
from .util import Timer


@dataclass(slots=True)
class BenchReport:
    inserted: int
    insert_ms: float
    rebuild_ms: float
    queries: int
    search_p50_ms: float
    search_p95_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inserted": self.inserted,
            "insert_ms": self.insert_ms,
            "rebuild_ms": self.rebuild_ms,
            "queries": self.queries,
            "search_p50_ms": self.search_p50_ms,
            "search_p95_ms": self.search_p95_ms,
        }


def run_bench(
    engine: SynapseEngine,
    *,
    memories: int,
    tokens_per_memory: int,
    vocab_size: int = 5000,
    queries: int = 100,
    prompt_terms: int = 3,
    seed: int = 1,
    storage_limit_bytes: int | None = None,
) -> BenchReport:
    if storage_limit_bytes is not None:
        engine.config["storage_limit_bytes"] = int(storage_limit_bytes)

    rnd = random.Random(seed)
    vocab = [f"t{i}" for i in range(vocab_size)]

    t = Timer.start_new()
    for i in range(memories):
        terms = (rnd.choice(vocab) for _ in range(tokens_per_memory))
        content = join_terms(terms)
        engine.add_memory(
            path=f"bench/{i % 50}",
            title=f"Mem {i}",
            content=content,
            rebuild_index=False,
        )
    insert_ms = t.ms()

    t = Timer.start_new()
    engine.rebuild_index()
    rebuild_ms = t.ms()

    times: list[float] = []
    for _ in range(queries):
        q = " ".join(rnd.choice(vocab) for _ in range(prompt_terms))
        res = engine.search(q)
        times.append(float(res.timings_ms["total"]))

    times.sort()
    p50 = statistics.median(times) if times else 0.0
    p95 = times[int(0.95 * (len(times) - 1))] if len(times) >= 2 else (times[0] if times else 0.0)

    return BenchReport(
        inserted=memories,
        insert_ms=insert_ms,
        rebuild_ms=rebuild_ms,
        queries=queries,
        search_p50_ms=p50,
        search_p95_ms=p95,
    )


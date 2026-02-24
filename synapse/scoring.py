from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from .index import InMemoryIndex, MemoryMeta
from .util import utc_now


def _idf(num_docs: int, df: int) -> float:
    # idf(t) = ln((N - df + 0.5)/(df + 0.5) + 1)
    if num_docs <= 0:
        return 0.0
    return math.log(((num_docs - df + 0.5) / (df + 0.5)) + 1.0)


def _bm25_term(tf: int, doc_len: int, avg_len: float, *, k1: float, b: float) -> float:
    if tf <= 0:
        return 0.0
    if avg_len <= 0:
        return 0.0
    denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
    return (tf * (k1 + 1.0)) / denom


def score_query(
    index: InMemoryIndex,
    query_terms: list[str],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
    debug: bool = False,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """
    Retorna:
      - scores: memory_id -> score final
      - debug_info (opcional)
    """
    now = now or utc_now()

    # Estágio 1: acumula idf por doc ao percorrer postings.
    stage1: dict[str, float] = defaultdict(float)
    per_term_stats: dict[str, dict[str, float]] = {}
    for term in query_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = _idf(index.num_docs, df)
        if debug:
            per_term_stats[term] = {"df": float(df), "idf": float(idf)}
        for mem_id, tf in postings.items():
            stage1[mem_id] += idf * (1.0 + math.log(1.0 + tf))

    candidate_ids = list(stage1.keys())

    threshold = int(config.get("stage2_candidate_threshold", 2000))
    max_stage2 = int(config.get("stage2_max_candidates", 1000))

    # Estágio 2: BM25 completo (apenas se necessário).
    use_stage2 = len(candidate_ids) > threshold
    if use_stage2:
        candidate_ids.sort(key=lambda mid: stage1[mid], reverse=True)
        candidate_ids = candidate_ids[:max_stage2]

    k1 = float(config.get("bm25_k1", 1.2))
    b = float(config.get("bm25_b", 0.75))

    scores: dict[str, float] = {}
    bm25_detail: dict[str, dict[str, float]] = {}
    for mem_id in candidate_ids:
        meta = index.meta.get(mem_id)
        if meta is None:
            continue
        doc_len = meta.token_count

        bm25 = 0.0
        for term in query_terms:
            postings = index.postings.get(term)
            if not postings:
                continue
            tf = postings.get(mem_id, 0)
            if tf <= 0:
                continue
            df = len(postings)
            idf = _idf(index.num_docs, df)
            bm25 += idf * _bm25_term(tf, doc_len, index.avg_len, k1=k1, b=b)

        score = bm25 if use_stage2 or len(candidate_ids) else 0.0
        if not use_stage2:
            # Se não houve stage2, bm25 já foi calculado acima do mesmo jeito.
            score = bm25

        score = _apply_recency(score, meta, config, now=now)
        score = _apply_obsolete_penalty(score, meta, config)

        scores[mem_id] = score
        if debug:
            bm25_detail[mem_id] = {"bm25": bm25}

    debug_info = None
    if debug:
        debug_info = {
            "now": now.isoformat(),
            "num_docs": index.num_docs,
            "avg_len": index.avg_len,
            "use_stage2": use_stage2,
            "stage1_candidates": len(stage1),
            "stage2_candidates": len(candidate_ids),
            "per_term": per_term_stats,
            "bm25": bm25_detail,
        }

    return scores, debug_info


def _apply_obsolete_penalty(score: float, meta: MemoryMeta, config: dict[str, Any]) -> float:
    if meta.status != "obsolete":
        return score
    penalty = float(config.get("obsolete_penalty", 0.5))
    return score * penalty


def _apply_recency(score: float, meta: MemoryMeta, config: dict[str, Any], *, now: datetime) -> float:
    half_life_days = float(config.get("recency_half_life_days", 30))
    weight = float(config.get("recency_weight", 0.25))
    if half_life_days <= 0 or weight <= 0:
        return score
    age_days = (now - meta.last_used_at).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    recency = math.exp(-age_days / half_life_days)
    return score * (1.0 + weight * recency)


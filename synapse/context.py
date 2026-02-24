from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from .index import InMemoryIndex
from .util import isoformat_z


def build_context(
    conn: sqlite3.Connection,
    index: InMemoryIndex,
    query_terms: list[str],
    scores: dict[str, float],
    config: dict[str, Any],
    *,
    now: datetime,
    debug: bool = False,
    score_debug: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    max_total_tokens = int(config["max_tokens_per_prompt"])
    max_memories = int(config["max_memories_per_query"])
    max_refs = int(config["max_references"])

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    selected: list[str] = []
    total_tokens = 0

    # 1) Seleção primária por score.
    for mem_id, _ in ordered:
        if len(selected) >= max_memories:
            break
        meta = index.meta.get(mem_id)
        if meta is None:
            continue
        if meta.token_count <= 0:
            continue
        if total_tokens + meta.token_count > max_total_tokens:
            continue
        selected.append(mem_id)
        total_tokens += meta.token_count

    primary_count = len(selected)

    # 2) Resolver partes (split) e referências diretas dentro dos limites.
    if selected and (len(selected) < max_memories):
        expansion_queue = list(selected)
        expanded: set[str] = set()
        qi = 0
        while qi < len(expansion_queue) and len(selected) < max_memories:
            parent_id = expansion_queue[qi]
            qi += 1
            if parent_id in expanded:
                continue
            expanded.add(parent_id)

            # 2.1) Parts (children por parent_id)
            for part_id in _list_parts(conn, parent_id):
                if len(selected) >= max_memories:
                    break
                if part_id in expanded or part_id in selected:
                    continue
                meta = index.meta.get(part_id)
                if meta is None:
                    continue
                if total_tokens + meta.token_count > max_total_tokens:
                    continue
                selected.append(part_id)
                expansion_queue.append(part_id)
                total_tokens += meta.token_count

            # 2.2) User references
            if max_refs <= 0:
                continue
            for ref_id in _list_references(conn, parent_id, limit=max_refs):
                if len(selected) >= max_memories:
                    break
                if ref_id in expanded or ref_id in selected:
                    continue
                meta = index.meta.get(ref_id)
                if meta is None:
                    continue
                if total_tokens + meta.token_count > max_total_tokens:
                    continue
                selected.append(ref_id)
                expansion_queue.append(ref_id)
                total_tokens += meta.token_count

    memories = [_fetch_memory(conn, mem_id) for mem_id in selected]

    payload: dict[str, Any] = {
        "schema": "synapse.context.v1",
        "generated_at": isoformat_z(now),
        "limits": {
            "max_tokens_per_prompt": max_total_tokens,
            "max_memories_per_query": max_memories,
            "max_references": max_refs,
        },
        "query_terms": query_terms,
        "memories": memories,
    }

    if debug:
        payload["debug"] = {
            "primary_count": primary_count,
            "selected_count": len(selected),
            "token_count_total": total_tokens,
            "scores": {mid: scores.get(mid, 0.0) for mid in selected},
            "scoring": score_debug or {},
        }

    return payload, selected


def _list_references(conn: sqlite3.Connection, memory_id: str, *, limit: int) -> list[str]:
    rows = conn.execute(
        "SELECT referenced_memory_id FROM memory_reference WHERE memory_id=? "
        "ORDER BY created_at ASC LIMIT ?;",
        (memory_id, limit),
    ).fetchall()
    return [r["referenced_memory_id"] for r in rows]


def _fetch_memory(conn: sqlite3.Connection, memory_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, path, title, content, token_count, status, parent_id, part_index, part_total, "
        "created_at, updated_at, last_used_at, use_count "
        "FROM memory WHERE id=?;",
        (memory_id,),
    ).fetchone()
    if row is None:
        return {"id": memory_id, "missing": True}

    refs = _list_references(conn, memory_id, limit=9999)

    return {
        "id": row["id"],
        "path": row["path"],
        "title": row["title"],
        "status": row["status"],
        "parent_id": row["parent_id"],
        "part_index": row["part_index"],
        "part_total": row["part_total"],
        "token_count": int(row["token_count"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_used_at": row["last_used_at"],
        "use_count": int(row["use_count"]),
        "references": refs,
        "content": row["content"],
    }


def format_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _list_parts(conn: sqlite3.Connection, parent_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM memory WHERE parent_id=? ORDER BY part_index ASC, created_at ASC;",
        (parent_id,),
    ).fetchall()
    return [r["id"] for r in rows]

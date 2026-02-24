from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .util import get_lock_owner, isoformat_z, utc_now


def run_gc(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    now = now or utc_now()

    obsolescence_days = int(config["obsolescence_days"])
    delete_after_days = int(config["obsolete_delete_after_days"])

    obsolete_cutoff = now - timedelta(days=obsolescence_days)
    delete_cutoff = now - timedelta(days=(obsolescence_days + delete_after_days))

    obsolete_cutoff_s = isoformat_z(obsolete_cutoff)
    delete_cutoff_s = isoformat_z(delete_cutoff)

    with conn:
        marked = conn.execute(
            "UPDATE memory SET status='obsolete', updated_at=? "
            "WHERE status='active' AND last_used_at <= ?;",
            (isoformat_z(now), obsolete_cutoff_s),
        ).rowcount

        deleted = conn.execute(
            "DELETE FROM memory WHERE status='obsolete' AND last_used_at <= ?;",
            (delete_cutoff_s,),
        ).rowcount

        cleaned_mem_locks = _cleanup_locks(conn, table="memory_lock", now=now)
        cleaned_op_locks = _cleanup_locks(conn, table="op_lock", now=now)

    return {
        "marked_obsolete": int(marked or 0),
        "deleted": int(deleted or 0),
        "cleaned_memory_locks": int(cleaned_mem_locks),
        "cleaned_op_locks": int(cleaned_op_locks),
    }


def acquire_memory_lock(conn: sqlite3.Connection, memory_id: str, *, now: datetime | None = None) -> str:
    now = now or utc_now()
    owner = get_lock_owner()
    with conn:
        conn.execute(
            "INSERT INTO memory_lock(memory_id, locked_at, lock_owner) VALUES(?, ?, ?);",
            (memory_id, isoformat_z(now), owner),
        )
    return owner


def release_memory_lock(conn: sqlite3.Connection, memory_id: str, owner: str) -> None:
    with conn:
        conn.execute(
            "DELETE FROM memory_lock WHERE memory_id=? AND lock_owner=?;",
            (memory_id, owner),
        )


def acquire_op_lock(conn: sqlite3.Connection, op: str, *, now: datetime | None = None) -> str:
    now = now or utc_now()
    owner = get_lock_owner()
    with conn:
        conn.execute(
            "INSERT INTO op_lock(op, locked_at, lock_owner) VALUES(?, ?, ?);",
            (op, isoformat_z(now), owner),
        )
    return owner


def release_op_lock(conn: sqlite3.Connection, op: str, owner: str) -> None:
    with conn:
        conn.execute("DELETE FROM op_lock WHERE op=? AND lock_owner=?;", (op, owner))


def _cleanup_locks(conn: sqlite3.Connection, *, table: str, now: datetime) -> int:
    # Limpeza simples: locks com > 10 minutos.
    cutoff = isoformat_z(now - timedelta(minutes=10))
    if table == "memory_lock":
        res = conn.execute("DELETE FROM memory_lock WHERE locked_at <= ?;", (cutoff,))
    elif table == "op_lock":
        res = conn.execute("DELETE FROM op_lock WHERE locked_at <= ?;", (cutoff,))
    else:
        raise ValueError("unknown lock table")
    return int(res.rowcount or 0)


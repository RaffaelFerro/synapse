from __future__ import annotations

import sqlite3
from typing import Any

from .util import json_dumps, json_loads

SCHEMA_VERSION = 3

DEFAULT_DB_PATH = "synapse.db"

DEFAULT_CONFIG: dict[str, Any] = {
    # ESCOPO defaults
    "storage_limit_bytes": 100 * 1024 * 1024,  # 100MB
    "max_levels": 6,
    "max_references": 10,
    "max_versions": 10,
    "max_tokens_per_memory": 512,
    "max_tokens_per_prompt": 1500,
    "max_memories_per_query": 10,
    "inactivity_minutes": 30,
    "obsolescence_days": 60,
    "obsolete_delete_after_days": 30,
    "obsolete_penalty": 0.5,
    "anonymous_mode": False,
    "debug_mode_default": False,
    "kg_extraction_enabled": True,
    "kg_auto_merge_threshold": 0.92,
    # Scoring / performance knobs (protótipo)
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "recency_half_life_days": 30,
    "recency_weight": 0.25,
    "stage2_candidate_threshold": 2000,
    "stage2_max_candidates": 1000,
    # Sessão de histórico temporário
    "current_session_id": None,
    "last_interaction_at": None,
}


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS config (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory (
              id TEXT PRIMARY KEY,
              path TEXT NOT NULL,
              title TEXT,
              content TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('active','obsolete')),
              parent_id TEXT,
              part_index INTEGER,
              part_total INTEGER,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              last_used_at TEXT NOT NULL,
              use_count INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(parent_id) REFERENCES memory(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS memory_path_idx ON memory(path);
            CREATE INDEX IF NOT EXISTS memory_status_idx ON memory(status);
            CREATE INDEX IF NOT EXISTS memory_last_used_idx ON memory(last_used_at);

            CREATE TABLE IF NOT EXISTS memory_version (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              memory_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              content TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE,
              UNIQUE(memory_id, version)
            );

            CREATE INDEX IF NOT EXISTS mv_memory_id_idx ON memory_version(memory_id);

            CREATE TABLE IF NOT EXISTS memory_reference (
              memory_id TEXT NOT NULL,
              referenced_memory_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY(memory_id, referenced_memory_id),
              FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE,
              FOREIGN KEY(referenced_memory_id) REFERENCES memory(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS temp_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              prompt TEXT NOT NULL,
              response TEXT
            );

            CREATE INDEX IF NOT EXISTS temp_history_session_idx ON temp_history(session_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_lock (
              memory_id TEXT PRIMARY KEY,
              locked_at TEXT NOT NULL,
              lock_owner TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS op_lock (
              op TEXT PRIMARY KEY,
              locked_at TEXT NOT NULL,
              lock_owner TEXT NOT NULL
            );

            -- Knowledge Graph Tables
            CREATE TABLE IF NOT EXISTS entities (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL UNIQUE,
              type TEXT,
              description TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS entities_name_idx ON entities(name);

            CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(id UNINDEXED, name);

            CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON entities BEGIN
                INSERT INTO entities_fts(rowid, id, name) VALUES (new.rowid, new.id, new.name);
            END;
            
            CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON entities BEGIN
                INSERT INTO entities_fts(entities_fts, rowid, id, name) VALUES('delete', old.rowid, old.id, old.name);
            END;
            
            CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON entities BEGIN
                INSERT INTO entities_fts(entities_fts, rowid, id, name) VALUES('delete', old.rowid, old.id, old.name);
                INSERT INTO entities_fts(rowid, id, name) VALUES (new.rowid, new.id, new.name);
            END;

            CREATE TABLE IF NOT EXISTS relations (
              id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              target_id TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES entities(id) ON DELETE CASCADE,
              FOREIGN KEY(target_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS relations_source_idx ON relations(source_id);
            CREATE INDEX IF NOT EXISTS relations_target_idx ON relations(target_id);

            CREATE TABLE IF NOT EXISTS observations (
              id TEXT PRIMARY KEY,
              entity_id TEXT,
              relation_id TEXT,
              content TEXT NOT NULL,
              memory_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
              FOREIGN KEY(relation_id) REFERENCES relations(id) ON DELETE CASCADE,
              FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS obs_entity_idx ON observations(entity_id);
            CREATE INDEX IF NOT EXISTS obs_relation_idx ON observations(relation_id);
            CREATE INDEX IF NOT EXISTS obs_memory_idx ON observations(memory_id);

            CREATE TABLE IF NOT EXISTS indexing_jobs (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','processing','completed','failed')),
                attempts INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memory(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS indexing_jobs_status_idx ON indexing_jobs(status);
            """
        )

        _set_meta(conn, "schema_version", SCHEMA_VERSION)

    ensure_default_config(conn)
    _ensure_columns(conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    # Migrações leves para dbs já inicializados.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory);").fetchall()}
    with conn:
        if "part_index" not in cols:
            conn.execute("ALTER TABLE memory ADD COLUMN part_index INTEGER;")
        if "part_total" not in cols:
            conn.execute("ALTER TABLE memory ADD COLUMN part_total INTEGER;")


def ensure_default_config(conn: sqlite3.Connection) -> None:
    with conn:
        for key, value in DEFAULT_CONFIG.items():
            conn.execute(
                "INSERT OR IGNORE INTO config(key, value) VALUES(?, ?);",
                (key, json_dumps(value)),
            )


def _set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
        (key, json_dumps(value)),
    )


def get_config(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM config WHERE key=?;", (key,)).fetchone()
    if row is None:
        return DEFAULT_CONFIG.get(key)
    return json_loads(row["value"])


def set_config(conn: sqlite3.Connection, key: str, value: Any) -> None:
    with conn:
        conn.execute(
            "INSERT INTO config(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
            (key, json_dumps(value)),
        )


def load_config(conn: sqlite3.Connection) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    rows = conn.execute("SELECT key, value FROM config;").fetchall()
    for row in rows:
        config[row["key"]] = json_loads(row["value"])
    return config

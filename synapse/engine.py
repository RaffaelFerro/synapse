from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from . import db as db_mod
from .context import build_context
from .index import InMemoryIndex
from .llm import get_llm_client
from .maintenance import (
    acquire_memory_lock,
    acquire_op_lock,
    release_memory_lock,
    release_op_lock,
    run_gc,
)
from .scoring import score_query
from .tokenize import count_tokens, extract_terms, join_terms, split_terms
from .util import Timer, isoformat_z, new_id, parse_dt, utc_now


class SynapseError(RuntimeError):
    pass


@dataclass(slots=True)
class SearchResult:
    context: dict[str, Any]
    selected_ids: list[str]
    timings_ms: dict[str, float]


class SynapseEngine:
    def __init__(self, db_path: str, *, debug: bool = False):
        self.conn = db_mod.connect(db_path)
        db_mod.init_db(self.conn)
        self.config = db_mod.load_config(self.conn)
        self.debug = debug or bool(self.config.get("debug_mode_default", False))
        self.index = InMemoryIndex.empty()
        self.rebuild_index()

    def close(self) -> None:
        self.conn.close()

    def refresh_config(self) -> None:
        self.config = db_mod.load_config(self.conn)

    # ---------- Mode ----------

    def is_anonymous(self) -> bool:
        return bool(self.config.get("anonymous_mode", False))

    def set_anonymous(self, enabled: bool) -> None:
        db_mod.set_config(self.conn, "anonymous_mode", enabled)
        self.refresh_config()

    # ---------- Index ----------

    def rebuild_index(self) -> None:
        rows = self.conn.execute(
            "SELECT id, path, title, content, token_count, status, last_used_at FROM memory;"
        ).fetchall()
        # sqlite3.Row é mapeável, mas convertemos para dict para manter tipo simples.
        self.index = InMemoryIndex.build_from_rows([dict(r) for r in rows])

    # ---------- Memory ops ----------

    def add_memory(
        self, *, path: str, content: str, title: str | None = None, rebuild_index: bool = True
    ) -> str:
        self._require_writable()
        path = self._normalize_path(path)
        self._validate_path_levels(path)

        now = utc_now()
        mem_id = new_id()

        # enforce storage limit (aprox, baseado apenas em memory.content)
        self._ensure_storage_budget(len(content))

        max_tokens_per_memory = int(self.config["max_tokens_per_memory"])
        full_terms = extract_terms(content)
        full_token_count = len(full_terms)

        if full_token_count <= max_tokens_per_memory:
            with self.conn:
                self._insert_memory_row(
                    mem_id,
                    path=path,
                    title=title,
                    content=content,
                    token_count=full_token_count,
                    status="active",
                    now=now,
                    parent_id=None,
                    part_index=None,
                    part_total=None,
                )
                self._insert_version(mem_id, version=1, content=content, token_count=full_token_count, now=now)
        else:
            self._split_and_store(
                mem_id=mem_id,
                path=path,
                title=title,
                full_content=content,
                full_terms=full_terms,
                now=now,
            )

        if rebuild_index:
            self.rebuild_index()
        return mem_id

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        path: str | None = None,
        title: str | None = None,
    ) -> None:
        self._require_writable()

        now = utc_now()
        owner = None
        try:
            owner = acquire_memory_lock(self.conn, memory_id, now=now)
        except sqlite3.IntegrityError as e:
            raise SynapseError("memória em escrita exclusiva (lock ativo)") from e

        try:
            row = self.conn.execute(
                "SELECT id, path, title, content, token_count FROM memory WHERE id=?;",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise SynapseError("memória não encontrada")

            new_path = self._normalize_path(path or row["path"])
            self._validate_path_levels(new_path)
            new_title = title if title is not None else row["title"]

            if content is None:
                content = row["content"]

            self._ensure_storage_budget(len(content), excluding_memory_id=memory_id)

            max_tokens_per_memory = int(self.config["max_tokens_per_memory"])
            full_terms = extract_terms(content)
            full_token_count = len(full_terms)

            next_version = self._next_version(memory_id)

            with self.conn:
                self.conn.execute("DELETE FROM memory_reference WHERE memory_id=?;", (memory_id,))
                self.conn.execute("DELETE FROM memory WHERE parent_id=?;", (memory_id,))

                if full_token_count <= max_tokens_per_memory:
                    self.conn.execute(
                        "UPDATE memory SET path=?, title=?, content=?, token_count=?, updated_at=? "
                        "WHERE id=?;",
                        (
                            new_path,
                            new_title,
                            content,
                            full_token_count,
                            isoformat_z(now),
                            memory_id,
                        ),
                    )
                    self._insert_version(
                        memory_id,
                        version=next_version,
                        content=content,
                        token_count=full_token_count,
                        now=now,
                    )
                else:
                    # mantém a memória (id) e recria children/parts
                    self.conn.execute(
                        "UPDATE memory SET path=?, title=?, updated_at=? WHERE id=?;",
                        (new_path, new_title, isoformat_z(now), memory_id),
                    )
                    self._split_and_store(
                        mem_id=memory_id,
                        path=new_path,
                        title=new_title,
                        full_content=content,
                        full_terms=full_terms,
                        now=now,
                        version=next_version,
                        update_existing=True,
                    )

                self._enforce_version_limit(memory_id)
        finally:
            if owner is not None:
                release_memory_lock(self.conn, memory_id, owner)

        self.rebuild_index()

    def show_memory(self, memory_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT id, path, title, content, token_count, status, parent_id, part_index, part_total, "
            "created_at, updated_at, last_used_at, use_count "
            "FROM memory WHERE id=?;",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise SynapseError("memória não encontrada")
        return dict(row)

    # ---------- References ----------

    def add_reference(self, memory_id: str, referenced_id: str) -> None:
        self._require_writable()
        now = utc_now()

        max_refs = int(self.config["max_references"])
        if max_refs <= 0:
            raise SynapseError("max_references=0 (referências desabilitadas)")

        exists = self.conn.execute("SELECT 1 FROM memory WHERE id=?;", (memory_id,)).fetchone()
        if exists is None:
            raise SynapseError("memória origem não encontrada")
        exists = self.conn.execute("SELECT 1 FROM memory WHERE id=?;", (referenced_id,)).fetchone()
        if exists is None:
            raise SynapseError("memória referenciada não encontrada")

        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM memory_reference WHERE memory_id=?;",
            (memory_id,),
        ).fetchone()
        if int(row["c"]) >= max_refs:
            raise SynapseError("limite de referências atingido")

        with self.conn:
            self.conn.execute(
                "INSERT INTO memory_reference(memory_id, referenced_memory_id, created_at) VALUES(?, ?, ?) "
                "ON CONFLICT(memory_id, referenced_memory_id) DO NOTHING;",
                (memory_id, referenced_id, isoformat_z(now)),
            )

    def list_references(self, memory_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT referenced_memory_id FROM memory_reference WHERE memory_id=? ORDER BY created_at ASC;",
            (memory_id,),
        ).fetchall()
        return [r["referenced_memory_id"] for r in rows]

    def clear_references(self, memory_id: str) -> int:
        self._require_writable()
        with self.conn:
            res = self.conn.execute("DELETE FROM memory_reference WHERE memory_id=?;", (memory_id,))
        return int(res.rowcount or 0)

    # ---------- Search ----------

    def search(self, prompt: str) -> SearchResult:
        t0 = Timer.start_new()
        now = utc_now()

        query_terms = extract_terms(prompt)
        t_terms = t0.ms()

        scores, score_dbg = score_query(self.index, query_terms, self.config, now=now, debug=self.debug)
        t_score = t0.ms()

        context_obj, selected_ids = build_context(
            self.conn,
            self.index,
            query_terms,
            scores,
            self.config,
            now=now,
            debug=self.debug,
            score_debug=score_dbg,
        )
        t_ctx = t0.ms()

        if (not self.is_anonymous()) and selected_ids:
            self._touch_used(selected_ids, now=now)
            # mantém o índice coerente para recency do próximo search
            for mem_id in selected_ids:
                meta = self.index.meta.get(mem_id)
                if meta is not None:
                    meta.last_used_at = now
            # ajusta o payload retornado para refletir o touch (evita confusão em debug)
            for mem in context_obj.get("memories", []):
                if mem.get("missing"):
                    continue
                mem["last_used_at"] = isoformat_z(now)
                try:
                    mem["use_count"] = int(mem.get("use_count", 0)) + 1
                except Exception:
                    mem["use_count"] = mem.get("use_count", 0)

        timings = {
            "extract_terms": t_terms,
            "scoring": t_score - t_terms,
            "context": t_ctx - t_score,
            "total": t_ctx,
        }
        return SearchResult(context=context_obj, selected_ids=selected_ids, timings_ms=timings)

    # ---------- History ----------

    def ingest(self, *, prompt: str, response: str | None = None) -> str:
        self._require_writable()

        now = utc_now()
        session_id = self._get_or_create_session(now=now)
        with self.conn:
            self.conn.execute(
                "INSERT INTO temp_history(session_id, created_at, prompt, response) VALUES(?, ?, ?, ?);",
                (session_id, isoformat_z(now), prompt, response),
            )
        return session_id

    def consolidate(self, *, path: str | None = None, title: str | None = None) -> list[str]:
        """
        Consolida a sessão atual usando LLM para extrair decisões, atualizações e conflitos.
        """
        self._require_writable()

        now = utc_now()
        session_id = self.config.get("current_session_id")
        if not session_id:
            raise SynapseError("nenhuma sessão ativa para consolidar")

        rows = self.conn.execute(
            "SELECT created_at, prompt, response FROM temp_history WHERE session_id=? ORDER BY created_at ASC;",
            (session_id,),
        ).fetchall()
        if not rows:
            raise SynapseError("histórico temporário vazio")

        # 1. Preparar histórico bruto
        history_lines = []
        for r in rows:
            history_lines.append(f"User: {r['prompt']}")
            if r['response']:
                history_lines.append(f"Assistant: {r['response']}")
        history_text = "\n".join(history_lines)

        # 2. Buscar contexto relevante para evitar duplicatas (e ajudar a LLM a identificar atualizações)
        search_res = self.search(history_text)
        context_json = json.dumps(search_res.context, ensure_ascii=False)

        # 3. Carregar Prompt
        sys_prompt = self._load_prompt("consolidate.txt")

        # 4. Chamar LLM
        llm = get_llm_client()
        user_input = f"CONTEXTO ATUAL:\n{context_json}\n\nHISTÓRICO DA SESSÃO:\n{history_text}"
        
        raw_res = llm.chat_completion(sys_prompt, user_input)
        if not raw_res:
            raise SynapseError("falha na resposta da LLM durante a consolidação")

        try:
            data = json.loads(raw_res)
        except json.JSONDecodeError:
            raise SynapseError("LLM retornou JSON inválido")

        # 5. Processar Resultados
        created_ids = []

        with self.conn:
            # Novas decisões
            for dec in data.get("novas_decisoes", []):
                mem_id = self.add_memory(
                    path=dec.get("path") or path or "consolidado",
                    title=dec.get("title") or title,
                    content=dec.get("content"),
                    rebuild_index=False
                )
                created_ids.append(mem_id)

            # Atualizações
            for upd in data.get("atualizacoes", []):
                mem_id = upd.get("id")
                new_content = upd.get("content")
                if mem_id and new_content:
                    self.update_memory(mem_id, content=new_content)
                    created_ids.append(mem_id)

            # TODO: Lidar com conflitos semânticos reportados (data.get("conflitos"))
            # Por enquanto, apenas registramos se houver log de debug.

            # 6. Limpar histórico
            self.conn.execute("DELETE FROM temp_history WHERE session_id=?;", (session_id,))
            db_mod.set_config(self.conn, "current_session_id", None) # Finaliza sessão

        self.rebuild_index()
        return created_ids

    def _load_prompt(self, name: str) -> str:
        prompt_path = Path(__file__).parent / "prompts" / name
        if not prompt_path.exists():
            raise SynapseError(f"prompt {name} não encontrado em {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    # ---------- Maintenance / Ops ----------

    def gc(self) -> dict[str, int]:
        self._require_writable()
        stats = run_gc(self.conn, self.config, now=utc_now())
        self.rebuild_index()
        return stats

    def optimize(self, *, vacuum: bool = True) -> None:
        self._require_writable()
        now = utc_now()
        owner = None
        try:
            owner = acquire_op_lock(self.conn, "optimize", now=now)
        except sqlite3.IntegrityError as e:
            raise SynapseError("operação /optimize já em execução") from e

        try:
            self.conn.execute("PRAGMA optimize;")
            if vacuum:
                self.conn.execute("VACUUM;")
        finally:
            if owner is not None:
                release_op_lock(self.conn, "optimize", owner)

    # ---------- /mem ----------

    def mem(self, path_prefix: str | None = None) -> dict[str, Any]:
        prefix = None if path_prefix is None else self._normalize_path(path_prefix)

        rows = self.conn.execute(
            "SELECT id, path, title, status, token_count, parent_id, part_index, part_total "
            "FROM memory ORDER BY path ASC, id ASC;"
        ).fetchall()

        if prefix is None or prefix == "":
            level1: set[str] = set()
            for r in rows:
                if not r["path"]:
                    continue
                level1.add(r["path"].split("/", 1)[0])
            return {"type": "categories", "prefix": "", "items": sorted(level1)}

        next_levels: set[str] = set()
        memories_here: list[dict[str, Any]] = []

        for r in rows:
            p = r["path"] or ""
            if p == prefix:
                memories_here.append(dict(r))
                continue
            if not p.startswith(prefix + "/"):
                continue
            rest = p[len(prefix) + 1 :]
            next_levels.add(rest.split("/", 1)[0])

        if next_levels:
            return {"type": "categories", "prefix": prefix, "items": sorted(next_levels)}
        return {"type": "memories", "prefix": prefix, "items": memories_here}

    # ---------- internals ----------

    def _require_writable(self) -> None:
        if self.is_anonymous():
            raise SynapseError("modo anônimo ativo: persistência desabilitada")

    def _normalize_path(self, path: str) -> str:
        path = (path or "").strip().strip("/")
        # colapsa múltiplas barras
        while "//" in path:
            path = path.replace("//", "/")
        return path

    def _validate_path_levels(self, path: str) -> None:
        if not path:
            raise SynapseError("path obrigatório")
        levels = [p for p in path.split("/") if p]
        if len(levels) > int(self.config["max_levels"]):
            raise SynapseError("path excede max_levels")

    def _insert_memory_row(
        self,
        mem_id: str,
        *,
        path: str,
        title: str | None,
        content: str,
        token_count: int,
        status: str,
        now: datetime,
        parent_id: str | None,
        part_index: int | None,
        part_total: int | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO memory(id, path, title, content, token_count, status, parent_id, part_index, part_total, "
            "created_at, updated_at, last_used_at, use_count) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);",
            (
                mem_id,
                path,
                title,
                content,
                int(token_count),
                status,
                parent_id,
                part_index,
                part_total,
                isoformat_z(now),
                isoformat_z(now),
                isoformat_z(now),
            ),
        )

    def _insert_version(self, mem_id: str, *, version: int, content: str, token_count: int, now: datetime) -> None:
        self.conn.execute(
            "INSERT INTO memory_version(memory_id, version, content, token_count, created_at) "
            "VALUES(?, ?, ?, ?, ?);",
            (mem_id, int(version), content, int(token_count), isoformat_z(now)),
        )

    def _next_version(self, mem_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM memory_version WHERE memory_id=?;",
            (mem_id,),
        ).fetchone()
        return int(row["v"]) + 1

    def _enforce_version_limit(self, mem_id: str) -> None:
        max_versions = int(self.config["max_versions"])
        rows = self.conn.execute(
            "SELECT version FROM memory_version WHERE memory_id=? ORDER BY version ASC;",
            (mem_id,),
        ).fetchall()
        if len(rows) <= max_versions:
            return
        to_delete = [int(r["version"]) for r in rows[: len(rows) - max_versions]]
        self.conn.executemany(
            "DELETE FROM memory_version WHERE memory_id=? AND version=?;",
            [(mem_id, v) for v in to_delete],
        )

    def _touch_used(self, memory_ids: list[str], *, now: datetime) -> None:
        with self.conn:
            self.conn.executemany(
                "UPDATE memory SET last_used_at=?, use_count=use_count+1 WHERE id=?;",
                [(isoformat_z(now), mid) for mid in memory_ids],
            )

    def _ensure_storage_budget(self, added_bytes: int, *, excluding_memory_id: str | None = None) -> None:
        limit_bytes = int(self.config["storage_limit_bytes"])
        if limit_bytes <= 0:
            return
        if excluding_memory_id:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)),0) AS s FROM memory WHERE id != ?;",
                (excluding_memory_id,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COALESCE(SUM(LENGTH(content)),0) AS s FROM memory;").fetchone()
        current = int(row["s"])
        if current + added_bytes > limit_bytes:
            raise SynapseError("storage_limit_bytes excedido (rode /gc ou aumente o limite)")

    def _get_or_create_session(self, *, now: datetime) -> str:
        inactivity = int(self.config["inactivity_minutes"])
        last = self.config.get("last_interaction_at")
        session = self.config.get("current_session_id")

        new_session_needed = False
        if not session:
            new_session_needed = True
        elif not last:
            new_session_needed = True
        else:
            try:
                last_dt = parse_dt(str(last))
                if now - last_dt > timedelta(minutes=inactivity):
                    new_session_needed = True
            except Exception:
                new_session_needed = True

        if new_session_needed:
            session = new_id()

        db_mod.set_config(self.conn, "current_session_id", session)
        db_mod.set_config(self.conn, "last_interaction_at", isoformat_z(now))
        self.refresh_config()
        return str(session)

    def _split_and_store(
        self,
        *,
        mem_id: str,
        path: str,
        title: str | None,
        full_content: str,
        full_terms: list[str],
        now: datetime,
        version: int = 1,
        update_existing: bool = False,
    ) -> None:
        max_tokens_per_memory = int(self.config["max_tokens_per_memory"])
        parts = split_terms(full_terms, max_tokens_per_memory)
        part_total = len(parts)

        stub = f"[SPLIT] Conteúdo dividido em {part_total} partes (max {max_tokens_per_memory} tokens/parte)."
        stub_token_count = count_tokens(stub)

        with self.conn:
            if update_existing:
                self.conn.execute(
                    "UPDATE memory SET content=?, token_count=?, status='active', part_index=NULL, part_total=?, "
                    "parent_id=NULL, updated_at=? WHERE id=?;",
                    (stub, stub_token_count, part_total, isoformat_z(now), mem_id),
                )
            else:
                self._insert_memory_row(
                    mem_id,
                    path=path,
                    title=title,
                    content=stub,
                    token_count=stub_token_count,
                    status="active",
                    now=now,
                    parent_id=None,
                    part_index=None,
                    part_total=part_total,
                )

            self._insert_version(
                mem_id, version=version, content=full_content, token_count=len(full_terms), now=now
            )

            # Children/parts (não contam como "references" do usuário; são ligados por parent_id).
            for idx, chunk_terms in enumerate(parts, start=1):
                chunk_id = new_id()
                chunk_content = join_terms(chunk_terms)
                chunk_title = title
                if chunk_title:
                    chunk_title = f"{chunk_title} (parte {idx}/{part_total})"
                else:
                    chunk_title = f"Parte {idx}/{part_total}"
                self._insert_memory_row(
                    chunk_id,
                    path=path,
                    title=chunk_title,
                    content=chunk_content,
                    token_count=len(chunk_terms),
                    status="active",
                    now=now,
                    parent_id=mem_id,
                    part_index=idx,
                    part_total=part_total,
                )
                self._insert_version(
                    chunk_id, version=1, content=chunk_content, token_count=len(chunk_terms), now=now
                )

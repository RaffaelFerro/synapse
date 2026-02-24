from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import run_bench
from .db import DEFAULT_DB_PATH
from .engine import SynapseEngine, SynapseError


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    extracted = _extract_globals(argv)
    argv = extracted.argv

    parser = argparse.ArgumentParser(prog="synapse", add_help=True)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do SQLite (default: synapse.db)")
    parser.add_argument("--debug", action="store_true", help="Modo debug (métricas e detalhes)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Inicializa banco e config padrão")

    p_add = sub.add_parser("add", help="Adiciona uma memória")
    p_add.add_argument("--path", required=True)
    p_add.add_argument("--title")
    p_add_src = p_add.add_mutually_exclusive_group(required=True)
    p_add_src.add_argument("--content")
    p_add_src.add_argument("--content-file")

    p_upd = sub.add_parser("update", help="Atualiza uma memória (gera nova versão)")
    p_upd.add_argument("id")
    p_upd.add_argument("--path")
    p_upd.add_argument("--title")
    p_upd_src = p_upd.add_mutually_exclusive_group(required=False)
    p_upd_src.add_argument("--content")
    p_upd_src.add_argument("--content-file")

    p_show = sub.add_parser("show", help="Mostra uma memória (JSON)")
    p_show.add_argument("id")

    p_search = sub.add_parser("search", help="Busca memórias relevantes e monta contexto (JSON)")
    p_search_src = p_search.add_mutually_exclusive_group(required=True)
    p_search_src.add_argument("--prompt")
    p_search_src.add_argument("--prompt-file")

    p_mem = sub.add_parser("mem", help="Inspeciona hierarquia /mem")
    p_mem.add_argument("prefix", nargs="?")

    p_anon = sub.add_parser("anonymous", help="Ativa/desativa modo anônimo")
    p_anon.add_argument("state", choices=["on", "off"])

    sub.add_parser("rebuild", help="Reconstrói o index em RAM")

    p_opt = sub.add_parser("optimize", help="Reorganização/otimização (protótipo)")
    p_opt.add_argument("--no-vacuum", action="store_true")

    sub.add_parser("gc", help="Marca obsoletas e apaga expiradas")

    p_ing = sub.add_parser("ingest", help="Salva interação no histórico temporário")
    p_ing.add_argument("--prompt", required=True)
    p_ing.add_argument("--response")

    p_con = sub.add_parser("consolidate", help="Consolida histórico temporário (placeholder MVP)")
    p_con.add_argument("--path")
    p_con.add_argument("--title")

    p_ref = sub.add_parser("ref", help="Gerencia referências (até max_references)")
    ref_sub = p_ref.add_subparsers(dest="ref_cmd", required=True)
    p_ref_add = ref_sub.add_parser("add", help="Adiciona referência: <origem> -> <destino>")
    p_ref_add.add_argument("from_id")
    p_ref_add.add_argument("to_id")
    p_ref_list = ref_sub.add_parser("list", help="Lista referências de uma memória")
    p_ref_list.add_argument("from_id")
    p_ref_clear = ref_sub.add_parser("clear", help="Remove todas as referências de uma memória")
    p_ref_clear.add_argument("from_id")

    p_bench = sub.add_parser("bench", help="Benchmark sintético (gera dataset e mede latência)")
    p_bench.add_argument("--memories", type=int, default=10_000)
    p_bench.add_argument("--tokens-per-memory", type=int, default=128)
    p_bench.add_argument("--vocab-size", type=int, default=5000)
    p_bench.add_argument("--queries", type=int, default=200)
    p_bench.add_argument("--prompt-terms", type=int, default=3)
    p_bench.add_argument("--seed", type=int, default=1)
    p_bench.add_argument(
        "--storage-limit-bytes",
        type=int,
        default=0,
        help="Override do storage_limit_bytes (ex.: 0 para desabilitar)",
    )

    args = parser.parse_args(argv)

    # Permite `--db/--debug` em qualquer posição (extração manual acima).
    if extracted.db is not None:
        args.db = extracted.db
    if extracted.debug:
        args.debug = True

    engine = SynapseEngine(args.db, debug=args.debug)
    try:
        if args.cmd == "init":
            engine.rebuild_index()
            print("ok")
            return 0

        if args.cmd == "add":
            content = _read_text(args.content, args.content_file)
            mem_id = engine.add_memory(path=args.path, title=args.title, content=content)
            print(mem_id)
            return 0

        if args.cmd == "update":
            content = _read_text(args.content, args.content_file) if (args.content or args.content_file) else None
            engine.update_memory(args.id, content=content, path=args.path, title=args.title)
            print("ok")
            return 0

        if args.cmd == "show":
            obj = engine.show_memory(args.id)
            print(json.dumps(obj, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "search":
            prompt = _read_text(args.prompt, args.prompt_file)
            res = engine.search(prompt)
            if args.debug:
                res.context.setdefault("debug", {})["timings_ms"] = res.timings_ms
            print(json.dumps(res.context, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "mem":
            out = engine.mem(args.prefix)
            if out["type"] == "categories":
                for item in out["items"]:
                    print(item)
            else:
                for m in out["items"]:
                    title = (m.get("title") or "").strip()
                    if title:
                        title = f" — {title}"
                    print(f"{m['id']}  {m['status']}  {m['token_count']}t{title}")
            return 0

        if args.cmd == "anonymous":
            engine.set_anonymous(args.state == "on")
            print("ok")
            return 0

        if args.cmd == "rebuild":
            engine.rebuild_index()
            print("ok")
            return 0

        if args.cmd == "optimize":
            engine.optimize(vacuum=not args.no_vacuum)
            print("ok")
            return 0

        if args.cmd == "gc":
            stats = engine.gc()
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "ingest":
            session_id = engine.ingest(prompt=args.prompt, response=args.response)
            print(session_id)
            return 0

        if args.cmd == "consolidate":
            ids = engine.consolidate(path=args.path, title=args.title)
            print(",".join(ids))
            return 0

        if args.cmd == "ref":
            if args.ref_cmd == "add":
                engine.add_reference(args.from_id, args.to_id)
                print("ok")
                return 0
            if args.ref_cmd == "list":
                refs = engine.list_references(args.from_id)
                for r in refs:
                    print(r)
                return 0
            if args.ref_cmd == "clear":
                deleted = engine.clear_references(args.from_id)
                print(deleted)
                return 0

        if args.cmd == "bench":
            rep = run_bench(
                engine,
                memories=int(args.memories),
                tokens_per_memory=int(args.tokens_per_memory),
                vocab_size=int(args.vocab_size),
                queries=int(args.queries),
                prompt_terms=int(args.prompt_terms),
                seed=int(args.seed),
                storage_limit_bytes=args.storage_limit_bytes,
            )
            print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
            return 0

        parser.error("comando inválido")
        return 2
    except SynapseError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    finally:
        engine.close()


def _read_text(text_arg: str | None, file_arg: str | None) -> str:
    if text_arg is not None:
        return text_arg
    if file_arg is not None:
        return Path(file_arg).read_text(encoding="utf-8")
    raise AssertionError("expected one of text_arg/file_arg")


class _Globals(argparse.Namespace):
    db: str | None
    debug: bool
    argv: list[str]


def _extract_globals(argv: list[str]) -> _Globals:
    db: str | None = None
    debug = False
    out: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--debug":
            debug = True
            i += 1
            continue
        if arg.startswith("--db="):
            db = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--db":
            if i + 1 >= len(argv):
                raise SystemExit("erro: --db requer um valor")
            db = argv[i + 1]
            i += 2
            continue
        out.append(arg)
        i += 1

    g = _Globals()
    g.db = db
    g.debug = debug
    g.argv = out
    return g

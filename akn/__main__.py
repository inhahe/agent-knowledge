"""Command line: `python -m akn <command>`.

    scan      parse transcripts into windows (no model; fast, incremental)
    extract   run the local LLM over pending windows
    report    embed, cluster, consolidate, check coverage, write the report
    all       scan + extract + report
    nightly   the same, unattended: locks, skips if the GPU is busy, logs to state/
    doccards  summary cards + embeddings for docs_repo's docsearch (shared cache)
    curate    the morning curator (--run), or its mechanical steps (--prepare, --finalize)
    curated   the curated sections: --show, --log, --undo, --drop TEXT
    stats     what the store holds
    lessons   print extracted lessons (with --show-window, the excerpt behind each)
    sample    print some windows (for tuning the signal heuristics)
    server    start/stop a model server by hand (for interactive use)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from . import config as config_mod
from .store import Store, text_of


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_scan(cfg, store, args) -> None:
    from . import scan

    _print_json(scan.run(cfg, store, force=args.force, verbose=args.verbose))


def cmd_stats(cfg, store, args) -> None:
    _print_json(store.stats())


def cmd_sample(cfg, store, args) -> None:
    q = "SELECT * FROM windows WHERE status != 'duplicate'"
    params: list = []
    if args.kind:
        q += " AND kinds LIKE ?"
        params.append(f"%{args.kind}%")
    rows = store.db.execute(q, params).fetchall()
    random.seed(args.seed)
    for row in random.sample(rows, min(args.n, len(rows))):
        print("=" * 100)
        print(f"id={row['id']} score={row['score']} kinds={row['kinds']} chars={row['n_chars']} status={row['status']}")
        text = text_of(row["text"], row["text_z"])
        print(text if len(text) <= args.chars else text[: args.chars] + f"\n...[{len(text) - args.chars} more chars]")


def cmd_lessons(cfg, store, args) -> None:
    """Print extracted lessons, newest windows first, optionally with the excerpt they came from."""
    q = ("SELECT l.*, w.kinds, w.score, w.text AS wtext, w.text_z AS wtext_z FROM lessons l JOIN windows w ON w.id = l.window_id"
         " WHERE l.confidence >= ? ORDER BY w.extracted_at DESC, l.id LIMIT ?")
    for r in store.db.execute(q, (args.min_conf, args.n)).fetchall():
        print("=" * 100)
        print(f"[{r['kind']} | {r['scope']} | {r['source']} | conf {r['confidence']:.2f}] {r['title']}")
        print(f"  {r['lesson']}")
        print(f"  when: {r['trigger']}")
        print(f"  evidence: {r['evidence'][:300]}")
        print(f"  from: {r['account']}/{r['project']} session {r['session_id'][:8]} {r['ts'][:16]}"
              f" (window {r['window_id']}, score {r['score']}, {r['kinds']})")
        if args.show_window:
            print("-" * 40)
            print(text_of(r["wtext"], r["wtext_z"])[: args.chars])


def cmd_extract(cfg, store, args) -> None:
    from . import extract
    from .llm import Client, LlamaServer, llm_spec

    with LlamaServer(llm_spec(cfg), cfg.state_dir, keep=args.keep_server) as srv:
        client = Client(srv.base_url)
        _print_json(extract.run(store, client, cfg.llm_model.name, workers=cfg.llm_parallel,
                                thinking=cfg.llm_thinking, limit=args.limit, retry_failed=args.retry_failed))


def cmd_report(cfg, store, args) -> None:
    from . import report

    path = report.run(cfg, store, keep_server=args.keep_server, top=args.top,
                      recluster=getattr(args, "recluster", False))
    print(f"report written: {path}")


def cmd_all(cfg, store, args) -> None:
    cmd_scan(cfg, store, args)
    cmd_extract(cfg, store, args)
    cmd_report(cfg, store, args)


def cmd_doccards(cfg, store, args) -> None:
    from . import doccards

    repo = cfg.docs_repo
    if args.repo:
        # As typed at a prompt (`--repo .`), from the current folder; a name only the
        # projects folder holds, as docs_repo is read.
        given = Path(args.repo).expanduser()
        # abspath, not resolve: on a drive that is a network share, resolve gives the share's UNC path.
        repo = Path(os.path.abspath(given)) if given.is_absolute() or given.exists() else cfg.in_projects(given)
    if repo is None:
        sys.exit("doccards: no repository: set docs_repo in config.toml, or pass --repo")
    _print_json(doccards.run(cfg, repo, cards=not args.no_cards, vectors=not args.no_vectors, limit=args.limit,
                             keep_server=args.keep_server, cpu_embed=args.cpu_embed))


def cmd_curate(cfg, store, args) -> int:
    """Exit code 1 when the operator has something to see to: the run failed or refused
    to run, worked from stale lessons (`stale_report`), found a section it could not
    write, or kept one whose draft failed its checks. Task Scheduler then shows the run
    as failed, rather than as one more quiet success."""
    from . import curate

    out = curate.run(cfg) if args.run else curate.prepare(cfg) if args.prepare else curate.finalize(cfg)
    _print_json(out)
    return 1 if any(out.get(k) for k in ("error", "warning", "blocked", "problems")) else 0


def cmd_curated(cfg, store, args) -> None:
    from . import curate

    if args.drop:
        _print_json(curate.drop(cfg, args.drop))
    elif args.undo:
        _print_json(curate.undo(cfg))
    elif args.log:
        print(curate.show_log(cfg))
    else:
        stale = curate.stale_report(cfg)
        if stale:
            print(f"warning: {stale}\n")
        for key in curate.sections(cfg):
            print(f"=== {key} ===")
            print(curate.canonical(cfg, key) or "(no rules yet)")
            print()
        for d in curate.differing_copies(cfg):
            print(f"note: the copy in {d.split(': ', 1)[1]} differs from this ({d.split(':', 1)[0]}): edited by hand, "
                  "or not yet written; the next run reconciles it")


def cmd_nightly(cfg, store, args) -> None:
    from . import nightly

    sys.exit(nightly.run(cfg, store))


def cmd_server(cfg, store, args) -> None:
    from .llm import LlamaServer, embed_spec, llm_spec

    spec = llm_spec(cfg) if args.which == "llm" else embed_spec(cfg)
    if args.which == "llm":
        # Interactive use: keep llama.cpp's built-in chat page, which batch runs turn off.
        spec.args = [a for a in spec.args if a != "--no-webui"]
    srv = LlamaServer(spec, cfg.state_dir, keep=True)
    srv.start()
    pid = srv.proc.pid if srv.proc else "reused"
    print(f"{args.which} server on {srv.base_url} (pid {pid}); it keeps running after this command exits.")
    if args.which == "llm":
        print(f"Chat in a browser at {srv.base_url}/ ; OpenAI-compatible API at {srv.base_url}/v1")
    print(f"Stop it with: taskkill /PID {pid}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akn", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=str, default=None, help="path to a config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan")
    p.add_argument("--force", action="store_true", help="re-parse every file, not just changed ones")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("stats")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("sample")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--kind", default="")
    p.add_argument("--chars", type=int, default=6000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("lessons")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--show-window", action="store_true", help="also print the excerpt each lesson came from")
    p.add_argument("--chars", type=int, default=4000)
    p.set_defaults(func=cmd_lessons)

    for name, func in (("extract", cmd_extract), ("all", cmd_all)):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=None, help="extract at most N windows (highest score first)")
        p.add_argument("--retry-failed", action="store_true")
        p.add_argument("--keep-server", action="store_true", help="leave the model server running afterwards")
        p.add_argument("--force", action="store_true")
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--top", type=int, default=120)
        p.set_defaults(func=func)

    p = sub.add_parser("report")
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--top", type=int, default=120, help="clusters to consolidate and check")
    p.add_argument("--recluster", action="store_true",
                   help="discard the kept clusters and rebuild them from every lesson (in bounded batches)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("server")
    p.add_argument("which", choices=["llm", "embed"])
    p.set_defaults(func=cmd_server)

    p = sub.add_parser("doccards", help="summary cards + embeddings for a project's shared documents")
    p.add_argument("--repo", default="", help="default: docs_repo. A path that exists from the current folder "
                   "is taken from there; otherwise it is a folder in the projects folder")
    p.add_argument("--no-cards", action="store_true")
    p.add_argument("--no-vectors", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="make at most N cards")
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--cpu-embed", action="store_true", help="run the embedding model on the CPU")
    p.set_defaults(func=cmd_doccards)

    p = sub.add_parser("curate", help="the morning curator (the 06:00 task runs --run)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true", help="write state/curator/input.md from the newest report")
    g.add_argument("--finalize", action="store_true", help="check the curator's draft and write the sections")
    g.add_argument("--run", action="store_true", help="prepare, run the Opus curator, finalize (the 06:00 task)")
    p.set_defaults(func=cmd_curate)

    p = sub.add_parser("curated", help="the curated sections of the CLAUDE.md files: show, log, undo, drop")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true", help="what the curator changed, newest first")
    g.add_argument("--undo", action="store_true", help="take back the newest curator run, keeping changes made since")
    g.add_argument("--drop", metavar="TEXT", help="remove the one rule containing TEXT, and never re-add it")
    p.set_defaults(func=cmd_curated)

    p = sub.add_parser("nightly", help="unattended scan+extract+report with safety checks (Task Scheduler)")
    p.set_defaults(func=cmd_nightly)

    args = ap.parse_args(argv)
    try:
        cfg = config_mod.load(None if args.config is None else Path(args.config))
    except Exception:
        _log_config_error(args.cmd)
        raise
    store = Store(cfg.db_path)
    try:
        return args.func(cfg, store, args) or 0
    finally:
        store.close()


def _log_config_error(cmd: str) -> None:
    """Record why the settings could not be loaded in state/config-error.log. The
    scheduled runs use pythonw, which has no stderr, and they log to files named by
    the very settings that failed to load: without this a typo in config.toml
    stops them without a trace."""
    import traceback
    from datetime import datetime

    try:
        state = config_mod.ROOT / "state"
        state.mkdir(exist_ok=True)
        with open(state / "config-error.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} `akn {cmd}`: {traceback.format_exc()}\n")
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())

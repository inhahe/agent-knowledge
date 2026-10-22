"""Summary cards and embeddings for a repository's shared documents, for its own docsearch.

It serves one kind of repository: one whose `scripts/docsearch.py` searches its
shared documents by words with SQLite FTS5 everywhere, and by meaning when a
shared cache holds embeddings, and which also shows a one-line *card* under each
hit (`docs_repo` names it; with none set, nothing here runs). Producing those
needs the local models, which live here, so this module fills that cache:

* a **card** per substantive entry (issues, decisions, questions, requests, spec
  passages): a one-sentence summary, the status the text states, a few subject
  tags, and two or three questions the entry answers -- the extra phrasings that
  let a keyword search find an entry by what someone would ask;
* an **embedding** per entry of every kind (Qwen3-Embedding, the same model the
  query side uses), of the entry's heading and the start of its body.

Both are keyed by the SHA-1 of the entry's text, so a card is made once per
distinct text however many worktrees hold it, and an edited entry gets a new one.
(Entries are byte-identical in both document layouts, so a worktree that has not
merged the per-entry cutover shares its cards and vectors with one that has.)
The cache is `<git-common-dir>/docsearch-cache.sqlite`, shared by every worktree.
Vectors are made before cards: they take minutes and cards take hours on a first
run, and docsearch uses meaning only once ~90% of the entries it searches have a
vector.
The repository's own `doc_entries.py` and `docsearch.py` are imported from its
`scripts/` directory, so the definitions of an entry and of the cache's schema
are the repository's, not a copy kept here.

A card is a summary for a search result, labelled as such where it is shown. It is
never written into a document.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import gitread
from .llm import Client

CARD_KINDS = ("issue", "decision", "question", "deferred", "request", "spec")
STATUSES = ["open", "fixed", "closed", "wontfix", "superseded", "decided", "answered", "deferred", "unknown"]
CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "status": {"type": "string", "enum": STATUSES},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["summary", "status", "tags", "questions"],
    "additionalProperties": False,
}
CARD_SYSTEM = """You write index cards for entries in a software project's internal documents{project}. A card helps someone searching the documents decide whether this entry is the one they need.

summary: one sentence, at most 30 words, saying what the entry is about and what it concludes. Plain words; keep exact names of programs, files, functions and commands.
status: what the text itself says about its state -- open (still a problem / not done), fixed, closed, wontfix, superseded, decided (a decision record), answered, deferred, or unknown if the text does not say. Judge from the whole text: later updates override the heading.
tags: 2 to 6 short lower-case subject tags{tags}.
questions: 2 or 3 short questions a developer might ask that this entry answers, phrased the way they would ask them."""


def card_system(project: str = "", tags: list[str] | None = None) -> str:
    """The card writer's instructions. `project` says what the project and its documents
    are (`docs_project`), `tags` are example subject tags (`docs_tags`); either may be
    empty, leaving the instructions general."""
    return CARD_SYSTEM.format(project=f" ({project})" if project else "",
                              tags=" (e.g. " + ", ".join(f'"{t}"' for t in tags) + ")" if tags else "")


def _load_repo_modules(repo: Path):
    scripts = repo / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import doc_entries  # noqa: PLC0415 - from the repository, deliberately
    import docsearch  # noqa: PLC0415
    return doc_entries, docsearch


def entries_of(repo: Path, worktrees: list[Path]):
    """Distinct entries (by text hash) across the given worktrees of `repo`."""
    return read_entries(repo, worktrees)[0]


def read_entries(repo: Path, worktrees: list[Path]) -> tuple[list, bool]:
    """Distinct entries (by text hash) across the given worktrees, and whether
    every worktree could be read -- only then is it safe to treat a cached card
    whose text appears nowhere as stale."""
    D, _S = _load_repo_modules(repo)
    seen: dict[str, object] = {}
    complete = bool(worktrees)
    for wt in worktrees:
        try:
            for e in D.read_tree(wt):
                seen.setdefault(e.text_hash, e)
        except Exception as exc:  # noqa: BLE001 - one unreadable worktree must not stop the rest
            print(f"  skipping {wt}: {exc}", flush=True)
            complete = False
    return list(seen.values()), complete


def prune_cache(repo: Path, live_hashes: set[str]) -> dict[str, int]:
    """Delete cards and vectors for texts no worktree holds any more (an entry
    that was edited, or a branch that was merged and removed). Without this the
    shared cache only ever grows: every edit of an entry adds a card and a vector.
    A text that comes back (a branch checked out again) is simply carded again."""
    _D, S = _load_repo_modules(repo)
    cache = S.open_cache(repo)
    cache.execute("CREATE TEMP TABLE IF NOT EXISTS live (h TEXT PRIMARY KEY)")
    cache.execute("DELETE FROM live")
    cache.executemany("INSERT OR IGNORE INTO live VALUES (?)", [(h,) for h in live_hashes])
    cards = cache.execute("DELETE FROM cards WHERE text_hash NOT IN (SELECT h FROM live)").rowcount
    vectors = cache.execute("DELETE FROM vectors WHERE text_hash NOT IN (SELECT h FROM live)").rowcount
    cache.commit()
    return {"stale_cards_deleted": cards, "stale_vectors_deleted": vectors}


def worktrees_of(repo: Path) -> list[Path]:
    import subprocess
    out = subprocess.run(["git", "-C", str(repo), "worktree", "list", "--porcelain"], capture_output=True, text=True,
                         env=gitread.clean_env())
    return [Path(ln[len("worktree "):]) for ln in out.stdout.splitlines() if ln.startswith("worktree ")]


def card_input(text: str) -> str:
    if len(text) <= 6000:
        return text
    return text[:4500] + "\n\n[... middle of the entry omitted ...]\n\n" + text[-1500:]


def make_cards(repo: Path, client: Client, entries, workers: int, limit: int | None = None,
               system: str | None = None) -> dict:
    """Make the missing cards. `system` is the card writer's instructions (`card_system`)."""
    system = system or card_system()
    _D, S = _load_repo_modules(repo)
    cache = S.open_cache(repo)
    have = {r[0] for r in cache.execute("SELECT text_hash FROM cards")}
    todo = [e for e in entries if e.kind in CARD_KINDS and e.text_hash not in have]
    todo.sort(key=lambda e: ("issue", "decision", "question", "deferred", "request", "spec").index(e.kind))
    if limit:
        todo = todo[:limit]
    print(f"cards: {len(todo)} to make ({len(have)} cached)", flush=True)
    done = failed = 0
    t0 = last = time.time()

    def one(e):
        res = client.chat_json(system, f"Entry ({e.kind}, {e.path}):\n\n{card_input(e.text)}", CARD_SCHEMA,
                               max_tokens=400, temperature=0.2, thinking=False)
        return e, res.data

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, e) for e in todo]
        for fut in as_completed(futures):
            try:
                e, card = fut.result()
                cache.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?)",
                              (e.text_hash, "qwen3.8-27b", card.get("summary", ""), card.get("status", "unknown"),
                               json.dumps(card.get("tags", [])), json.dumps(card.get("questions", [])),
                               datetime.now(timezone.utc).isoformat(timespec="seconds")))
                done += 1
            except Exception as exc:  # noqa: BLE001 - count and continue
                failed += 1
                print(f"  card failed: {exc!s:.200}", flush=True)
            if done % 25 == 0 or time.time() - last > 60:
                cache.commit()
                last = time.time()
                rate = done / (last - t0) if last > t0 else 0
                print(f"  {done + failed}/{len(todo)} cards ({failed} failed), "
                      f"~{(len(todo) - done - failed) / rate / 60 if rate else 0:.0f} min left", flush=True)
    cache.execute("INSERT OR REPLACE INTO meta VALUES ('cards_updated', ?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"),))
    cache.commit()
    return {"cards_made": done, "cards_failed": failed, "seconds": round(time.time() - t0)}


def make_vectors(repo: Path, client: Client, entries, batch: int = 32) -> dict:
    _D, S = _load_repo_modules(repo)
    cache = S.open_cache(repo)
    model = S.EMBED_MODEL_NAME
    have = {r[0] for r in cache.execute("SELECT text_hash FROM vectors WHERE model=?", (model,))}
    todo = [e for e in entries if e.text_hash not in have]
    print(f"vectors: {len(todo)} to embed ({len(have)} cached)", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), 256):
        chunk = todo[i : i + 256]
        vecs = client.embed([S.embedding_input(e.text) for e in chunk], batch=batch)
        cache.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                          [(e.text_hash, model, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
                           for e, v in zip(chunk, vecs)])
        cache.commit()
        print(f"  {min(i + 256, len(todo))}/{len(todo)} embedded", flush=True)
    return {"vectors_made": len(todo), "seconds": round(time.time() - t0)}


def tools_checkout(repo: Path) -> Path | None:
    """A checkout of `repo` whose scripts/ holds doc_entries.py and docsearch.py.

    `repo` itself first; failing that, any of its worktrees. A checkout that lags
    its trunk may predate the tools while the other worktrees have them. All
    worktrees share one git directory, hence one cache, so it does not matter which
    checkout's copy of the tools reads them."""
    for cand in [repo, *worktrees_of(repo)]:
        if (cand / "scripts" / "doc_entries.py").is_file() and (cand / "scripts" / "docsearch.py").is_file():
            return cand
    return None


def run(cfg, repo: Path, cards: bool, vectors: bool, limit: int | None = None, keep_server: bool = False,
        cpu_embed: bool = False) -> dict:
    from .llm import LlamaServer, embed_spec, llm_spec

    tools = tools_checkout(repo)
    if tools is None:
        return {"skipped": f"no checkout of {repo} has scripts/docsearch.py yet"}
    repo = tools
    entries, complete = read_entries(repo, worktrees_of(repo))
    print(f"{len(entries)} distinct entries across the worktrees of {repo}", flush=True)
    out: dict = {"entries": len(entries)}
    # Vectors first: minutes, against hours of cards on a first run, and docsearch
    # ranks by meaning only once nearly every entry has one -- so this order makes
    # semantic search available hours earlier, and a run cut short still leaves it.
    if vectors:
        with LlamaServer(embed_spec(cfg, cpu=cpu_embed), cfg.state_dir) as srv:
            out |= make_vectors(repo, Client(srv.base_url), entries)
    if cards:
        with LlamaServer(llm_spec(cfg), cfg.state_dir, keep=keep_server) as srv:
            out |= make_cards(repo, Client(srv.base_url), entries, workers=cfg.llm_parallel, limit=limit,
                              system=card_system(cfg.docs_project, cfg.docs_tags))
    if not complete:
        out["prune"] = "skipped: not every worktree could be read"
    elif not entries:
        out["prune"] = "skipped: no entries read"
    else:
        out |= prune_cache(repo, {e.text_hash for e in entries})
    return out

"""doccards: the order of its two passes, what reaches the shared cache, what is
skipped, and that its rows fit the cache schema the repository actually defines."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from akn import config as config_mod
from akn import doccards, llm

# The schema docsearch.py creates. The contract test at the bottom checks this copy
# against the real one, so a change on that side fails here rather than at night.
CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
  text_hash TEXT PRIMARY KEY, model TEXT, summary TEXT, status TEXT, tags TEXT, questions TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS vectors (
  text_hash TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER, vec BLOB, PRIMARY KEY (text_hash, model)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Entry:
    kind: str
    path: str
    text: str

    @property
    def text_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


class FakeClient:
    """Stands in for the model server: a fixed card, unit vectors, and a failure on demand."""

    def __init__(self) -> None:
        self.chats: list[str] = []
        self.embedded: list[str] = []

    def chat_json(self, system, user, schema, **kw):
        self.chats.append(user)
        if "BOOM" in user:
            raise RuntimeError("the model returned nonsense")
        return types.SimpleNamespace(data={"summary": "what it is", "status": "open", "tags": ["ext4", "posix"],
                                           "questions": ["why?"]})

    def embed(self, texts, batch=32):
        self.embedded.extend(texts)
        return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3)


def fake_repo_modules(monkeypatch, conn: sqlite3.Connection, schema_module=None) -> None:
    S = schema_module or types.SimpleNamespace(EMBED_MODEL_NAME="test-embed", embedding_input=lambda t: t[:50])
    S.open_cache = lambda repo: conn
    monkeypatch.setattr(doccards, "_load_repo_modules", lambda repo: (types.SimpleNamespace(), S))


@pytest.fixture
def cache() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(CACHE_SCHEMA)
    yield conn
    conn.close()


class FakeServer:
    def __init__(self, spec, log_dir, keep=False, reuse=True):
        self.base_url = "http://127.0.0.1:1"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_vectors_are_made_before_cards(monkeypatch, tmp_path):
    # Vectors take minutes and cards hours, and docsearch uses meaning only once
    # nearly every entry has a vector: cards first would hold semantic search back.
    order: list[str] = []
    monkeypatch.setattr(doccards, "tools_checkout", lambda repo: tmp_path)
    monkeypatch.setattr(doccards, "worktrees_of", lambda repo: [tmp_path])
    monkeypatch.setattr(doccards, "read_entries", lambda repo, wts: ([], True))
    monkeypatch.setattr(doccards, "make_vectors", lambda *a, **k: order.append("vectors") or {"vectors_made": 0})
    monkeypatch.setattr(doccards, "make_cards", lambda *a, **k: order.append("cards") or {"cards_made": 0})
    monkeypatch.setattr(llm, "LlamaServer", FakeServer)
    out = doccards.run(config_mod.Config(), tmp_path, cards=True, vectors=True)
    assert order == ["vectors", "cards"]
    assert out["entries"] == 0


def test_each_pass_can_be_turned_off(monkeypatch, tmp_path):
    order: list[str] = []
    monkeypatch.setattr(doccards, "tools_checkout", lambda repo: tmp_path)
    monkeypatch.setattr(doccards, "worktrees_of", lambda repo: [tmp_path])
    monkeypatch.setattr(doccards, "read_entries", lambda repo, wts: ([], True))
    monkeypatch.setattr(doccards, "make_vectors", lambda *a, **k: order.append("vectors") or {})
    monkeypatch.setattr(doccards, "make_cards", lambda *a, **k: order.append("cards") or {})
    monkeypatch.setattr(llm, "LlamaServer", FakeServer)
    doccards.run(config_mod.Config(), tmp_path, cards=False, vectors=True)
    doccards.run(config_mod.Config(), tmp_path, cards=True, vectors=False)
    assert order == ["vectors", "cards"]


def test_a_repository_without_the_tools_is_skipped_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(doccards, "tools_checkout", lambda repo: None)
    out = doccards.run(config_mod.Config(), tmp_path, cards=True, vectors=True)
    assert "skipped" in out


def test_cards_are_made_once_per_text_only_for_card_kinds_and_a_failure_is_counted(monkeypatch, cache, tmp_path):
    cached = Entry("issue", "known-issues/A.md", "## A\nalready carded")
    fresh = Entry("decision", "design-decisions/0001-x.md", "## 1. X\nbody")
    roadmap = Entry("roadmap", "roadmap.md", "- [ ] a task")  # embedded, never carded
    broken = Entry("issue", "known-issues/B.md", "## B\nBOOM")
    cache.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?)",
                  (cached.text_hash, "m", "old summary", "open", "[]", "[]", "2026-10-01T00:00:00+00:00"))
    fake_repo_modules(monkeypatch, cache)
    client = FakeClient()
    out = doccards.make_cards(tmp_path, client, [cached, fresh, roadmap, broken], workers=2)
    assert (out["cards_made"], out["cards_failed"]) == (1, 1)
    assert len(client.chats) == 2 and not any("a task" in c or "already carded" in c for c in client.chats)
    row = cache.execute("SELECT summary, status, tags, questions FROM cards WHERE text_hash=?",
                        (fresh.text_hash,)).fetchone()
    assert row == ("what it is", "open", json.dumps(["ext4", "posix"]), json.dumps(["why?"]))
    assert cache.execute("SELECT summary FROM cards WHERE text_hash=?", (cached.text_hash,)).fetchone() == (
        "old summary",)
    assert cache.execute("SELECT 1 FROM cards WHERE text_hash=?", (broken.text_hash,)).fetchone() is None
    assert cache.execute("SELECT value FROM meta WHERE key='cards_updated'").fetchone() is not None


def test_cards_go_issues_first_and_the_limit_is_honoured(monkeypatch, cache, tmp_path):
    fake_repo_modules(monkeypatch, cache)
    client = FakeClient()
    ents = [Entry("spec", "design.txt", "a spec passage"), Entry("decision", "d.md", "a decision"),
            Entry("issue", "i.md", "an issue")]
    doccards.make_cards(tmp_path, client, ents, workers=1, limit=1)
    assert len(client.chats) == 1 and "an issue" in client.chats[0]


def test_vectors_skip_what_this_model_has_and_store_float32(monkeypatch, cache, tmp_path):
    have = Entry("issue", "a.md", "embedded already")
    other_model = Entry("issue", "b.md", "embedded only by another model")
    new = Entry("roadmap", "roadmap.md", "x" * 200)
    cache.execute("INSERT INTO vectors VALUES (?,?,?,?)", (have.text_hash, "test-embed", 3, b"\0" * 12))
    cache.execute("INSERT INTO vectors VALUES (?,?,?,?)", (other_model.text_hash, "older-model", 3, b"\0" * 12))
    fake_repo_modules(monkeypatch, cache)
    client = FakeClient()
    out = doccards.make_vectors(tmp_path, client, [have, other_model, new])
    assert out["vectors_made"] == 2
    assert client.embedded == ["embedded only by another model", "x" * 50]  # through embedding_input
    dim, blob = cache.execute("SELECT dim, vec FROM vectors WHERE text_hash=? AND model='test-embed'",
                              (new.text_hash,)).fetchone()
    assert dim == 3 and len(blob) == 12
    assert np.allclose(np.frombuffer(blob, dtype=np.float32), 1 / np.sqrt(3))


def test_the_card_writer_is_told_about_the_project_only_what_the_settings_say(monkeypatch, cache, tmp_path):
    general = doccards.card_system()
    assert "internal documents. A card" in general and "(e.g." not in general and "{" not in general
    told = doccards.card_system("a compiler written in OCaml", ["parser", "codegen"])
    assert "internal documents (a compiler written in OCaml). A card" in told
    assert 'subject tags (e.g. "parser", "codegen").' in told

    seen: list[str] = []
    fake_repo_modules(monkeypatch, cache)
    client = FakeClient()
    monkeypatch.setattr(client, "chat_json", lambda system, user, schema, **kw: seen.append(system) or
                        types.SimpleNamespace(data={"summary": "s", "status": "open", "tags": [], "questions": []}))
    doccards.make_cards(tmp_path, client, [Entry("issue", "i.md", "an issue")], workers=1, system=told)
    assert seen == [told]


def test_card_input_keeps_both_ends_of_a_long_entry():
    short = "a" * 6000
    assert doccards.card_input(short) == short
    long = "H" * 4500 + "Z" * 5000 + "T" * 1500
    got = doccards.card_input(long)
    assert got.startswith("H" * 4500) and got.endswith("T" * 1500) and "Z" not in got
    assert "middle of the entry omitted" in got


def test_tools_checkout_prefers_the_repo_then_any_worktree(monkeypatch, tmp_path):
    old, new = tmp_path / "repo", tmp_path / "repo-wt"
    old.mkdir()
    (new / "scripts").mkdir(parents=True)
    for name in ("doc_entries.py", "docsearch.py"):
        (new / "scripts" / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(doccards, "worktrees_of", lambda repo: [old, new])
    assert doccards.tools_checkout(old) == new  # that checkout lags the trunk
    assert doccards.tools_checkout(new) == new
    monkeypatch.setattr(doccards, "worktrees_of", lambda repo: [old])
    assert doccards.tools_checkout(old) is None


def test_rows_fit_the_cache_schema_the_repository_defines(monkeypatch, tmp_path):
    """The contract with the other side: run both passes into a cache built from the
    real docsearch.py's schema, then read a card back the way docsearch does."""
    cfg = config_mod.load()  # docs_repo is this machine's setting, if it has one
    tools = doccards.tools_checkout(cfg.docs_repo) if cfg.docs_repo and cfg.docs_repo.is_dir() else None
    if tools is None:
        pytest.skip(f"no docs_repo checkout has scripts/docsearch.py (docs_repo: {cfg.docs_repo})")
    _D, S = doccards._load_repo_modules(tools)
    assert " ".join(S.CACHE_SCHEMA.split()) == " ".join(CACHE_SCHEMA.split()), \
        "docsearch.py's cache schema changed; update doccards and this test's copy together"
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(S.CACHE_SCHEMA)
    real = types.SimpleNamespace(EMBED_MODEL_NAME=S.EMBED_MODEL_NAME, embedding_input=S.embedding_input)
    fake_repo_modules(monkeypatch, conn, schema_module=real)
    e = Entry("issue", "known-issues/X.md", "## X\nbody")
    doccards.make_vectors(tmp_path, FakeClient(), [e])
    doccards.make_cards(tmp_path, FakeClient(), [e], workers=1)
    card = conn.execute("SELECT summary, status, tags, questions FROM cards").fetchone()
    assert S.card_text(card) == "what it is\next4\nposix\nwhy?"
    assert conn.execute("SELECT model FROM vectors").fetchone() == (S.EMBED_MODEL_NAME,)
    conn.close()


def test_stale_cards_and_vectors_are_pruned_only_when_every_worktree_was_read(monkeypatch, cache, tmp_path):
    keep, stale = Entry("issue", "a.md", "still here"), Entry("issue", "b.md", "edited away")
    for e in (keep, stale):
        cache.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?)", (e.text_hash, "m", "s", "open", "[]", "[]", "t"))
        cache.execute("INSERT INTO vectors VALUES (?,?,?,?)", (e.text_hash, "test-embed", 3, b"\0" * 12))
    cache.commit()
    fake_repo_modules(monkeypatch, cache)
    monkeypatch.setattr(doccards, "tools_checkout", lambda repo: tmp_path)
    monkeypatch.setattr(doccards, "worktrees_of", lambda repo: [tmp_path])
    monkeypatch.setattr(llm, "LlamaServer", FakeServer)
    monkeypatch.setattr(doccards, "make_vectors", lambda *a, **k: {})
    monkeypatch.setattr(doccards, "make_cards", lambda *a, **k: {})

    # A worktree could not be read: the live set is incomplete, so nothing is deleted.
    monkeypatch.setattr(doccards, "read_entries", lambda repo, wts: ([keep], False))
    out = doccards.run(config_mod.Config(), tmp_path, cards=True, vectors=True)
    assert out["prune"].startswith("skipped") and cache.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 2

    # Every worktree read: the text no worktree holds loses its card and vector.
    monkeypatch.setattr(doccards, "read_entries", lambda repo, wts: ([keep], True))
    out = doccards.run(config_mod.Config(), tmp_path, cards=True, vectors=True)
    assert (out["stale_cards_deleted"], out["stale_vectors_deleted"]) == (1, 1)
    assert [r[0] for r in cache.execute("SELECT text_hash FROM cards")] == [keep.text_hash]
    assert [r[0] for r in cache.execute("SELECT text_hash FROM vectors")] == [keep.text_hash]


def test_read_entries_reports_an_unreadable_worktree(monkeypatch, tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"

    def read_tree(wt):
        if wt == bad:
            raise OSError("cannot read")
        return [Entry("issue", "a.md", "x")]

    monkeypatch.setattr(doccards, "_load_repo_modules", lambda repo: (types.SimpleNamespace(read_tree=read_tree), None))
    entries, complete = doccards.read_entries(tmp_path, [good, bad])
    assert len(entries) == 1 and complete is False
    assert doccards.read_entries(tmp_path, [good])[1] is True
    assert doccards.read_entries(tmp_path, [])[1] is False  # no worktree at all is not a complete picture

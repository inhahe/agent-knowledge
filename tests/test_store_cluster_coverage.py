from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from akn import config as config_mod
from akn import extract
from akn.cluster import cluster_lessons
from akn.coverage import chunk_markdown
from akn.llm import Client, ServerError
from akn.store import Store
from akn.windows import Window


def make_window(path: str, session: str, start: int, anchors: list[str], text: str = "body") -> Window:
    return Window(session_id=session, path=path, account="claude", project="p", title="", start_idx=start,
                  end_idx=start + 3, start_ts="2026-09-20T10:00:00Z", end_ts="2026-09-20T10:05:00Z", score=2.0,
                  kinds=["human"], anchor_uuids=anchors, text=text)


# --- store -------------------------------------------------------------------

def test_store_replace_keeps_changes_and_removes(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    p = Path("a.jsonl")
    w1, w2 = make_window("a.jsonl", "s", 0, ["u1"]), make_window("a.jsonl", "s", 10, ["u2"])
    assert st.replace_file_windows(p, [w1, w2])["new"] == 2
    st.finish_window(w1.id, [{"title": "t", "lesson": "l", "confidence": 0.9}], "m", 10, 5, 1.0)
    # unchanged w1 keeps its lesson; changed w2 is reset; a new w3 appears
    w2b = make_window("a.jsonl", "s", 10, ["u2"], text="body grew")
    w3 = make_window("a.jsonl", "s", 20, ["u3"])
    counts = st.replace_file_windows(p, [w1, w2b, w3])
    assert counts == {"kept": 1, "new": 1, "changed": 1, "removed": 0, "duplicate": 0}
    assert len(st.lessons()) == 1
    # dropping w1 removes it and its lesson
    counts = st.replace_file_windows(p, [w2b, w3])
    assert counts["removed"] == 1 and st.lessons() == []


def test_store_marks_copied_history_duplicate(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    st.replace_file_windows(Path("orig.jsonl"), [make_window("orig.jsonl", "s1", 0, ["u1", "u2"])])
    copy = make_window("copy.jsonl", "s2", 0, ["u1", "u2"])
    partial = make_window("copy.jsonl", "s2", 50, ["u2", "u9"])  # one new anchor: not a duplicate
    counts = st.replace_file_windows(Path("copy.jsonl"), [copy, partial])
    assert counts["duplicate"] == 1
    statuses = {r["id"]: r["status"] for r in st.db.execute("SELECT id, status FROM windows")}
    assert statuses[copy.id] == "duplicate" and statuses[partial.id] == "pending"
    assert [r["id"] for r in st.pending_windows()] and copy.id not in [r["id"] for r in st.pending_windows()]


def test_pending_windows_put_operator_windows_first(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    errors = make_window("a.jsonl", "s", 0, ["u1"])
    errors.kinds, errors.score = ["error-streak", "tool-error"], 5.0
    human = make_window("a.jsonl", "s", 10, ["u2"])
    human.kinds, human.score = ["human"], 1.0
    selfc = make_window("a.jsonl", "s", 20, ["u3"])
    selfc.kinds, selfc.score = ["self-correction"], 2.0
    st.replace_file_windows(Path("a.jsonl"), [errors, human, selfc])
    assert [r["id"] for r in st.pending_windows()] == [human.id, errors.id, selfc.id]
    assert [r["id"] for r in st.pending_windows(limit=1)] == [human.id]


def test_store_embedding_cache_roundtrip(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    v = np.arange(4, dtype=np.float32)
    st.put_embeddings({"h1": v}, "m")
    got = st.cached_embeddings({"h1", "h2"}, "m")
    assert list(got) == ["h1"] and np.array_equal(got["h1"], v)
    assert st.cached_embeddings({"h1"}, "other-model") == {}


def test_file_unchanged(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    st.record_file(Path("a"), "c", "p", "s", "", 10, 1.5, 3, 1, "", "")
    assert st.file_unchanged(Path("a"), 10, 1.5)
    assert not st.file_unchanged(Path("a"), 11, 1.5)
    assert not st.file_unchanged(Path("b"), 10, 1.5)


# --- clustering ----------------------------------------------------------------

def _lessons(st: Store, specs):
    w = make_window("x.jsonl", "s", 0, ["u"])
    st.replace_file_windows(Path("x.jsonl"), [w])
    ids = []
    for sess, source, conf in specs:
        st.db.execute(
            "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
            " evidence, source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (w.id, sess, "claude", "p", "2026-09-20T10:00:00Z", "t", "l", "tool-quirk", "this-machine", "w", "e",
             source, conf),
        )
        ids.append(sess)
    st.db.commit()
    return st.lessons()


def test_cluster_groups_by_direction_and_ranks_by_sessions(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    rows = _lessons(st, [("s1", "agent-discovered", 0.8), ("s2", "agent-discovered", 0.8),
                         ("s3", "agent-discovered", 0.8), ("s1", "operator", 0.9), ("s1", "operator", 0.9)])
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    emb = np.vstack([a, a + [0, 0.05, 0], a + [0, 0, 0.05], b, b + [0.05, 0, 0]]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    clusters = cluster_lessons(rows, emb, threshold=0.3)
    assert len(clusters) == 2
    top, second = clusters
    assert top.n_sessions == 3 and not top.operator  # three sessions beat one operator session...
    assert second.n_sessions == 1 and second.operator  # ...even with the operator boost
    assert len(second.members) == 2


def test_cluster_single_lesson(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    rows = _lessons(st, [("s1", "operator", 0.9)])
    clusters = cluster_lessons(rows, np.array([[1.0, 0.0]], dtype=np.float32), 0.3)
    assert len(clusters) == 1 and clusters[0].operator


# --- coverage chunking -----------------------------------------------------------

def test_chunk_markdown_splits_bullets_under_headings():
    md = (
        "# Rules\n\nIntro paragraph that is long enough to keep as a chunk.\n\n"
        "## Shell\n\n- **Backslashes through Bash.** They get eaten.\n  Continuation line of the same bullet.\n"
        "- Second bullet that is also long enough to keep.\n\n"
        "```\ncode block line\n\n# not a heading inside code\n```\n"
    )
    chunks = chunk_markdown("f.md", md)
    texts = [c.text for c in chunks]
    assert chunks[0].heading == "Rules" and texts[0].startswith("Intro paragraph")
    assert chunks[1].heading == "Rules > Shell" and "Continuation line" in texts[1]
    assert texts[2].startswith("- Second bullet")
    assert any("# not a heading inside code" in t for t in texts)


def test_chunk_markdown_caps_long_sections():
    md = "# H\n\n" + "\n".join(f"line {i} " + "x" * 60 for i in range(200))
    chunks = chunk_markdown("f.md", md)
    assert len(chunks) > 1 and all(len(c.text) <= 1600 for c in chunks)


# --- LLM client & extraction --------------------------------------------------------

def _chat_transport(content: str, finish: str = "stop", status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings":
            body = json.loads(request.content)
            data = [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": data})
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(status, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    return httpx.MockTransport(handler)


def test_client_chat_json_parses_and_reports_usage():
    c = Client("http://x", transport=_chat_transport('{"lessons": []}'))
    res = c.chat_json("s", "u", extract.SCHEMA)
    assert res.data == {"lessons": []} and res.prompt_tokens == 7 and res.completion_tokens == 3


def test_client_chat_json_errors():
    with pytest.raises(ValueError, match="truncated"):
        Client("http://x", transport=_chat_transport('{"lessons": [', finish="length")).chat_json("s", "u", {})
    with pytest.raises(ValueError, match="not JSON"):
        Client("http://x", transport=_chat_transport("nope")).chat_json("s", "u", {})
    with pytest.raises(ServerError):
        Client("http://x", transport=_chat_transport("{}", status=500)).chat_json("s", "u", {})


def test_client_embed_normalises():
    vecs = Client("http://x", transport=_chat_transport("")).embed(["a", "b", "c"], batch=2)
    assert vecs.shape == (3, 2) and np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], [0.6, 0.8])


def test_extract_clean_drops_empty_and_clamps():
    out = extract._clean([{"lesson": "  do X  ", "confidence": 3}, {"lesson": ""}, {"lesson": "y", "confidence": "bad"}])
    assert [o["lesson"] for o in out] == ["do X", "y"]
    assert out[0]["confidence"] == 1.0 and out[1]["confidence"] == 0.0


def test_extract_one_retries_truncated_reply_with_more_tokens():
    calls = []

    class Fake:
        def chat_json(self, system, user, schema, *, max_tokens, temperature, thinking):
            calls.append(max_tokens)
            if len(calls) == 1:
                raise ValueError("reply truncated")
            from akn.llm import ChatResult
            return ChatResult({"lessons": [{"lesson": "do Y", "confidence": 0.5}]}, 100, 20, 1.0, "")

    lessons, pt, ct, _ = extract.extract_one(Fake(), "text", thinking=False)
    assert calls == [1800, 3600] and lessons[0]["lesson"] == "do Y" and (pt, ct) == (100, 20)


def test_schema_puts_evidence_first():
    assert list(extract.LESSON_SCHEMA["properties"])[0] == "evidence"


def test_extract_run_records_lessons_and_failures(tmp_path):
    from akn.llm import ChatResult

    st = Store(tmp_path / "db.sqlite")
    good = make_window("a.jsonl", "s1", 0, ["u1"], text="GOOD window")
    bad = make_window("a.jsonl", "s1", 10, ["u2"], text="BAD window")
    st.replace_file_windows(Path("a.jsonl"), [good, bad])

    class Fake:
        def chat_json(self, system, user, schema, *, max_tokens, temperature, thinking):
            if "BAD" in user:
                raise ServerError("boom")
            return ChatResult({"lessons": [{"evidence": "q", "title": "t", "lesson": "do Z", "trigger": "w",
                                            "kind": "tool-quirk", "scope": "this-machine", "source": "tool-failure",
                                            "confidence": 0.7}]}, 50, 10, 0.5, "")

    import akn.extract as ex
    real_sleep = ex.time.sleep
    ex.time.sleep = lambda s: None  # the retry backoff is irrelevant here
    try:
        summary = ex.run(st, Fake(), "m", workers=2, thinking=False)
    finally:
        ex.time.sleep = real_sleep
    assert summary["done"] == 1 and summary["failed"] == 1 and summary["lessons"] == 1
    rows = {r["id"]: r for r in st.db.execute("SELECT id, status, error FROM windows")}
    assert rows[good.id]["status"] == "done" and rows[bad.id]["status"] == "failed"
    assert "boom" in rows[bad.id]["error"]
    les = st.lessons()
    assert len(les) == 1 and les[0]["lesson"] == "do Z" and les[0]["session_id"] == "s1"
    # failed windows are retried only on request
    assert [r["id"] for r in st.pending_windows()] == []
    assert [r["id"] for r in st.pending_windows(retry_failed=True)] == [bad.id]


# --- config ------------------------------------------------------------------------

def test_config_override_and_unknown_key(tmp_path):
    good = tmp_path / "c.toml"
    good.write_text('llm_port = 9999\nstate_dir = "X:/s"\naccounts = ["Y:/a"]\n', encoding="utf-8")
    cfg = config_mod.load(good)
    assert cfg.llm_port == 9999 and cfg.state_dir == Path("X:/s") and cfg.accounts == [Path("Y:/a")]
    bad = tmp_path / "b.toml"
    bad.write_text("llm_prot = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="llm_prot"):
        config_mod.load(bad)

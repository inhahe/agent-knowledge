"""What keeps the store and the reports from growing without bound, and the
incremental clustering that keeps a report's memory bounded."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from akn import cluster as cluster_mod
from akn import report as report_mod
from akn.store import Store, pack_text, window_text
from tests.test_store_cluster_coverage import make_window


# --- window text ----------------------------------------------------------------

def test_window_text_is_stored_compressed_and_read_back_exactly(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    body = "[human] do the thing\n[agent] done -- é, 漢字, a backslash \\ and a NUL-free line\n" * 50
    w = make_window("a.jsonl", "s", 0, ["u1"], text=body)
    st.replace_file_windows(Path("a.jsonl"), [w])
    row = st.db.execute("SELECT * FROM windows WHERE id=?", (w.id,)).fetchone()
    assert row["text"] is None and row["text_z"] is not None
    assert len(row["text_z"]) < len(body.encode("utf-8")) / 3
    assert window_text(row) == body
    assert window_text(st.pending_windows()[0]) == body
    assert row["n_chars"] == len(body)


def test_a_duplicate_window_keeps_no_text(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    st.replace_file_windows(Path("orig.jsonl"), [make_window("orig.jsonl", "s1", 0, ["u1"], text="original")])
    copy = make_window("copy.jsonl", "s2", 0, ["u1"], text="original")
    st.replace_file_windows(Path("copy.jsonl"), [copy])
    row = st.db.execute("SELECT * FROM windows WHERE id=?", (copy.id,)).fetchone()
    assert row["status"] == "duplicate" and row["text"] is None and row["text_z"] is None
    assert window_text(row) == ""


OLD_WINDOWS = """
CREATE TABLE windows (
  id TEXT PRIMARY KEY, path TEXT NOT NULL, session_id TEXT, account TEXT, project TEXT, title TEXT,
  start_idx INTEGER, end_idx INTEGER, start_ts TEXT, end_ts TEXT, score REAL, kinds TEXT,
  text TEXT, text_hash TEXT, n_chars INTEGER,
  status TEXT NOT NULL DEFAULT 'pending',
  extracted_at TEXT, model TEXT, error TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL
);
CREATE TABLE emb_cache (text_hash TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER, emb BLOB,
  PRIMARY KEY (text_hash, model));
CREATE TABLE llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created TEXT);
"""


def test_a_store_from_before_compression_is_migrated_once_and_loses_nothing(tmp_path):
    path = tmp_path / "db.sqlite"
    old = sqlite3.connect(path)
    old.executescript(OLD_WINDOWS)
    texts = {f"w{i}": f"excerpt number {i} " * 400 for i in range(1203)}  # more than one batch
    for wid, t in texts.items():
        old.execute("INSERT INTO windows (id, path, text, text_hash, n_chars, status) VALUES (?,?,?,?,?,?)",
                    (wid, "a.jsonl", t, wid, len(t), "duplicate" if wid == "w7" else "done"))
    old.execute("INSERT INTO emb_cache VALUES ('h', 'm', 1, x'00000000')")
    old.execute("INSERT INTO llm_cache VALUES ('k', '{}', '2026-01-01T00:00:00+00:00')")
    old.commit()
    old.close()
    before = path.stat().st_size

    st = Store(path)
    rows = {r["id"]: r for r in st.db.execute("SELECT * FROM windows")}
    assert all(r["text"] is None for r in rows.values())
    assert rows["w7"]["text_z"] is None  # the duplicate's text is dropped
    assert all(window_text(rows[w]) == t for w, t in texts.items() if w != "w7")
    assert st.meta_get("text_compressed") == "1"
    assert path.stat().st_size < before / 2  # VACUUM gave the space back
    # the cache tables gained their last_used column and kept their rows
    assert st.db.execute("SELECT last_used FROM emb_cache").fetchone()[0] is None
    assert st.cache_get("k") == {}
    st.close()
    st = Store(path)  # a second open finds nothing left to do
    assert st.compress_legacy_text() == 0
    st.close()


# --- caches -------------------------------------------------------------------------

def test_caches_keep_what_was_used_recently_and_prune_the_rest(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    st.put_embeddings({"used": np.ones(2, np.float32), "stale": np.ones(2, np.float32)}, "m")
    st.cache_put("consolidate:used", {"a": 1})
    st.cache_put("consolidate:stale", {"a": 2})
    long_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
    st.db.execute("UPDATE emb_cache SET last_used=?", (long_ago,))
    st.db.execute("UPDATE llm_cache SET last_used=?, created=?", (long_ago, long_ago))
    st.db.commit()
    # This run uses one of each: reading marks it as used now.
    assert set(st.cached_embeddings(["used"], "m")) == {"used"}
    assert st.cache_get("consolidate:used") == {"a": 1}
    assert st.prune_caches(14) == {"embeddings_pruned": 1, "answers_pruned": 1}
    assert set(st.cached_embeddings(["used", "stale"], "m")) == {"used"}
    assert st.cache_get("consolidate:stale") is None and st.cache_get("consolidate:used") == {"a": 1}


def test_cached_embeddings_reads_many_hashes_in_chunks(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    st.put_embeddings({f"h{i}": np.full(3, i, np.float32) for i in range(1234)}, "m")
    got = st.cached_embeddings([f"h{i}" for i in range(1300)], "m")
    assert len(got) == 1234 and float(got["h777"][0]) == 777.0


# --- incremental clustering -------------------------------------------------------------

def _add_lessons(st: Store, specs: list[tuple[str, str]]) -> None:
    """specs: (session, topic); the topic word decides the vector (see _vectors)."""
    w = st.db.execute("SELECT id FROM windows").fetchone()
    if w is None:
        win = make_window("x.jsonl", "s", 0, ["u"])
        st.replace_file_windows(Path("x.jsonl"), [win])
        wid = win.id
    else:
        wid = w["id"]
    for sess, topic in specs:
        st.db.execute(
            "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
            " evidence, source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (wid, sess, "claude", "p", "2026-09-30T10:00:00Z", topic, f"about {topic}", "tool-quirk",
             "this-machine", "w", "e", "agent-discovered", 0.8),
        )
    st.db.commit()


AXES = {"alpha": 0, "beta": 1, "gamma": 2}


def _vectors(calls: list[int] | None = None):
    """Each lesson points along its topic's axis, with a little per-id jitter."""
    def vectors_of(rows):
        if calls is not None:
            calls.append(len(rows))
        out = []
        for r in rows:
            v = np.zeros(4)
            v[AXES[r["title"]]] = 1.0
            v[3] = 0.02 * (r["id"] % 5)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out).astype(np.float32)
    return vectors_of


def _groups(members: dict[int, list[int]], st: Store) -> set[tuple[str, int]]:
    titles = {r["id"]: r["title"] for r in st.lessons()}
    out = set()
    for mids in members.values():
        names = {titles[i] for i in mids}
        assert len(names) == 1, f"a cluster mixes topics: {names}"
        out.add((names.pop(), len(mids)))
    return out


def test_join_rule_is_average_linkage_for_one_new_lesson():
    rng = np.random.default_rng(1)
    members = rng.normal(size=(7, 16))
    members /= np.linalg.norm(members, axis=1, keepdims=True)
    e = rng.normal(size=16)
    e /= np.linalg.norm(e)
    mean_pairwise_distance = np.mean([1 - members[i] @ e for i in range(7)])
    assert np.isclose(1 - (members.sum(axis=0) @ e) / 7, mean_pairwise_distance)


def test_update_clusters_builds_then_extends_without_reclustering(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    _add_lessons(st, [("s1", "alpha"), ("s2", "alpha"), ("s3", "beta")])
    calls: list[int] = []
    members = cluster_mod.update_clusters(st, st.lessons(), _vectors(calls), 0.3, batch=100, space="sp")
    assert _groups(members, st) == {("alpha", 2), ("beta", 1)}
    alpha_cid = next(c for c, m in members.items() if len(m) == 2)

    # A later run: two new lessons. Only they are embedded; the alpha one joins
    # the existing cluster (same id), the gamma one starts a cluster of its own.
    _add_lessons(st, [("s4", "alpha"), ("s5", "gamma")])
    calls.clear()
    members = cluster_mod.update_clusters(st, st.lessons(), _vectors(calls), 0.3, batch=100, space="sp")
    assert _groups(members, st) == {("alpha", 3), ("beta", 1), ("gamma", 1)}
    assert len(members[alpha_cid]) == 3
    assert calls[0] == 2  # the new lessons, not all five
    stored = {r["id"]: r for r in st.load_clusters()}
    assert stored[alpha_cid]["n"] == 3

    # Nothing new: nothing is embedded.
    calls.clear()
    cluster_mod.update_clusters(st, st.lessons(), _vectors(calls), 0.3, batch=100, space="sp")
    assert calls == []


def test_update_clusters_follows_lessons_that_disappear(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    _add_lessons(st, [("s1", "alpha"), ("s2", "alpha"), ("s3", "beta")])
    members = cluster_mod.update_clusters(st, st.lessons(), _vectors(), 0.3, batch=100, space="sp")
    ids = {r["title"]: [] for r in st.lessons()}
    for r in st.lessons():
        ids[r["title"]].append(r["id"])
    # one alpha lesson and the only beta lesson are gone (their window changed)
    st.db.execute("DELETE FROM lessons WHERE id IN (?, ?)", (ids["alpha"][0], ids["beta"][0]))
    st.db.commit()
    members = cluster_mod.update_clusters(st, st.lessons(), _vectors(), 0.3, batch=100, space="sp")
    assert _groups(members, st) == {("alpha", 1)}
    assert len(st.load_clusters()) == 1 and st.load_clusters()[0]["n"] == 1
    assert set(st.assignments()) == {ids["alpha"][1]}


def test_small_batches_give_the_same_clusters_as_one_batch(tmp_path):
    specs = [(f"s{i}", ("alpha", "beta", "gamma")[i % 3]) for i in range(30)]
    results = []
    for batch in (1, 4, 1000):
        st = Store(tmp_path / f"db{batch}.sqlite")
        _add_lessons(st, specs)
        results.append(_groups(cluster_mod.update_clusters(st, st.lessons(), _vectors(), 0.3, batch, "sp"), st))
        st.close()
    assert results[0] == results[1] == results[2] == {("alpha", 10), ("beta", 10), ("gamma", 10)}


def test_a_new_embedding_space_discards_the_clusters(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    _add_lessons(st, [("s1", "alpha"), ("s2", "beta")])
    cluster_mod.update_clusters(st, st.lessons(), _vectors(), 0.3, batch=100, space="old-model")
    calls: list[int] = []
    cluster_mod.update_clusters(st, st.lessons(), _vectors(calls), 0.3, batch=100, space="new-model")
    assert calls == [2]  # everything placed again, in the new space
    assert st.meta_get("cluster_space") == "new-model"


def test_ranked_clusters_rank_by_sessions(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    _add_lessons(st, [("s1", "alpha"), ("s2", "alpha"), ("s3", "alpha"), ("s1", "beta")])
    members = cluster_mod.update_clusters(st, st.lessons(), _vectors(), 0.3, batch=100, space="sp")
    ranked = cluster_mod.ranked_clusters(st, st.lessons(), members)
    assert [c.n_sessions for c in ranked] == [3, 1] and ranked[0].rep["title"] == "alpha"


# --- reports ----------------------------------------------------------------------------

def test_old_reports_beyond_the_newest_n_are_deleted(tmp_path):
    for day in range(1, 36):
        for ext in (".md", ".json"):
            (tmp_path / f"lessons-2026-09-{day:02d}-0430{ext}").write_text("x", encoding="utf-8")
    (tmp_path / "latest.md").write_text("x", encoding="utf-8")
    assert report_mod._prune_reports(tmp_path, 30) == 10
    left = sorted(p.name for p in tmp_path.iterdir())
    assert "latest.md" in left and len(left) == 61
    assert "lessons-2026-09-05-0430.md" not in left and "lessons-2026-09-06-0430.md" in left


def test_clusters_that_grow_together_are_merged_as_average_linkage_would(tmp_path):
    """alpha and beta start just too far apart (cosine distance 0.35). Lessons in
    between join alpha and pull its average towards beta, to 0.203: clustering
    everything at once makes one cluster, and so must the incremental path."""
    angle = {"alpha": 0.0, "beta": np.degrees(np.arccos(0.65)), "gamma": 20.0}

    def vectors_of(rows):
        return np.vstack([[np.cos(np.radians(angle[r["title"]])), np.sin(np.radians(angle[r["title"]]))]
                          for r in rows]).astype(np.float32)

    st = Store(tmp_path / "db.sqlite")
    _add_lessons(st, [("s1", "alpha"), ("s2", "alpha"), ("s3", "beta"), ("s4", "beta")]
                 + [(f"s{5 + i}", "gamma") for i in range(4)])
    rows = st.lessons()
    full = cluster_mod.cluster_lessons(rows, vectors_of(rows), 0.3)
    assert len(full) == 1  # the reference: one cluster
    members = cluster_mod.update_clusters(st, rows, vectors_of, 0.3, batch=4, space="sp")
    assert len(members) == 1 and len(next(iter(members.values()))) == 8
    assert len(st.load_clusters()) == 1 and st.load_clusters()[0]["n"] == 8
    assert len(set(st.assignments().values())) == 1

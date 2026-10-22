"""SQLite state: which transcripts were scanned, the windows found, and the lessons extracted.

The store makes runs incremental. A transcript is re-parsed only when its size or
mtime changed; a window is re-sent to the model only when its rendered text
changed (a window at the end of a live session grows as the session continues);
an embedding is computed once per distinct text.

Duplicate history. When a session is copied (the D: -> E: migration copied
sessions wholesale), the same events appear in two files with the same `uuid`s.
`anchors` records which window first claimed each signal event; a window whose
every signal event is already claimed by a window in another file is marked
`duplicate` and not extracted, so a copied incident is not counted twice.

Size. Everything the store keeps either is bounded or grows only with new
transcript material, and as slowly as it can (measured 2026-10-02: excerpt text
was 128 of the database's 146 MB, growing ~2 MB a night):

* a window's text is kept zlib-compressed (`text_z`, ~3.2x smaller on real
  excerpts) -- it is the evidence behind each lesson and outlives transcripts
  Claude Code may delete -- and a duplicate window keeps no text at all, since
  it is never extracted;
* cached embeddings and model answers record when they were last used, and
  `prune_caches` deletes those unused for a while (an answer for a cluster that
  has since changed can never be asked for again);
* clusters are kept between runs (`clusters`, `lesson_cluster`), so a report
  clusters only the new lessons -- see cluster.py.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .windows import Window

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY, account TEXT, project TEXT, session_id TEXT, title TEXT,
  size INTEGER, mtime REAL, n_events INTEGER, n_signals INTEGER,
  first_ts TEXT, last_ts TEXT, scanned_at TEXT
);
CREATE TABLE IF NOT EXISTS windows (
  id TEXT PRIMARY KEY, path TEXT NOT NULL, session_id TEXT, account TEXT, project TEXT, title TEXT,
  start_idx INTEGER, end_idx INTEGER, start_ts TEXT, end_ts TEXT, score REAL, kinds TEXT,
  text TEXT, text_hash TEXT, n_chars INTEGER,
  status TEXT NOT NULL DEFAULT 'pending',
  extracted_at TEXT, model TEXT, error TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL,
  text_z BLOB
);
CREATE INDEX IF NOT EXISTS windows_path ON windows(path);
CREATE INDEX IF NOT EXISTS windows_status ON windows(status);
CREATE TABLE IF NOT EXISTS anchors (
  uuid TEXT PRIMARY KEY, window_id TEXT NOT NULL, path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS anchors_window ON anchors(window_id);
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY AUTOINCREMENT, window_id TEXT NOT NULL,
  session_id TEXT, account TEXT, project TEXT, ts TEXT,
  title TEXT, lesson TEXT, kind TEXT, scope TEXT, trigger TEXT,
  evidence TEXT, source TEXT, confidence REAL
);
CREATE INDEX IF NOT EXISTS lessons_window ON lessons(window_id);
CREATE TABLE IF NOT EXISTS emb_cache (
  text_hash TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER, emb BLOB, last_used TEXT,
  PRIMARY KEY (text_hash, model)
);
CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, created TEXT, last_used TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY AUTOINCREMENT, n INTEGER NOT NULL, dim INTEGER NOT NULL,
  sum BLOB NOT NULL, rep_id INTEGER, created TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS lesson_cluster (
  lesson_id INTEGER PRIMARY KEY, cluster_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS lesson_cluster_cluster ON lesson_cluster(cluster_id);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT
);
"""

# Columns added after the first release, created on stores that predate them.
_ADDED_COLUMNS = {
    "windows": [("text_z", "BLOB")],
    "emb_cache": [("last_used", "TEXT")],
    "llm_cache": [("last_used", "TEXT")],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pack_text(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8"), 6)


def text_of(text: str | None, text_z: bytes | None) -> str:
    """A window's text from its stored columns: compressed, or (in a store not yet
    migrated) plain, or empty for a duplicate, which keeps none."""
    if text_z is not None:
        return zlib.decompress(text_z).decode("utf-8")
    return text or ""


def window_text(row: sqlite3.Row) -> str:
    return text_of(row["text"], row["text_z"])


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def close(self) -> None:
        self.db.close()

    # --- schema migration -----------------------------------------------------------
    def _migrate(self) -> None:
        for table, cols in _ADDED_COLUMNS.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols:
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.db.commit()
        if self.meta_get("text_compressed") != "1":
            self.compress_legacy_text()

    def compress_legacy_text(self, batch: int = 500) -> int:
        """Move plain window text into `text_z` (compressed), dropping it for
        duplicates. Commits per batch, so an interrupted run resumes where it
        stopped; afterwards the file is rewritten (VACUUM) to give the space back.
        Returns the number of windows migrated."""
        done = 0
        while True:
            rows = self.db.execute("SELECT id, status, text FROM windows WHERE text IS NOT NULL LIMIT ?",
                                   (batch,)).fetchall()
            if not rows:
                break
            self.db.executemany(
                "UPDATE windows SET text_z=?, text=NULL WHERE id=?",
                [(None if r["status"] == "duplicate" else pack_text(r["text"]), r["id"]) for r in rows],
            )
            self.db.commit()
            done += len(rows)
        self.meta_put("text_compressed", "1")
        if done:
            self.vacuum()
        return done

    def vacuum(self) -> bool:
        """Rewrite the file so freed pages are returned to the disk. Needs every
        other connection to be idle; if one is not, the space is simply reused by
        later writes instead, so failing here is harmless."""
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")
            # In WAL mode the rewritten pages land in the log; the main file only
            # shrinks once they are checkpointed back into it.
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return True
        except sqlite3.OperationalError:
            return False

    # --- files -----------------------------------------------------------------
    def file_unchanged(self, path: Path, size: int, mtime: float) -> bool:
        row = self.db.execute("SELECT size, mtime FROM files WHERE path=?", (str(path),)).fetchone()
        return row is not None and row["size"] == size and abs(row["mtime"] - mtime) < 1e-6

    def record_file(self, path: Path, account: str, project: str, session_id: str, title: str,
                    size: int, mtime: float, n_events: int, n_signals: int, first_ts: str, last_ts: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(path), account, project, session_id, title, size, mtime, n_events, n_signals, first_ts, last_ts, now()),
        )

    # --- windows ---------------------------------------------------------------
    def replace_file_windows(self, path: Path, windows: list[Window]) -> dict[str, int]:
        """Make the stored windows for `path` match `windows`, keeping finished
        extractions whose text did not change. Returns counts by outcome."""
        p = str(path)
        counts = {"kept": 0, "new": 0, "changed": 0, "removed": 0, "duplicate": 0}
        cur = self.db.cursor()
        existing = {r["id"]: r for r in cur.execute("SELECT id, text_hash, status FROM windows WHERE path=?", (p,))}
        new_ids = {w.id for w in windows}
        for wid in set(existing) - new_ids:
            cur.execute("DELETE FROM lessons WHERE window_id=?", (wid,))
            cur.execute("DELETE FROM anchors WHERE window_id=?", (wid,))
            cur.execute("DELETE FROM windows WHERE id=?", (wid,))
            counts["removed"] += 1
        for w in windows:
            old = existing.get(w.id)
            if old is not None and old["text_hash"] == w.text_hash:
                counts["kept"] += 1
                continue
            if old is not None:
                cur.execute("DELETE FROM lessons WHERE window_id=?", (w.id,))
                cur.execute("DELETE FROM windows WHERE id=?", (w.id,))
                counts["changed"] += 1
            else:
                counts["new"] += 1
            status = "pending"
            owners = [
                cur.execute("SELECT path FROM anchors WHERE uuid=?", (u,)).fetchone() for u in w.anchor_uuids if u
            ]
            if owners and all(o is not None and o["path"] != p for o in owners):
                status = "duplicate"
                counts["duplicate"] += 1
            cur.execute(
                "INSERT INTO windows (id, path, session_id, account, project, title, start_idx, end_idx, start_ts, end_ts,"
                " score, kinds, text, text_z, text_hash, n_chars, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (w.id, p, w.session_id, w.account, w.project, w.title, w.start_idx, w.end_idx, w.start_ts, w.end_ts,
                 w.score, ",".join(w.kinds), None, None if status == "duplicate" else pack_text(w.text),
                 w.text_hash, len(w.text), status),
            )
            for u in w.anchor_uuids:
                if u:
                    cur.execute("INSERT OR IGNORE INTO anchors VALUES (?,?,?)", (u, w.id, p))
        self.db.commit()
        return counts

    def pending_windows(self, limit: int | None = None, retry_failed: bool = False) -> list[sqlite3.Row]:
        """Windows still to extract, most valuable first: anything the operator said
        or did (a prompt, a correction, an interrupt) before anything else, then by
        score. A run that is stopped early has then covered the richest material.
        Read each one's text with `window_text(row)`."""
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        q = (
            f"SELECT * FROM windows WHERE status IN ({','.join('?' * len(statuses))}) "
            "ORDER BY (kinds LIKE '%human%' OR kinds LIKE '%interrupt%') DESC, score DESC, start_ts"
        )
        if limit:
            q += f" LIMIT {int(limit)}"
        return self.db.execute(q, statuses).fetchall()

    def finish_window(self, wid: str, lessons: list[dict], model: str, prompt_tokens: int,
                      completion_tokens: int, seconds: float) -> None:
        row = self.db.execute("SELECT session_id, account, project, end_ts FROM windows WHERE id=?", (wid,)).fetchone()
        if row is None:
            return  # the window was replaced by a rescan while the model worked on it
        cur = self.db.cursor()
        cur.execute("DELETE FROM lessons WHERE window_id=?", (wid,))
        for les in lessons:
            cur.execute(
                "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
                " evidence, source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (wid, row["session_id"], row["account"], row["project"], row["end_ts"], les.get("title", ""),
                 les.get("lesson", ""), les.get("kind", ""), les.get("scope", ""), les.get("trigger", ""),
                 les.get("evidence", ""), les.get("source", ""), float(les.get("confidence", 0) or 0)),
            )
        cur.execute(
            "UPDATE windows SET status='done', extracted_at=?, model=?, error=NULL, prompt_tokens=?,"
            " completion_tokens=?, seconds=? WHERE id=?",
            (now(), model, prompt_tokens, completion_tokens, seconds, wid),
        )
        self.db.commit()

    def fail_window(self, wid: str, error: str) -> None:
        self.db.execute("UPDATE windows SET status='failed', error=? WHERE id=?", (error[:2000], wid))
        self.db.commit()

    # --- lessons & embeddings --------------------------------------------------
    def lessons(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM lessons ORDER BY ts, id").fetchall()

    def cached_embeddings(self, hashes: Iterable[str], model: str) -> dict[str, np.ndarray]:
        """Cached vectors for `hashes`, marking each one found as used now."""
        out: dict[str, np.ndarray] = {}
        hashes = list(dict.fromkeys(hashes))
        for i in range(0, len(hashes), 500):
            chunk = hashes[i : i + 500]
            q = f"SELECT text_hash, emb FROM emb_cache WHERE model=? AND text_hash IN ({','.join('?' * len(chunk))})"
            for row in self.db.execute(q, (model, *chunk)):
                out[row["text_hash"]] = np.frombuffer(row["emb"], dtype=np.float32)
        if out:
            stamp = now()
            self.db.executemany("UPDATE emb_cache SET last_used=? WHERE text_hash=? AND model=?",
                                [(stamp, h, model) for h in out])
            self.db.commit()
        return out

    def put_embeddings(self, items: dict[str, np.ndarray], model: str) -> None:
        stamp = now()
        self.db.executemany(
            "INSERT OR REPLACE INTO emb_cache (text_hash, model, dim, emb, last_used) VALUES (?,?,?,?,?)",
            [(h, model, int(v.shape[0]), v.astype(np.float32).tobytes(), stamp) for h, v in items.items()],
        )
        self.db.commit()

    # --- cached model answers and small state -------------------------------------
    def cache_get(self, key: str) -> dict | None:
        """A model answer stored under `key` (a hash of everything the answer depends
        on), so an unchanged cluster is not consolidated again on the next run."""
        row = self.db.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        self.db.execute("UPDATE llm_cache SET last_used=? WHERE key=?", (now(), key))
        self.db.commit()
        return json.loads(row["value"])

    def cache_put(self, key: str, value: dict) -> None:
        stamp = now()
        self.db.execute("INSERT OR REPLACE INTO llm_cache (key, value, created, last_used) VALUES (?,?,?,?)",
                        (key, json.dumps(value, ensure_ascii=False), stamp, stamp))
        self.db.commit()

    def prune_caches(self, max_age_days: int) -> dict[str, int]:
        """Delete cached embeddings and model answers not used in `max_age_days`.
        Run at the end of a report, after it has touched everything it uses."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
        emb = self.db.execute("DELETE FROM emb_cache WHERE COALESCE(last_used, '') < ?", (cutoff,)).rowcount
        llm = self.db.execute("DELETE FROM llm_cache WHERE COALESCE(last_used, created, '') < ?", (cutoff,)).rowcount
        self.db.commit()
        return {"embeddings_pruned": emb, "answers_pruned": llm}

    def meta_get(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def meta_put(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
        self.db.commit()

    # --- clusters kept between runs ------------------------------------------------
    def reset_clusters(self) -> None:
        self.db.execute("DELETE FROM lesson_cluster")
        self.db.execute("DELETE FROM clusters")
        self.db.commit()

    def load_clusters(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT id, n, dim, sum, rep_id FROM clusters ORDER BY id").fetchall()

    def assignments(self) -> dict[int, int]:
        return {r[0]: r[1] for r in self.db.execute("SELECT lesson_id, cluster_id FROM lesson_cluster")}

    def save_cluster(self, cid: int | None, n: int, total: np.ndarray, rep_id: int | None) -> int:
        """Insert (cid None) or update a cluster's member count, vector sum and
        representative; returns its id. The caller commits."""
        blob, dim, stamp = total.astype(np.float32).tobytes(), int(total.shape[0]), now()
        if cid is None:
            cur = self.db.execute("INSERT INTO clusters (n, dim, sum, rep_id, created, updated) VALUES (?,?,?,?,?,?)",
                                  (n, dim, blob, rep_id, stamp, stamp))
            return int(cur.lastrowid)
        self.db.execute("UPDATE clusters SET n=?, dim=?, sum=?, rep_id=?, updated=? WHERE id=?",
                        (n, dim, blob, rep_id, stamp, cid))
        return cid

    def delete_cluster(self, cid: int) -> None:
        self.db.execute("DELETE FROM lesson_cluster WHERE cluster_id=?", (cid,))
        self.db.execute("DELETE FROM clusters WHERE id=?", (cid,))

    def assign(self, pairs: Iterable[tuple[int, int]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO lesson_cluster (lesson_id, cluster_id) VALUES (?,?)", list(pairs))

    def unassign(self, lesson_ids: Iterable[int]) -> None:
        self.db.executemany("DELETE FROM lesson_cluster WHERE lesson_id=?", [(i,) for i in lesson_ids])

    # --- stats -------------------------------------------------------------------
    def stats(self) -> dict:
        q = self.db.execute
        by_status = {r[0]: (r[1], r[2] or 0) for r in q("SELECT status, COUNT(*), SUM(n_chars) FROM windows GROUP BY status")}
        size = sum(p.stat().st_size for p in (self.path, Path(str(self.path) + "-wal")) if p.exists())
        return {
            "files": q("SELECT COUNT(*) FROM files").fetchone()[0],
            "windows_by_status": {k: {"count": v[0], "chars": v[1]} for k, v in by_status.items()},
            "lessons": q("SELECT COUNT(*) FROM lessons").fetchone()[0],
            "clusters": q("SELECT COUNT(*) FROM clusters").fetchone()[0],
            "tokens": dict(q("SELECT COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(completion_tokens),0) c,"
                             " COALESCE(SUM(seconds),0) s FROM windows WHERE status='done'").fetchone()),
            "store_mb": round(size / 1e6, 1),
            "stored_text_mb": round((q("SELECT COALESCE(SUM(LENGTH(text_z)),0) + COALESCE(SUM(LENGTH(text)),0)"
                                       " FROM windows").fetchone()[0]) / 1e6, 1),
            "cached_embeddings": q("SELECT COUNT(*) FROM emb_cache").fetchone()[0],
            "cached_answers": q("SELECT COUNT(*) FROM llm_cache").fetchone()[0],
        }


def dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

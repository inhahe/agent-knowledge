"""Group lessons that give the same advice, and rank the groups.

A lesson seen once may be an accident; one that several sessions rediscovered
independently is exactly what an instruction file should carry. Lessons are
embedded (Qwen3-Embedding, with an instruction tuned for "same advice"),
clustered by average-linkage agglomerative clustering on cosine distance, and
each cluster is ranked by how many *distinct sessions* produced it (not how
many lessons: one session can restate a lesson several times), boosted when the
operator was the source and when it happened recently.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .llm import Client, qwen_query
from .store import Store

LESSON_INSTRUCTION = (
    "Represent this lesson learned by an AI coding agent so that lessons giving the same advice are close together"
)


def lesson_text(row: sqlite3.Row) -> str:
    return f"{row['title']}. {row['lesson']} (When: {row['trigger']})"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_texts(store: Store, client: Client, model: str, texts: list[str], instruction: str | None) -> np.ndarray:
    """Embed `texts` (with an optional Qwen query instruction), using the cache."""
    inputs = [qwen_query(instruction, t) if instruction else t for t in texts]
    hashes = [text_hash(t) for t in inputs]
    cached = store.cached_embeddings(set(hashes), model)
    missing = sorted({h: t for h, t in zip(hashes, inputs) if h not in cached}.items())
    if missing:
        vecs = client.embed([t for _, t in missing])
        new = {h: v for (h, _), v in zip(missing, vecs)}
        store.put_embeddings(new, model)
        cached.update(new)
    return np.vstack([cached[h] for h in hashes]) if hashes else np.zeros((0, 0), dtype=np.float32)


@dataclass
class Cluster:
    cid: int
    members: list[sqlite3.Row]
    centroid: np.ndarray
    rep: sqlite3.Row
    sessions: set[str] = field(default_factory=set)
    accounts: set[str] = field(default_factory=set)
    projects: set[str] = field(default_factory=set)
    first_ts: str = ""
    last_ts: str = ""
    operator: bool = False
    score: float = 0.0

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


def _days_ago(ts: str, ref: datetime) -> float:
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return 999.0
    return (ref - t).total_seconds() / 86400


def _labels(emb: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage agglomerative labels on cosine distance: merge while the
    mean pairwise distance between two groups is below `threshold`. Builds an
    n x n distance matrix, so callers pass bounded batches (see update_clusters)."""
    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    ).fit_predict(emb)


def _finish(cid: int, members: list[sqlite3.Row], centroid: np.ndarray, rep: sqlite3.Row,
            ref: datetime) -> Cluster:
    """A Cluster with its sessions, span, operator flag and score filled in."""
    c = Cluster(cid=int(cid), members=members, centroid=centroid, rep=rep)
    c.sessions = {m["session_id"] for m in members}
    c.accounts = {m["account"] for m in members}
    c.projects = {m["project"] for m in members}
    stamps = sorted(m["ts"] for m in members if m["ts"])
    c.first_ts, c.last_ts = (stamps[0], stamps[-1]) if stamps else ("", "")
    c.operator = any(m["source"] == "operator" or str(m["kind"]).startswith("operator") for m in members)
    conf = float(np.mean([m["confidence"] or 0 for m in members]))
    recency = 1.0 if _days_ago(c.last_ts, ref) <= 21 else 0.7
    breadth = 1.0 + 0.15 * (len(c.accounts) - 1) + 0.15 * (len(c.projects) - 1)
    c.score = round(c.n_sessions * (1.6 if c.operator else 1.0) * (0.5 + conf) * recency * breadth, 3)
    return c


def cluster_lessons(rows: list[sqlite3.Row], emb: np.ndarray, threshold: float) -> list[Cluster]:
    """Cluster lessons whose embeddings lie within `threshold` cosine distance
    (average linkage), all at once. Memory is quadratic in len(rows): fine for a
    batch, not for every lesson ever extracted -- reports use update_clusters."""
    if len(rows) == 0:
        return []
    labels = _labels(emb, threshold)
    ref = datetime.now(timezone.utc)
    clusters: list[Cluster] = []
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        centroid = emb[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        rep_i = idx[int(np.argmax(emb[idx] @ centroid))]
        clusters.append(_finish(int(cid), [rows[i] for i in idx], centroid, rows[rep_i], ref))
    clusters.sort(key=lambda c: c.score, reverse=True)
    return clusters


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _rep(lesson_ids: list[int], vecs: np.ndarray) -> int:
    """The member whose vector points most nearly along the cluster's mean."""
    return lesson_ids[int(np.argmax(vecs @ vecs.sum(axis=0)))]


def _merge_close(counts: np.ndarray, sums: np.ndarray, active: np.ndarray, changed: set[int],
                 threshold: float) -> list[tuple[int, int]]:
    """Merge, closest pair first, every pair of active clusters that involves a
    changed one and lies below `threshold` in average-linkage distance -- which,
    for clusters of unit vectors kept as (count, sum), is exactly
    1 - (S_a . S_b) / (n_a n_b). Two unchanged clusters cannot have moved closer,
    so only rows for changed clusters are kept: memory O(changed), each merge
    O(clusters x dim). Updates the arrays in place; returns (absorbed, kept)
    index pairs in merge order. Without this pass, clusters that grow towards
    each other over several runs would stay split, and a lesson's recurrence --
    what the report ranks by -- would be undercounted."""
    def row(i: int) -> np.ndarray:
        sims = (sums @ sums[i]) / (counts * counts[i])
        sims[~active] = -np.inf
        sims[i] = -np.inf
        return sims

    best_val: dict[int, float] = {}
    best_idx: dict[int, int] = {}

    def refresh(i: int) -> None:
        r = row(i)
        j = int(np.argmax(r))
        best_val[i], best_idx[i] = float(r[j]), j

    for i in sorted(changed):
        if active[i]:
            refresh(i)
    merges: list[tuple[int, int]] = []
    while best_val:
        i = max(best_val, key=lambda k: (best_val[k], -k))
        if not np.isfinite(best_val[i]) or 1.0 - best_val[i] >= threshold:
            break
        j = best_idx[i]
        # The larger cluster absorbs the smaller (its id and history carry on).
        keep, gone = (i, j) if (counts[i], -i) >= (counts[j], -j) else (j, i)
        sums[keep] += sums[gone]
        counts[keep] += counts[gone]
        active[gone] = False
        merges.append((gone, keep))
        changed.add(keep)
        best_val.pop(gone, None)
        best_idx.pop(gone, None)
        for r in list(best_val):
            if r == keep or best_idx[r] in (gone, keep):
                refresh(r)
            else:
                s = float(sums[r] @ sums[keep]) / (counts[r] * counts[keep])
                if s > best_val[r]:
                    best_val[r], best_idx[r] = s, keep
        if keep not in best_val:
            refresh(keep)
    return merges


def update_clusters(store: Store, rows: list[sqlite3.Row], vectors_of, threshold: float, batch: int,
                    space: str) -> dict[int, list[int]]:
    """Bring the clusters kept in the store up to date with `rows` (every current
    lesson) and return cluster id -> member lesson ids.

    WHY. Clustering every lesson at once needs an n x n distance matrix: ~8,700
    lessons after the first backlog, ~200 more a night, so tens of GB within
    months. Clusters are therefore kept between runs, and a run touches only:

    * lessons that disappeared (their window changed or was dropped): removed,
      and their clusters' vector sums recomputed from the remaining members;
    * new lessons, in batches of at most `batch`: each joins the existing
      cluster whose members it is closest to on average, if that average cosine
      distance is below `threshold`; the rest are clustered among themselves.

    The join test is exactly average linkage's merge test for one new lesson,
    because for unit vectors the mean cosine similarity to a cluster's members
    is the dot product with the members' mean -- which is why a cluster is
    stored as its member count and vector sum. What full re-clustering would add
    is merging two existing clusters that drifted together; that is the price of
    bounded memory, and `report --recluster` rebuilds from scratch (still in
    batches) when wanted.

    `vectors_of(rows)` returns one embedding per row (unit length or not);
    `space` names the embedding space -- a different one discards the clusters."""
    if store.meta_get("cluster_space") != space:
        store.reset_clusters()
        store.meta_put("cluster_space", space)
    live = {r["id"]: r for r in rows}
    assign = store.assignments()
    stored = {r["id"]: r for r in store.load_clusters()}

    # 1. Lessons that are gone, and assignments to clusters that are gone.
    gone = [lid for lid, cid in assign.items() if lid not in live or cid not in stored]
    affected = {assign[lid] for lid in gone if assign[lid] in stored}
    store.unassign(gone)
    for lid in gone:
        del assign[lid]
    members: dict[int, list[int]] = {}
    for lid, cid in assign.items():
        members.setdefault(cid, []).append(lid)
    for cid in stored:
        if cid not in members:
            affected.add(cid)  # no member left at all
    for cid in sorted(affected):
        mids = sorted(members.get(cid, []))
        if not mids:
            store.delete_cluster(cid)
            stored.pop(cid, None)
            members.pop(cid, None)
            continue
        vecs = _unit(vectors_of([live[i] for i in mids]))
        store.save_cluster(cid, len(mids), vecs.sum(axis=0), _rep(mids, vecs))
    store.db.commit()

    # 2. The clusters as arrays: ids, member counts, vector sums, still-alive flags.
    crows = store.load_clusters()
    cids = [r["id"] for r in crows]
    dim = int(crows[0]["dim"]) if crows else 0
    counts = np.array([r["n"] for r in crows], dtype=np.float64)
    sums = (np.vstack([np.frombuffer(r["sum"], dtype=np.float32) for r in crows]).astype(np.float64)
            if crows else np.zeros((0, dim)))
    active = np.ones(len(cids), dtype=bool)
    members = {cid: set(m) for cid, m in members.items()}
    grown: set[int] = set()

    def merge_pass(changed: set[int]) -> None:
        nonlocal grown
        for gone, keep in _merge_close(counts, sums, active, changed, threshold):
            moved = members.pop(cids[gone], set())
            store.assign((lid, cids[keep]) for lid in moved)
            store.delete_cluster(cids[gone])
            members.setdefault(cids[keep], set()).update(moved)
            for lid in moved:
                assign[lid] = cids[keep]
            grown.discard(gone)
            grown.add(keep)

    # Clusters that lost members have moved too: they may now sit close to another.
    merge_pass({cids.index(c) for c in affected if c in stored and c in cids})
    store.db.commit()

    # 3. New lessons, oldest first, a bounded batch at a time.
    new_ids = sorted((lid for lid in live if lid not in assign), key=lambda i: (live[i]["ts"] or "", i))
    for start in range(0, len(new_ids), max(1, batch)):
        chunk = new_ids[start : start + max(1, batch)]
        vecs = _unit(vectors_of([live[i] for i in chunk]))
        if sums.shape[0] == 0:
            sums = np.zeros((0, vecs.shape[1]))
        joined, rest, changed = [], [], set()
        for k, lid in enumerate(chunk):
            if active.any():
                sims = (sums @ vecs[k]) / np.where(counts > 0, counts, 1.0)  # mean cosine similarity to members
                sims[~active] = -np.inf
                j = int(np.argmax(sims))
                if 1.0 - sims[j] < threshold:
                    sums[j] += vecs[k]
                    counts[j] += 1
                    grown.add(j)
                    changed.add(j)
                    assign[lid] = cids[j]
                    members.setdefault(cids[j], set()).add(lid)
                    joined.append((lid, cids[j]))
                    continue
            rest.append(k)
        store.assign(joined)
        if rest:
            labels = _labels(vecs[rest], threshold)
            new_sums, new_counts = [], []
            for lab in np.unique(labels):
                idx = [rest[i] for i in np.flatnonzero(labels == lab)]
                ids = [chunk[i] for i in idx]
                total = vecs[idx].sum(axis=0)
                cid = store.save_cluster(None, len(idx), total, _rep(ids, vecs[idx]))
                store.assign((lid, cid) for lid in ids)
                for lid in ids:
                    assign[lid] = cid
                members[cid] = set(ids)
                changed.add(len(cids))
                cids.append(cid)
                new_sums.append(total)
                new_counts.append(len(idx))
            sums = np.vstack([sums, *new_sums])
            counts = np.concatenate([counts, np.array(new_counts, dtype=np.float64)])
            active = np.concatenate([active, np.ones(len(new_sums), dtype=bool)])
        merge_pass(changed)
        store.db.commit()

    # 4. Clusters that gained members: store their sums, and pick a representative again.
    for j in sorted(grown):
        if not active[j]:
            continue
        mids = sorted(members[cids[j]])
        vecs = _unit(vectors_of([live[i] for i in mids]))
        store.save_cluster(cids[j], int(counts[j]), sums[j], _rep(mids, vecs))
    store.db.commit()

    return {cid: sorted(m) for cid, m in members.items() if m}


def ranked_clusters(store: Store, rows: list[sqlite3.Row], members: dict[int, list[int]]) -> list[Cluster]:
    """Cluster objects for `members` (from update_clusters), best first."""
    live = {r["id"]: r for r in rows}
    stored = {r["id"]: r for r in store.load_clusters()}
    ref = datetime.now(timezone.utc)
    out = []
    for cid, mids in members.items():
        total = np.frombuffer(stored[cid]["sum"], dtype=np.float32).astype(np.float64)
        centroid = total / (np.linalg.norm(total) or 1.0)
        rep = live.get(stored[cid]["rep_id"]) or live[mids[0]]
        out.append(_finish(cid, [live[i] for i in mids], centroid, rep, ref))
    out.sort(key=lambda c: c.score, reverse=True)
    return out

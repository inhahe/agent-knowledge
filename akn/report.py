"""Embed, cluster, consolidate, check coverage, and write the report.

Phases, each with only one model server loaded at a time (the 27B model leaves
no room in VRAM for a second server):

1. Embedding server: bring the clusters kept in the store up to date -- only
   new lessons are embedded and placed (cluster.update_clusters), so memory does
   not grow with every lesson ever extracted -- then embed the top clusters'
   members as queries against the instruction files, and the instruction
   chunks. Embeddings are cached in the store, so re-runs embed only what is new.
2. LLM server: for the top clusters, write one canonical lesson from the members,
   then judge it against its nearest instruction chunks.
3. Write `reports/lessons-<timestamp>.md` (for a human) and `.json` (the
   consolidated clusters in full, the rest summarised), keep the newest
   `reports_keep` of them, and prune cache entries unused for
   `cache_max_age_days`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .cluster import LESSON_INSTRUCTION, Cluster, embed_texts, lesson_text, ranked_clusters, text_hash, update_clusters
from .config import Config
from .coverage import COVERAGE_SCHEMA, COVERAGE_SYSTEM, Chunk, coverage_prompt, load_chunks
from .curate import settled
from .extract import KINDS, SCOPES
from .llm import Client, LlamaServer, embed_spec, llm_spec
from .mdtext import md_inline, md_quote
from .store import Store

COVERAGE_INSTRUCTION = (
    "Given a lesson learned by an AI coding agent, retrieve existing agent instructions that already give the same advice"
)
CLUSTER_THRESHOLD = 0.30  # cosine distance; tuned by reading clusters on real data
DESTINATIONS = [
    "global-instructions",
    "workspace-instructions",
    "project-instructions",
    "project-known-issues",
    "not-worth-recording",
]

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "lesson": {"type": "string"},
        "applies_when": {"type": "string"},
        "kind": {"type": "string", "enum": KINDS},
        "scope": {"type": "string", "enum": SCOPES},
        "destination": {"type": "string", "enum": DESTINATIONS},
        "consistent": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["title", "lesson", "applies_when", "kind", "scope", "destination", "consistent", "notes"],
    "additionalProperties": False,
}

CONSOLIDATE_SYSTEM = """You are given lessons that were extracted independently from different work sessions of AI coding agents on one Windows 11 machine. They were grouped because they seem to give the same advice. Write the single best version of that advice for an instruction file the agents read at the start of every session.

Rules:
- lesson: 2-4 sentences, imperative, concrete (name the tool, command, path or situation), and self-contained. Keep the operator's intent exactly when the lessons come from the operator.
- If the grouped lessons do not actually agree, set consistent=false, explain in notes, and write the lesson for the majority view.
- destination: where such a rule belongs.
  global-instructions = applies to any project on this machine (shell and OS quirks, tool behaviour, the operator's general preferences);
  workspace-instructions = applies to all projects in the operator's projects folder but is not about the machine itself;
  project-instructions = only one project's agents need it;
  project-known-issues = it describes a bug or limitation of one project rather than a rule for agents;
  not-worth-recording = trivial, obvious, already standard practice, or too specific to recur.
- notes: anything a human reviewer should know (contradictions, doubts); empty if none."""


def _consolidate_prompt(c: Cluster) -> str:
    lines = [
        f"{len(c.members)} lessons from {c.n_sessions} sessions, projects: {', '.join(sorted(c.projects))}",
        "",
    ]
    for i, m in enumerate(sorted(c.members, key=lambda m: -(m["confidence"] or 0))[:12]):
        lines.append(
            f"[{i}] ({m['project']}, {str(m['ts'])[:10]}, source: {m['source']})\n"
            f"title: {m['title']}\nlesson: {m['lesson']}\nwhen: {m['trigger']}\nevidence: {str(m['evidence'])[:400]}"
        )
    return "\n\n".join(lines)


def _md_quote(text: str, limit: int = 500) -> str:
    return md_quote(text, limit)


def _when(ts: str) -> str:
    return str(ts)[:10] if ts else "?"


def run(cfg: Config, store: Store, keep_server: bool = False, top: int = 120, recluster: bool = False) -> Path:
    t0 = time.time()
    rows = store.lessons()
    if not rows:
        raise SystemExit("no lessons in the store yet: run `python -m akn extract` first")
    embed_model = cfg.embed_model.name
    chunks: list[Chunk] = load_chunks(cfg)
    print(f"{len(rows)} lessons, {len(chunks)} instruction chunks", flush=True)
    if recluster:
        store.reset_clusters()

    # 1. clusters (only new lessons are placed), then the query embeddings the
    # coverage check needs: the top clusters' members and the instruction chunks.
    with LlamaServer(embed_spec(cfg), cfg.state_dir) as srv:
        client = Client(srv.base_url)

        def lesson_vectors(rs) -> np.ndarray:
            return embed_texts(store, client, embed_model, [lesson_text(r) for r in rs], LESSON_INSTRUCTION)

        members = update_clusters(store, rows, lesson_vectors, CLUSTER_THRESHOLD, cfg.cluster_batch,
                                  space=f"{embed_model}|{text_hash(LESSON_INSTRUCTION)}")
        clusters = ranked_clusters(store, rows, members)
        eligible = [
            c for c in clusters
            if c.n_sessions >= 2 or (c.operator and float(np.mean([m["confidence"] or 0 for m in c.members])) >= 0.6)
        ]
        candidates, set_aside = select_open(eligible, settled(cfg), top)
        cand_rows = list({m["id"]: m for c in candidates for m in c.members}.values())
        qvecs = embed_texts(store, client, embed_model, [lesson_text(r) for r in cand_rows], COVERAGE_INSTRUCTION)
        qemb = {r["id"]: qvecs[i] for i, r in enumerate(cand_rows)}
        cemb = embed_texts(store, client, embed_model, [c.text for c in chunks], None) if chunks else np.zeros((0, 1))
    print(f"{len(clusters)} clusters; consolidating the top {len(candidates)} "
          f"({set_aside} the curator has decided on are left out until they recur)", flush=True)

    # 3. consolidation + coverage. Answers are cached under a hash of everything
    # they depend on (the member lessons; the canonical lesson and the rules it
    # was compared with), so a nightly run asks the model only about clusters
    # that changed, and starts the LLM server only if it has to.
    results = []
    stack = contextlib.ExitStack()
    client: Client | None = None
    calls = hits = 0

    def model() -> Client:
        nonlocal client
        if client is None:
            srv = stack.enter_context(LlamaServer(llm_spec(cfg), cfg.state_dir, keep=keep_server))
            client = Client(srv.base_url)
        return client

    with stack:
        for n, c in enumerate(candidates, 1):
            entry: dict = {"cluster": c}
            ckey = "consolidate:" + _digest(MODEL_TAG, cfg.llm_model.name, *sorted(str(m["id"]) for m in c.members))
            canon = store.cache_get(ckey)
            if canon is None:
                calls += 1
                try:
                    canon = model().chat_json(CONSOLIDATE_SYSTEM, _consolidate_prompt(c), CONSOLIDATE_SCHEMA,
                                              max_tokens=1200, temperature=0.2, thinking=cfg.llm_thinking).data
                    store.cache_put(ckey, canon)
                except Exception as exc:  # noqa: BLE001 - one cluster failing must not lose the report
                    canon = {"title": c.rep["title"], "lesson": c.rep["lesson"], "applies_when": c.rep["trigger"],
                             "kind": c.rep["kind"], "scope": c.rep["scope"], "destination": "project-instructions",
                             "consistent": True, "notes": f"consolidation failed: {exc}"}
            else:
                hits += 1
            entry["canon"] = canon
            cov = {"verdict": "not-checked", "best_rule": -1, "missing": "", "reasoning": "", "rule": None}
            if canon.get("destination") != "not-worth-recording" and len(chunks):
                q = np.mean([qemb[m["id"]] for m in c.members], axis=0)
                q /= np.linalg.norm(q) or 1.0
                near = np.argsort(-(cemb @ q))[:6]
                near_chunks = [chunks[i] for i in near]
                vkey = "coverage:" + _digest(MODEL_TAG, cfg.llm_model.name, canon.get("lesson", ""),
                                             *[ch.key for ch in near_chunks])
                cached = store.cache_get(vkey)
                if cached is None:
                    calls += 1
                    try:
                        cached = model().chat_json(COVERAGE_SYSTEM, coverage_prompt(canon["lesson"], near_chunks),
                                                   COVERAGE_SCHEMA, max_tokens=500, temperature=0.1,
                                                   thinking=cfg.llm_thinking).data
                        store.cache_put(vkey, cached)
                    except Exception as exc:  # noqa: BLE001
                        cached = {"verdict": "not-checked", "best_rule": -1, "missing": "", "reasoning": str(exc)}
                else:
                    hits += 1
                cov = dict(cached)
                b = int(cov.get("best_rule", -1))
                cov["rule"] = near_chunks[b] if 0 <= b < len(near_chunks) else None
            entry["coverage"] = cov
            results.append(entry)
            if n % 10 == 0 or n == len(candidates):
                print(f"  consolidated {n}/{len(candidates)} ({calls} model calls, {hits} cached)", flush=True)

    # Clusters first seen since the previous report are marked new in this one.
    previous_latest = store.meta_get("report_latest_lesson_ts")
    for e in results:
        e["new"] = bool(previous_latest) and e["cluster"].first_ts > previous_latest
    latest = max((r["ts"] for r in rows if r["ts"]), default="")

    # 4. write
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    md_path = cfg.reports_dir / f"lessons-{stamp}.md"
    md = _markdown(store, rows, clusters, results, chunks, time.time() - t0, set_aside)
    md_path.write_text(md, encoding="utf-8", newline="\n")
    (cfg.reports_dir / "latest.md").write_text(md, encoding="utf-8", newline="\n")
    (cfg.reports_dir / f"lessons-{stamp}.json").write_text(
        json.dumps(_json(results, clusters), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    if latest:
        store.meta_put("report_latest_lesson_ts", latest)

    # 4. keep what is kept bounded
    removed = _prune_reports(cfg.reports_dir, cfg.reports_keep)
    pruned = store.prune_caches(cfg.cache_max_age_days)
    print(f"kept the newest {cfg.reports_keep} reports ({removed} older files deleted); "
          f"cache: {pruned['embeddings_pruned']} embeddings and {pruned['answers_pruned']} answers unused for "
          f"{cfg.cache_max_age_days} days deleted", flush=True)
    return md_path


def _prune_reports(reports_dir: Path, keep: int) -> int:
    """Delete dated reports beyond the newest `keep` (both the .md and the .json;
    latest.md is never touched). Names sort by date: lessons-YYYY-MM-DD-HHMM."""
    stamps = sorted({p.stem for p in reports_dir.glob("lessons-*.md")} | {p.stem for p in reports_dir.glob("lessons-*.json")},
                    reverse=True)
    removed = 0
    for stem in stamps[max(0, keep):]:
        for ext in (".md", ".json"):
            p = reports_dir / f"{stem}{ext}"
            if p.exists():
                p.unlink()
                removed += 1
    return removed


# Bump when a prompt or schema above changes, so cached answers are not reused.
MODEL_TAG = "v1"


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _cluster_meta(c: Cluster) -> str:
    return (
        f"{c.n_sessions} session{'s' if c.n_sessions != 1 else ''} · {len(c.members)} mentions · "
        f"accounts: {', '.join(sorted(c.accounts))} · projects: {', '.join(sorted(c.projects))} · "
        f"seen {_when(c.first_ts)} … {_when(c.last_ts)}" + (" · **operator-sourced**" if c.operator else "")
    )


def _entry_md(i: int, e: dict, show_rule: bool) -> list[str]:
    c: Cluster = e["cluster"]
    canon, cov = e["canon"], e["coverage"]
    new = " **[new since last report]**" if e.get("new") else ""
    out = [f"### {i}. {md_inline(canon.get('title', '').strip())}{new}", ""]
    out.append(md_inline(canon.get("lesson", "").strip()))
    out.append("")
    out.append(f"*When:* {md_inline(canon.get('applies_when', '').strip())}  ")
    out.append(f"*Kind:* {canon.get('kind')} · *scope:* {canon.get('scope')} · *suggested home:* {canon.get('destination')}  ")
    out.append(f"*Evidence base:* {_cluster_meta(c)}  ")
    out.append(f"*Score:* {c.score}")
    if not canon.get("consistent", True) or canon.get("notes"):
        out.append("")
        out.append(f"*Reviewer note:* {md_inline(canon.get('notes', '').strip())}")
    if show_rule and cov.get("rule") is not None:
        rule: Chunk = cov["rule"]
        missing = ": " + md_inline(cov["missing"]) if cov.get("missing") else ""
        out.append("")
        out.append(f"*Closest existing rule* ({Path(rule.source.split(' @ ')[0]).name} › "
                   f"{md_inline(rule.heading or '-')}) — {cov.get('verdict')}{missing}")
        out.append(_md_quote(rule.text, 400))
    # Plain Markdown, not a <details> block: viewers that render HTML one fragment
    # at a time choked on 120 of them (see mdtext.py).
    out.append("")
    out.append("**Evidence:**")
    out.append("")
    for m in sorted(c.members, key=lambda m: -(m["confidence"] or 0))[:4]:
        out.append(f"- {m['account']}/{m['project']} · session {str(m['session_id'])[:8]} · {_when(m['ts'])} · "
                   f"source {m['source']} · conf {m['confidence']:.2f}")
        out.append(f"  {_md_quote(m['evidence'], 350)}")
    out.append("")
    return out


def select_open(eligible: list[Cluster], reopen: dict[int, int], top: int) -> tuple[list[Cluster], int]:
    """The best `top` of `eligible` (sorted best first) that the curator has not
    settled, and how many settled ones were passed over. `reopen` maps a decided
    cluster to the session count at which it is looked at again (curate.settled).

    WHY. Every report ranks the whole history, and ranking barely moves from one
    night to the next: on 2026-10-03 a cluster needed ~5 sessions to enter the top
    120, none of the 120 was under a week old, and ~10 clusters a day were newly
    seen in 2 sessions. Once the curator has decided on the top, ranking alone would
    show it the same decided lessons every morning, and a new lesson only after it
    outranked all of them. Leaving decided clusters out works through the backlog in
    order of strength, 120 a night, and then leaves the places to what is new."""
    open_ = [c for c in eligible if c.n_sessions >= reopen.get(c.cid, 0)]
    return open_[:top], len(eligible) - len(open_)


def _markdown(store: Store, rows, clusters: list[Cluster], results: list[dict], chunks, seconds: float,
              set_aside: int = 0) -> str:
    st = store.stats()
    by_verdict: dict[str, list[dict]] = {}
    for e in results:
        if e["canon"].get("destination") == "not-worth-recording":
            by_verdict.setdefault("skip", []).append(e)
            continue
        verdict = e["coverage"].get("verdict", "not-checked")
        if verdict == "covered":
            recent = e["cluster"].last_ts >= _days_before_latest(rows, 14)
            verdict = "covered-recurring" if recent else "covered-quiet"
        by_verdict.setdefault(verdict, []).append(e)
    w = st["windows_by_status"]
    lines = [
        f"# Lessons mined from Claude Code transcripts — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Candidates for instruction files, not rules. Each was extracted by a local model "
        "from moments where an operator corrected an agent, an agent admitted a mistake, or tools kept failing, "
        "then grouped across sessions and checked against the CLAUDE.md files the agents already load. "
        "The morning curator reads this report and decides which become rules (agent-knowledge's "
        "CURATED-RULES.md); the evidence for each is listed with its session and date.",
        "",
        "## Corpus",
        "",
        f"- transcripts scanned: {st['files']}; windows extracted: {w.get('done', {}).get('count', 0)} "
        f"(pending {w.get('pending', {}).get('count', 0)}, failed {w.get('failed', {}).get('count', 0)}, "
        f"skipped as copied history {w.get('duplicate', {}).get('count', 0)})",
        f"- lessons extracted: {len(rows)}; clusters: {len(clusters)}; consolidated: {len(results)}",
        f"- left out because the curator has decided on them (each returns if it recurs): {set_aside}",
        f"- instruction chunks compared against: {len(chunks)}",
        f"- report built in {seconds / 60:.1f} min",
        "",
    ]
    sections = [
        ("not-covered", "Not in any instruction file", False),
        ("partly-covered", "Partly covered — an existing rule leaves something out", True),
        ("covered-recurring", "Already a rule, yet still recurring in the last 14 days — the rule may not be working", True),
        ("not-checked", "Coverage not checked", False),
        ("covered-quiet", "Already a rule, and not seen recently", True),
    ]
    for key, title, show_rule in sections:
        entries = by_verdict.get(key, [])
        if not entries:
            continue
        lines += [f"## {title} ({len(entries)})", ""]
        for i, e in enumerate(entries, 1):
            lines += _entry_md(i, e, show_rule)
    skipped = by_verdict.get("skip", [])
    if skipped:
        lines += [f"## Judged not worth recording ({len(skipped)})", ""]
        for e in skipped:
            lines.append(f"- {md_inline(e['canon'].get('title', ''))} — {e['cluster'].n_sessions} sessions")
        lines.append("")
    return "\n".join(lines)


def _days_before_latest(rows, days: int) -> str:
    """ISO timestamp `days` before the newest lesson (transcript time, not wall time)."""
    latest = max((r["ts"] for r in rows if r["ts"]), default="")
    if not latest:
        return ""
    t = datetime.fromisoformat(latest.replace("Z", "+00:00"))
    return (t - timedelta(days=days)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _json(results: list[dict], clusters: list[Cluster]) -> dict:
    def member(m) -> dict:
        return {k: m[k] for k in m.keys()}

    out = {"consolidated": [], "other_clusters": [], "other_clusters_total": 0,
           "newest_lesson_ts": max((c.last_ts for c in clusters if c.last_ts), default="")}
    done = set()
    for e in results:
        c = e["cluster"]
        done.add(c.cid)
        cov = dict(e["coverage"])
        rule = cov.pop("rule", None)
        cov["rule"] = None if rule is None else {"source": rule.source, "heading": rule.heading, "text": rule.text}
        out["consolidated"].append({
            "cluster_id": c.cid, "mentions": len(c.members),
            "canon": e["canon"], "coverage": cov, "score": c.score, "sessions": sorted(c.sessions),
            "accounts": sorted(c.accounts), "projects": sorted(c.projects), "first_ts": c.first_ts,
            "last_ts": c.last_ts, "operator": c.operator, "members": [member(m) for m in c.members],
        })
    # Every cluster is in the store; a report lists the best OTHER_CLUSTERS_LISTED
    # of the rest (clusters come sorted by score), so its size does not grow with
    # every lesson ever extracted -- most of the tail is single-session lessons.
    others = [c for c in clusters if c.cid not in done]
    out["other_clusters_total"] = len(others)
    for c in others[:OTHER_CLUSTERS_LISTED]:
        out["other_clusters"].append({
            "title": c.rep["title"], "lesson": c.rep["lesson"], "score": c.score, "sessions": c.n_sessions,
            "mentions": len(c.members), "projects": sorted(c.projects), "last_ts": c.last_ts,
        })
    return out


OTHER_CLUSTERS_LISTED = 500

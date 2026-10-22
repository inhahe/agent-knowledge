"""Run the whole report stage against fake model servers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from akn import config as config_mod
from akn import report as report_mod
from akn.llm import ChatResult
from akn.store import Store
from tests.test_store_cluster_coverage import make_window


class _NoServer:
    def __init__(self, *a, **k):
        self.base_url = "http://fake"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeClient:
    """Embeds by topic keyword; consolidates and judges deterministically."""

    TOPICS = ("backslash", "pid", "lock")

    calls = 0

    def __init__(self, *a, **k):
        pass

    def embed(self, texts, batch=32):
        rows = []
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if k in low else 0.0 for k in self.TOPICS] + [0.05], dtype=np.float32)
            h = int(hashlib.md5(t.encode()).hexdigest()[:4], 16) / 65535 * 0.02  # tiny jitter
            v[-1] += h
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)

    def chat_json(self, system, user, schema, **kw):
        type(self).calls += 1
        if "EXISTING INSTRUCTIONS" in user:
            verdict = "covered" if "backslash" in user.split("EXISTING INSTRUCTIONS")[0].lower() else "not-covered"
            return ChatResult({"reasoning": "r", "verdict": verdict, "best_rule": 0 if verdict == "covered" else -1,
                               "missing": ""}, 1, 1, 0.1, "")
        topic = next((k for k in self.TOPICS if k in user.lower()), "other")
        dest = "not-worth-recording" if topic == "lock" else "global-instructions"
        return ChatResult({"title": f"About {topic}", "lesson": f"Canonical {topic} lesson.", "applies_when": "w",
                           "kind": "tool-quirk", "scope": "this-machine", "destination": dest, "consistent": True,
                           "notes": ""}, 1, 1, 0.1, "")


def test_report_end_to_end(tmp_path, monkeypatch):
    cfg = config_mod.Config()
    cfg.state_dir = tmp_path / "state"
    cfg.reports_dir = tmp_path / "reports"
    rules = tmp_path / "CLAUDE.md"
    rules.write_text("# Rules\n\n- Backslashes through the Bash tool get eaten; use forward slashes.\n", encoding="utf-8")
    cfg.instruction_globs = [str(rules)]
    cfg.instruction_git = []
    st = Store(cfg.db_path)
    w = make_window("x.jsonl", "s", 0, ["u"])
    st.replace_file_windows(Path("x.jsonl"), [w])
    specs = [
        ("s1", "backslash", "agent-discovered"), ("s2", "backslash", "agent-discovered"),
        ("s3", "pid", "operator"), ("s4", "pid", "operator"),
        ("s5", "lock", "tool-failure"), ("s6", "lock", "tool-failure"),
        ("s7", "unique", "agent-discovered"),
    ]
    for sess, topic, source in specs:
        st.db.execute(
            "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
            " evidence, source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (w.id, sess, "claude", "p", "2026-09-30T10:00:00Z", f"{topic} title", f"A lesson about {topic}.",
             "tool-quirk", "this-machine", "when", f"evidence for {topic}: cargo build -p <crate>", source, 0.8),
        )
    st.db.commit()
    monkeypatch.setattr(report_mod, "LlamaServer", _NoServer)
    monkeypatch.setattr(report_mod, "Client", _FakeClient)

    path = report_mod.run(cfg, st, top=10)
    md = path.read_text(encoding="utf-8")
    assert "## Not in any instruction file (1)" in md and "About pid" in md
    assert "Already a rule, yet still recurring" in md and "About backslash" in md
    assert "## Judged not worth recording (1)" in md and "About lock" in md
    assert "unique" not in md.split("## Corpus")[1]  # a one-session, non-operator lesson is not consolidated
    assert "**operator-sourced**" in md
    js = path.with_suffix(".json").read_text(encoding="utf-8")
    assert '"other_clusters"' in js and "unique title" in js
    assert (cfg.reports_dir / "latest.md").read_text(encoding="utf-8") == md
    from tests.test_mdtext import raw_html
    assert raw_html(md) == [] and "<details>" not in md  # plain Markdown only
    assert "cargo build -p \\<crate>" in md  # tag-like text from the transcripts is escaped

    # A second run over unchanged lessons answers everything from the cache.
    first_calls = _FakeClient.calls
    assert first_calls > 0
    md2 = report_mod.run(cfg, st, top=10).read_text(encoding="utf-8")
    assert _FakeClient.calls == first_calls
    assert "[new since last report]" not in md2  # nothing is newer than the last report

    # A lesson the curator has decided on gives its place up, until it recurs.
    js = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    pid = next(e for e in js["consolidated"] if e["canon"]["title"] == "About pid")
    decided = cfg.state_dir / "curator" / "state.json"
    decided.parent.mkdir(parents=True, exist_ok=True)
    decided.write_text(json.dumps({"clusters": {str(pid["cluster_id"]): {"status": "skipped", "sessions": 2,
                                                                          "date": "2026-10-03"}}}), encoding="utf-8")

    def consolidated() -> tuple[list[str], str]:
        p = report_mod.run(cfg, st, top=10)
        out = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return [e["canon"]["title"] for e in out["consolidated"]], p.read_text(encoding="utf-8")

    titles, md3 = consolidated()
    assert "About pid" not in titles and "About backslash" in titles
    assert "left out because the curator has decided on them (each returns if it recurs): 1" in md3
    for sess in ("s8", "s9", "s10"):  # 5 sessions: grown by half, and by 3 -- it is looked at again
        st.db.execute(
            "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
            " evidence, source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (w.id, sess, "claude", "p", "2026-10-01T10:00:00Z", "pid title", "A lesson about pid.", "tool-quirk",
             "this-machine", "when", "evidence for pid", "operator", 0.8))
    st.db.commit()
    titles, _md = consolidated()
    assert "About pid" in titles


def test_select_open_passes_over_settled_clusters_and_counts_them():
    class C:
        def __init__(self, cid, n):
            self.cid, self.n_sessions = cid, n

    eligible = [C(1, 9), C(2, 8), C(3, 7), C(4, 2)]
    chosen, set_aside = report_mod.select_open(eligible, {1: 12, 3: 7}, top=2)
    assert [c.cid for c in chosen] == [2, 3] and set_aside == 1  # 3 has recurred enough to return

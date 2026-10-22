"""The curator: which lessons it considers, how its sections are written into the
instruction files (never touching the operator's part), the budgets, the history,
and the operator's controls (drop, undo)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from akn import config as config_mod
from akn import curate
from akn.curate import SECTION_HEADING
from akn.store import Store
from tests.conftest import git_repo


def entry(cid, title, *, sessions=3, verdict="not-covered", dest="global-instructions", last="2026-10-01T10:00:00Z",
          score=5.0):
    return {"cluster_id": cid, "mentions": sessions, "score": score, "operator": False,
            "sessions": [f"s{i}" for i in range(sessions)], "projects": ["E--x-proj"],
            "first_ts": "2026-09-01T10:00:00Z", "last_ts": last,
            "canon": {"title": title, "lesson": f"{title} lesson.", "applies_when": "w", "kind": "tool-quirk",
                      "scope": "this-machine", "destination": dest, "consistent": True, "notes": ""},
            "coverage": {"verdict": verdict, "best_rule": -1, "missing": "", "reasoning": "", "rule": None},
            "members": [{"evidence": f"evidence for {title}", "session_id": "s0", "ts": last, "project": "p",
                         "source": "agent-discovered"}]}


def report(*entries, newest="2026-10-02T10:00:00Z"):
    return {"newest_lesson_ts": newest, "consolidated": list(entries)}


EMPTY = {"clusters": {}, "last_report": "", "dropped": []}


# --- which lessons are considered ---------------------------------------------------------------

def test_candidates_are_uncovered_or_still_recurring_and_ranked():
    rep = report(
        entry(1, "uncovered", score=3),
        entry(2, "partly", verdict="partly-covered", score=9),
        entry(3, "covered, recent", verdict="covered", last="2026-09-30T00:00:00Z", score=4),
        entry(4, "covered, quiet", verdict="covered", last="2026-08-01T00:00:00Z", score=8),
        entry(5, "not worth it", dest="not-worth-recording", score=10),
    )
    got = curate.select_candidates(rep, EMPTY, recur_more=3, limit=10)
    assert [e["canon"]["title"] for e in got] == ["partly", "covered, recent", "uncovered"]
    assert got[1]["verdict"] == "covered-recurring"


def test_a_rules_lesson_returns_after_recurring_more_and_a_skipped_one_after_growing_by_half():
    assert curate.reopen_at({"status": "added", "sessions": 20}, 3) == 23  # recurring despite the rule
    assert curate.reopen_at({"status": "skipped", "sessions": 20}, 3) == 30  # its own steady rate is no news
    assert curate.reopen_at({"status": "skipped", "sessions": 2}, 3) == 5


def test_settled_tells_the_report_which_clusters_to_leave_out(cfg):
    curate.save_state(cfg, {**curate.load_state(cfg), "clusters": {
        "7": {"status": "added", "sessions": 4, "date": "2026-10-03"},
        "8": {"status": "skipped", "sessions": 10, "date": "2026-10-03"},
        "junk": {"status": "skipped"}}})
    assert curate.settled(cfg) == {7: 7, 8: 15}


def test_where_each_rule_came_from_is_recorded_and_shown_to_later_runs(cfg):
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    st = Store(cfg.db_path)  # the miner's store: lessons in clusters 1 and 2

    def lesson(cluster, session, ts):
        cur = st.db.execute(
            "INSERT INTO lessons (window_id, session_id, account, project, ts, title, lesson, kind, scope, trigger,"
            " evidence, source, confidence) VALUES ('w', ?, 'a', 'p', ?, 't', 'l', 'k', 's', 'tr', 'e', 'x', 0.8)",
            (session, ts))
        st.db.execute("INSERT INTO lesson_cluster (lesson_id, cluster_id) VALUES (?, ?)", (cur.lastrowid, cluster))
        st.db.commit()

    for s in ("s1", "s2", "s3"):
        lesson(1, s, "2026-09-20T10:00:00Z")
    lesson(2, "s4", "2026-09-21T10:00:00Z")
    write_report(cfg, report(entry(1, "alpha"), entry(2, "beta")))
    curate.prepare(cfg)
    draft(cfg, {"global": {"rules": "- **Alpha.** Do alpha.\n- **Beta.** Do beta.",
                           "changes": [{"change": "added", "rule": "Alpha", "candidates": ["C1"]},
                                       {"change": "added", "rule": "**Beta.**", "candidates": ["C2"]}]}})
    curate.finalize(cfg)
    rules = curate.load_state(cfg)["rules"]
    assert rules["alpha"]["clusters"] == [1] and rules["beta"]["clusters"] == [2]
    lesson(1, "s9", "2099-01-01T10:00:00Z")  # the lesson came up again after its rule was written
    write_report(cfg, report(entry(3, "gamma")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    text = (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    assert "**Alpha.** -- written" in text and "4 sessions in all, 1 of them after the rule was written" in text
    assert "**Beta.** -- written" in text and "none of them after the rule was written" in text
    draft(cfg, {"global": {"rules": "- **Alpha and beta.** Do both.",
                           "changes": [{"change": "merged", "rule": "Alpha and beta", "replaces": ["Alpha", "Beta"],
                                        "candidates": ["C1"]}]}})
    curate.finalize(cfg)
    rules = curate.load_state(cfg)["rules"]
    assert list(rules) == ["alpha and beta"] and rules["alpha and beta"]["clusters"] == [1, 2, 3]
    curate.drop(cfg, "do both")
    assert curate.load_state(cfg)["rules"] == {}  # a rule that left its section is forgotten


def test_a_decided_lesson_returns_only_after_recurring_more():
    state = {**EMPTY, "clusters": {"1": {"status": "skipped", "sessions": 3, "date": "2026-09-20"}}}
    assert curate.select_candidates(report(entry(1, "x", sessions=5)), state, 3, 10) == []
    back = curate.select_candidates(report(entry(1, "x", sessions=6)), state, 3, 10)
    assert len(back) == 1 and "3 more sessions since it was skipped" in back[0]["recur_note"]


# --- the section inside a file -----------------------------------------------------------------------

def test_the_section_is_added_replaced_and_removed_without_touching_the_rest(tmp_path):
    lf = tmp_path / "lf.md"
    lf.write_bytes(b"# Operator\n\n- my rule\n")
    crlf = tmp_path / "crlf.md"
    crlf.write_bytes(b"# graphify\r\n\r\n# Operator\r\n\r\n- my rule\r\n")
    for p in (lf, crlf):
        before = p.read_bytes()
        curate.write_section(p, "- **One.** First rule.")
        text = p.read_bytes()
        assert text.startswith(before.rstrip(b"\r\n"))  # the operator's part, byte for byte
        assert SECTION_HEADING.encode() in text and b"- **One.** First rule." in text
        curate.write_section(p, "- **Two.** Second rule.")
        _own, rules, _nl = curate.read_section(p)
        assert rules == "- **Two.** Second rule." and b"First rule" not in p.read_bytes()
        assert (b"\r\n" in p.read_bytes()) == (p is crlf)  # line endings kept
        curate.write_section(p, "")
        assert p.read_bytes().rstrip(b"\r\n") == before.rstrip(b"\r\n")  # removed cleanly


def test_check_rules_enforces_budget_and_shape():
    sec = curate.Section("global", [], max_chars=60, max_rules=2, note="")
    assert curate.check_rules("- **A.** a\n- **B.** b", sec) == []
    assert any("over the budget of 60" in p for p in curate.check_rules("- " + "x" * 80, sec))
    assert any("3 rules" in p for p in curate.check_rules("- a\n- b\n- c", sec))
    assert any("heading" in p for p in curate.check_rules("## Not allowed\n- a", sec))
    assert curate.check_rules("### Some project\n- a", sec) == []
    assert any("tag-like" in p for p in curate.check_rules("- run <pid> now", sec))
    assert curate.check_rules("- run `<pid>` now", sec) == []
    assert any("@docs/x.md" in p for p in curate.check_rules("- see @docs/x.md", sec))  # Claude Code imports it
    assert curate.check_rules("- see `@docs/x.md`, or mail a@b.com", sec) == []


# --- prepare, finalize, history ------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = config_mod.Config()
    c.state_dir, c.reports_dir, c.curated_dir = tmp_path / "state", tmp_path / "reports", tmp_path / "curated"
    c.reports_dir.mkdir()
    c.curated_global_chars, c.curated_global_rules = 400, 4
    g1, g2, ws = tmp_path / "a" / "CLAUDE.md", tmp_path / "b" / "CLAUDE.md", tmp_path / "ws" / "CLAUDE.md"
    for p, text in ((g1, "# graphify\n\n# Killing processes\n\n- Kill only your own PID.\n"),
                    (g2, "# Killing processes\n\n- Kill only your own PID.\n"),
                    (ws, "# Workspace rules\n\n- Clean up after yourself.\n")):
        p.parent.mkdir()
        p.write_text(text, encoding="utf-8", newline="\n")
    c.accounts = [g1.parent, g2.parent]
    # The projects folder: its workspace CLAUDE.md is ws. No instruction file is
    # read from git, and there are no notes on the folder.
    c.projects_dir = ws.parent
    c.instruction_git, c.curator_notes = [], ""
    return c


def write_report(c, rep, name="lessons-2026-10-02-0500.json"):
    (c.reports_dir / name).write_text(json.dumps(rep), encoding="utf-8")


def draft(c, obj):
    (curate.work_dir(c) / "draft.json").write_text(json.dumps(obj), encoding="utf-8")


def test_prepare_shows_sections_budgets_operator_rules_and_candidates(cfg):
    write_report(cfg, report(entry(1, "alpha"), entry(2, "beta", verdict="partly-covered")))
    out = curate.prepare(cfg)
    assert out["candidates"] == 2
    text = (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    assert "## Section `global`" in text and "at most 4 rules and 400 characters" in text
    assert "## C1 -- alpha" in text and "evidence for alpha" in text and "(none yet)" in text
    ctx = (curate.work_dir(cfg) / "context" / "global-operator-rules.md").read_text(encoding="utf-8")
    assert "Kill only your own PID" in ctx


def test_prepare_shows_the_operator_rules_an_account_file_imports(cfg):
    shared = cfg.global_files[0].parent.parent / "shared.md"
    shared.write_text("# Shared rules\n\n- Stage what you changed, by path.\n", encoding="utf-8", newline="\n")
    for p in cfg.global_files:
        p.write_text("@../shared.md\n", encoding="utf-8", newline="\n")
    write_report(cfg, report(entry(1, "alpha")))
    curate.prepare(cfg)
    ctx = (curate.work_dir(cfg) / "context" / "global-operator-rules.md").read_text(encoding="utf-8")
    assert any(f"(Imported by {p}: {shared})" in ctx for p in cfg.global_files)
    assert "- Stage what you changed, by path." in ctx


def test_prepare_gives_the_files_read_from_git_and_the_notes_on_the_projects_folder(cfg, tmp_path):
    repo = git_repo(tmp_path / "ws" / "repo", {"CLAUDE.md": "# Repo\n\n- Rule from git.\n",
                                               "RULES.txt": "- A rule in a text file.\n"})
    cfg.instruction_git = [[repo.as_posix(), "HEAD", "CLAUDE.md"], [repo.as_posix(), "HEAD", "RULES.txt"]]
    cfg.curator_notes = "The `repo` checkout is out of date: read `repo-b`."
    write_report(cfg, report(entry(1, "alpha")))
    curate.prepare(cfg)
    wd = curate.work_dir(cfg)
    text = (wd / "input.md").read_text(encoding="utf-8")
    assert (wd / "context" / "1-repo-CLAUDE.md").read_text(encoding="utf-8") == "# Repo\n\n- Rule from git.\n"
    assert f"context/1-repo-CLAUDE.md (CLAUDE.md of {repo.as_posix()} at HEAD)" in text
    assert "context/2-repo-RULES.txt" in text
    assert "## Notes on the projects folder\n\nThe `repo` checkout is out of date: read `repo-b`." in text
    assert text.index("Notes on the projects folder") < text.index("# Candidate lessons")

    # A file git can no longer show is neither offered nor left behind from an earlier
    # morning, whatever its name ends in: the curator could still find a stale copy.
    cfg.instruction_git = [[repo.as_posix(), "HEAD", "CLAUDE.md"], [repo.as_posix(), "HEAD", "missing.md"]]
    write_report(cfg, report(entry(2, "beta")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    text = (wd / "input.md").read_text(encoding="utf-8")
    assert "missing.md" not in text and "RULES" not in text
    assert sorted(p.name for p in (wd / "context").iterdir()) == ["1-repo-CLAUDE.md", "global-operator-rules.md",
                                                                  "workspace-operator-rules.md"]


def test_prepare_shows_the_operator_text_of_every_copy_not_just_one(cfg):
    g1, g2 = cfg.global_files
    g2.write_text("# Killing processes\n\n- Only in the second account.\n", encoding="utf-8", newline="\n")
    write_report(cfg, report(entry(1, "alpha")))
    curate.prepare(cfg)
    ctx = (curate.work_dir(cfg) / "context" / "global-operator-rules.md").read_text(encoding="utf-8")
    assert "# graphify" in ctx and "- Only in the second account." in ctx
    assert f"(In {g1})" in ctx and f"(In {g2})" in ctx


def test_a_section_whose_imported_rules_cannot_be_read_is_not_written(cfg):
    for p in cfg.global_files:
        p.write_text("@../missing.md\n", encoding="utf-8", newline="\n")
    write_report(cfg, report(entry(1, "alpha")))
    curate.prepare(cfg)
    text = (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    assert "This section cannot be written today (its operator's rules cannot all be read:" in text
    assert "@../missing.md" in text
    draft(cfg, {"global": {"rules": "- **Alpha.** Do alpha.",
                           "changes": [{"change": "added", "summary": "alpha", "candidates": ["C1"]}]}})
    out = curate.finalize(cfg)
    assert out["applied"] == {} and "its operator's rules cannot all be read" in out["problems"][0]
    assert SECTION_HEADING not in cfg.global_files[0].read_text(encoding="utf-8")


def test_finalize_writes_every_copy_records_and_keeps_history(cfg):
    write_report(cfg, report(entry(1, "alpha"), entry(2, "beta"), entry(3, "gamma")))
    curate.prepare(cfg)
    draft(cfg, {
        "global": {"rules": "- **Alpha.** Do alpha.",
                   "changes": [{"change": "added", "summary": "alpha", "reason": "5 sessions", "candidates": ["C1"]}]},
        "skipped": [{"candidate": "C2", "reason": "already said"}],
        "operator_rules_missed": [{"file": "global", "rule": "Kill only your own PID", "suggestion": "name taskkill"}],
    })
    out = curate.finalize(cfg)
    assert out["applied"] == {"global": 1} and out["problems"] == []
    for p in cfg.global_files:
        own, rules, _nl = curate.read_section(p)
        assert rules == "- **Alpha.** Do alpha." and "Kill only your own PID" in own
    assert "graphify" in cfg.global_files[0].read_text(encoding="utf-8")  # each copy keeps its own part
    assert SECTION_HEADING not in cfg.workspace_file.read_text(encoding="utf-8")  # untouched: no changes
    state = curate.load_state(cfg)
    assert state["clusters"]["1"]["status"] == "added" and state["clusters"]["2"]["status"] == "skipped"
    assert "3" not in state["clusters"]  # not reached: comes back with the next report
    log = curate.show_log(cfg)
    assert "added: alpha -- 5 sessions" in log and "Rules of yours that agents keep missing" in log
    assert "curator" in curate._git(cfg, "log", "--format=%s").stdout
    assert curate.prepare(cfg)["skipped"].endswith("was already curated")


def test_finalize_refuses_a_section_over_budget_or_changing_too_much(cfg):
    write_report(cfg, report(entry(1, "alpha")))
    curate.prepare(cfg)
    big = "\n".join(f"- **Rule {i}.** text" for i in range(9))
    draft(cfg, {"global": {"rules": big, "changes": [{"change": "added", "candidates": ["C1"]}] * 9}})
    out = curate.finalize(cfg)
    assert out["applied"] == {} and "kept yesterday's rules" in out["problems"][0]
    assert "over the budget" in out["problems"][0] and "9 changes" in out["problems"][0]
    assert SECTION_HEADING not in cfg.global_files[1].read_text(encoding="utf-8")


def curate_once(c, name, rules, cands):
    write_report(c, report(*[entry(i, t) for i, t in cands]), name=name)
    curate.prepare(c)
    draft(c, {"global": {"rules": rules, "changes": [{"change": "added", "summary": "x", "candidates": ["C1"]}]}})
    return curate.finalize(c)


def test_drop_removes_one_rule_everywhere_and_keeps_it_out(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.\n- **Beta.** Do beta.", [(1, "a")])
    assert "error" in curate.drop(cfg, "Do")  # ambiguous: two rules match
    out = curate.drop(cfg, "beta")
    assert out["dropped"] == "- **Beta.** Do beta."
    for p in cfg.global_files:
        assert curate.read_section(p)[1] == "- **Alpha.** Do alpha."
    write_report(cfg, report(entry(7, "delta")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    assert "- **Beta.** Do beta. (dropped" in (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")


def test_undo_takes_back_the_newest_run_and_keeps_later_changes(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.\n- **Beta.** Do beta.", [(1, "a")])
    curate_once(cfg, "lessons-2026-10-03-0500.json",
                "- **Alpha.** Do alpha, sharper.\n- **Beta.** Do beta.\n- **Gamma.** Do gamma.\n- **Delta.** Do delta.",
                [(2, "b")])
    curate.drop(cfg, "delta")  # the operator's own change, after that run: undo must keep it
    out = curate.undo(cfg)
    assert out["sections"] == ["global"]
    for p in cfg.global_files:
        assert curate.read_section(p)[1] == "- **Alpha.** Do alpha.\n- **Beta.** Do beta."
    dropped = [d["text"] for d in curate.load_state(cfg)["dropped"]]
    assert "- **Gamma.** Do gamma." in dropped  # added by the undone run: not simply added back tomorrow
    assert "- **Alpha.** Do alpha, sharper." not in dropped  # reworded: the rule itself stays
    curate.undo(cfg)  # the run before
    assert curate.read_section(cfg.global_files[0])[1] is None
    assert "error" in curate.undo(cfg)  # nothing left to undo


def test_undo_brings_back_a_retired_subsection_in_place():
    before = "- **A.** a\n\n### Some project\n\n- **S.** s\n- **T.** t"
    after = "- **A.** a\n- **B.** b"
    assert curate._revert(after, after, before) == before


# --- never destroying the operator's text -----------------------------------------------------------------

def test_a_section_appended_below_the_curated_one_stays_and_the_curated_one_moves_below_it(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_bytes(b"# Operator\n\n- my rule\n")
    curate.write_section(p, "- **One.** First rule.")
    with p.open("ab") as f:
        f.write(b"\n# Added later\n\n- a rule the operator added at the end\n")
    own, rules, _nl = curate.read_section(p)
    assert rules == "- **One.** First rule." and "a rule the operator added at the end" in own
    curate.write_section(p, "- **Two.** Second rule.")
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# Operator\n\n- my rule\n\n# Added later\n\n- a rule the operator added at the end\n")
    assert text.index("# Added later") < text.index(SECTION_HEADING) and "First rule" not in text


def test_an_older_wording_of_the_intro_is_still_recognised():
    text = f"# Mine\n\n{SECTION_HEADING}\n\n{curate.INTRO_PREFIX}, in an older wording.\n\n- **A.** a  \n"
    assert curate.split_file(text) == ("# Mine", "- **A.** a")
    assert curate.split_file("# Mine\n") == ("# Mine\n", None)


def test_a_hand_edit_is_adopted_into_every_copy_and_then_kept_word_for_word(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.\n- **Beta.** Do beta.\n- **Gamma.** g",
                [(1, "a")])
    g1, g2 = cfg.global_files
    text = g1.read_text(encoding="utf-8").replace("Do alpha.", "Do alpha, carefully.").replace("- **Beta.** Do beta.\n", "")
    g1.write_text(text.rstrip("\n") + "\n- **Mine.** The operator's own.\n", encoding="utf-8", newline="\n")
    write_report(cfg, report(entry(2, "b")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    want = "- **Alpha.** Do alpha, carefully.\n- **Gamma.** g\n- **Mine.** The operator's own."
    assert [curate.read_section(p)[1] for p in (g1, g2)] == [want, want] and curate.canonical(cfg, "global") == want
    assert "graphify" in g1.read_text(encoding="utf-8") and "graphify" not in g2.read_text(encoding="utf-8")
    state = curate.load_state(cfg)
    assert {h["text"] for h in state["hand_rules"]} == {"- **Alpha.** Do alpha, carefully.",
                                                        "- **Mine.** The operator's own."}
    assert [d["text"] for d in state["dropped"]] == ["- **Beta.** Do beta."]  # reworded Alpha is not dropped
    text = (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    assert "wrote or reworded these rules by hand" in text and "adopted the edit made by hand" in curate.show_log(cfg)
    draft(cfg, {"global": {"rules": want.replace("The operator's own.", "Reworded by the curator."),
                           "changes": [{"change": "sharpened", "candidates": ["C1"]}]}})
    out = curate.finalize(cfg)
    assert out["applied"] == {} and "1 rule(s) the operator wrote by hand" in out["problems"][0]
    assert curate.read_section(g2)[1] == want


def test_copies_edited_in_different_ways_are_left_alone(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.", [(1, "a")])
    for p, how in zip(cfg.global_files, ("one way", "another way")):
        p.write_text(p.read_text(encoding="utf-8").replace("Do alpha.", f"Do alpha {how}."), encoding="utf-8",
                     newline="\n")
    before = [p.read_bytes() for p in cfg.global_files]
    write_report(cfg, report(entry(2, "b")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    assert "cannot be written today" in (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    draft(cfg, {"global": {"rules": "- **Alpha.** Do alpha.\n- **B.** b",
                           "changes": [{"change": "added", "candidates": ["C1"]}]}})
    out = curate.finalize(cfg)
    assert out["applied"] == {} and "edited by hand in different ways" in out["problems"][0]
    assert [p.read_bytes() for p in cfg.global_files] == before
    assert "edited by hand in different ways" in curate.drop(cfg, "alpha")["error"]
    assert len(curate.differing_copies(cfg)) == 2


def test_an_edit_made_while_the_curator_runs_is_adopted_not_overwritten(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.", [(1, "a")])
    write_report(cfg, report(entry(2, "b")), name="lessons-2026-10-03-0500.json")
    curate.prepare(cfg)
    g1 = cfg.global_files[0]
    g1.write_text(g1.read_text(encoding="utf-8").replace("Do alpha.", "Do alpha now."), encoding="utf-8", newline="\n")
    draft(cfg, {"global": {"rules": "- **Alpha.** Do alpha.\n- **B.** b",
                           "changes": [{"change": "added", "candidates": ["C1"]}]}})
    out = curate.finalize(cfg)
    assert out["applied"] == {} and "edited by hand while the curator ran" in out["problems"][0]
    assert [curate.read_section(p)[1] for p in cfg.global_files] == ["- **Alpha.** Do alpha now."] * 2
    assert "2" not in curate.load_state(cfg)["clusters"]  # its lesson comes back with the next report


def test_a_stale_or_missing_copy_is_brought_up_to_date_not_adopted(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.", [(1, "a")])
    g1, g2 = cfg.global_files
    old = g2.read_bytes()
    curate_once(cfg, "lessons-2026-10-03-0500.json", "- **Alpha.** Do alpha.\n- **Beta.** b", [(2, "b")])
    g2.write_bytes(old)  # an older version the curator wrote (a backup restored)
    g1.write_text(curate.read_section(g1)[0] + "\n", encoding="utf-8", newline="\n")  # no section at all
    out = curate.sync(cfg)
    assert out.adopted == set() and out.conflicts == {} and len(out.done) == 1
    assert [curate.read_section(p)[1] for p in (g1, g2)] == ["- **Alpha.** Do alpha.\n- **Beta.** b"] * 2
    assert curate.load_state(cfg)["hand_rules"] == [] and curate.load_state(cfg)["dropped"] == []


def test_blank_lines_and_trailing_spaces_an_editor_changes_are_not_a_hand_edit(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.   \n- **Beta.** b", [(1, "a")])
    assert curate.canonical(cfg, "global") == "- **Alpha.** Do alpha.\n- **Beta.** b"  # trailing spaces never stored
    g1 = cfg.global_files[0]
    g1.write_bytes(g1.read_bytes() + b"\n\n")
    out = curate.sync(cfg)
    assert out.adopted == set() and out.done == []


def test_a_file_holding_the_heading_twice_is_refused_and_no_copy_is_written(cfg):
    curate_once(cfg, "lessons-2026-10-02-0500.json", "- **Alpha.** Do alpha.", [(1, "a")])
    g1, g2 = cfg.global_files
    g2.write_text(g2.read_text(encoding="utf-8") + f"\n{SECTION_HEADING}\n\n- a second copy\n", encoding="utf-8",
                  newline="\n")
    before = g1.read_bytes()
    out = curate_once(cfg, "lessons-2026-10-03-0500.json", "- **Alpha.** Do alpha.\n- **Beta.** b", [(2, "b")])
    assert out["applied"] == {} and "heading twice" in out["problems"][0]
    assert g1.read_bytes() == before and curate.canonical(cfg, "global") == "- **Alpha.** Do alpha."


def test_a_replace_refused_for_a_moment_is_retried_and_one_refused_for_good_writes_in_place(tmp_path, monkeypatch):
    p = tmp_path / "f.md"
    p.write_bytes(b"old")
    calls = []
    real_replace = os.replace

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(curate.time, "sleep", lambda _s: None)
    monkeypatch.setattr(curate.os, "replace", flaky)
    curate._atomic_write(p, b"new")
    assert p.read_bytes() == b"new" and len(calls) == 3

    def refuse(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(curate.os, "replace", refuse)
    curate._atomic_write(p, b"newer")
    assert p.read_bytes() == b"newer" and not (tmp_path / "f.md.curator-tmp").exists()


# --- the headless run --------------------------------------------------------------------------------------

def test_the_curator_may_read_and_write_only_its_draft(cfg, monkeypatch):
    monkeypatch.setattr(curate, "claude_exe", lambda: "claude")
    cmd = curate.curator_command(cfg, "PROMPT")
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--disallowedTools")]
    denied = cmd[cmd.index("--disallowedTools") + 1 : cmd.index("--strict-mcp-config")]
    assert "Edit(state/curator/draft.json)" in allowed and "Bash" in denied and "PowerShell" in denied
    assert "Write" not in allowed and "Edit" not in allowed and "Edit" not in denied
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk" and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--add-dir") + 1] == str(cfg.projects_dir)  # read access, to check facts


def test_accounts_are_tried_in_order_until_one_writes_the_draft(cfg, monkeypatch, tmp_path):
    a, b = cfg.accounts
    c = tmp_path / ".claude-account-c"
    c.mkdir()
    (c / "CLAUDE.md").write_text("# Killing processes\n\n- Kill only your own PID.\n", encoding="utf-8", newline="\n")
    cfg.accounts, cfg.curator_config_dir = [a, b, c], c
    assert curate.curator_accounts(cfg) == [c, a, b]
    write_report(cfg, report(entry(1, "alpha")))
    monkeypatch.setattr(curate, "claude_exe", lambda: "claude")
    tried = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[0] != "claude":  # the history repository's git calls run for real
            return real_run(cmd, *args, **kwargs)
        env = kwargs["env"]
        tried.append(Path(env["CLAUDE_CONFIG_DIR"]).name)
        if len(tried) == 1:
            return subprocess.CompletedProcess(cmd, 1, b"You've hit your weekly limit", b"")
        draft(cfg, {"global": {"rules": "- **A.** a", "changes": [{"change": "added", "candidates": ["C1"]}]}})
        return subprocess.CompletedProcess(cmd, 0, b"global: 1 added", b"")

    monkeypatch.setattr(curate.subprocess, "run", fake_run)
    out = curate.run(cfg, wait_hours=0)
    assert tried == [".claude-account-c", "a"] and out["applied"] == {"global": 1}
    assert all(curate.read_section(p)[1] == "- **A.** a" for p in cfg.global_files)
    log = next(cfg.state_dir.glob("curator-*.log")).read_text(encoding="utf-8")
    assert "weekly limit" in log and "finalize" in log


def test_a_stale_report_is_said_in_the_log_the_result_the_exit_code_and_curated_log(cfg, monkeypatch):
    assert "no lessons report yet" in curate.stale_report(cfg)
    write_report(cfg, report(entry(1, "alpha")))  # lessons-2026-10-02-0500.json
    written = datetime(2026, 10, 2, 5, 0).timestamp()
    assert curate.stale_report(cfg, now=written + 86400) is None
    assert "is 5 days old" in curate.stale_report(cfg, now=written + 5 * 86400)
    # Dated by its name: a copy of reports/ that makes the file look new hides nothing.
    rep = next(cfg.reports_dir.glob("lessons-*.json"))
    os.utime(rep)
    assert "lessons-2026-10-02-0500.json" in curate.stale_report(cfg)  # today is long after 2026-10-02
    assert curate.show_log(cfg).startswith("**Warning: the newest")

    curate.save_state(cfg, {**curate.load_state(cfg), "last_report": rep.name})  # nothing new: the quiet case
    out = curate.run(cfg, wait_hours=0)
    assert out["skipped"].endswith("was already curated") and "days old" in out["warning"]
    log = next(cfg.state_dir.glob("curator-*.log")).read_text(encoding="utf-8")
    assert "WARNING: the newest lessons report" in log
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)
    from akn import __main__ as cli
    assert cli.main(["curate", "--run"]) == 1


def test_a_named_account_without_a_claude_md_blocks_the_global_section_aloud(cfg, tmp_path):
    new = tmp_path / ".claude-new"
    new.mkdir()
    cfg.curated_accounts = [*cfg.accounts, new]
    write_report(cfg, report(entry(1, "alpha")))
    out = curate.prepare(cfg)
    assert "no CLAUDE.md" in out["blocked"]["global"] and "workspace" not in out["blocked"]
    assert "This section cannot be written today" in (curate.work_dir(cfg) / "input.md").read_text(encoding="utf-8")
    draft(cfg, {"global": {"rules": "- **A.** a", "changes": [{"change": "added", "candidates": ["C1"]}]}})
    fin = curate.finalize(cfg)
    assert fin["applied"] == {} and "no CLAUDE.md" in fin["problems"][0]
    assert all(curate.read_section(p)[1] is None for p in cfg.global_files if p.is_file())  # none written


def test_a_projects_folder_refused_is_an_error_not_a_quiet_skip(cfg):
    cfg.workspace_file.unlink()
    assert "is not the projects folder" in curate.run(cfg, wait_hours=0)["error"]


def test_the_run_waits_for_a_live_nightly_run(cfg):
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "nightly.lock").write_text(str(os.getpid()), encoding="utf-8")  # a live PID: ours
    assert curate.run(cfg, wait_hours=0) == {"skipped": "nightly run still running"}

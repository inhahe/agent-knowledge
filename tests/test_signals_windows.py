from __future__ import annotations

from akn import transcripts as T
from akn.signals import find_signals
from akn.windows import build_windows, render_event


def parse(tb, tmp_path, name="s.jsonl"):
    return T.parse_file(tb.write(tmp_path / name), "claude", "proj")


def sig_kinds(sigs):
    return [s.kind for s in sigs]


def test_operator_correction_outweighs_plain_prompt(tb, tmp_path):
    tb.human("add a --verbose flag to the scan command")
    tb.human("no, don't put it there, I told you the CLI lives in __main__")
    sigs = find_signals(parse(tb, tmp_path).events)
    assert sig_kinds(sigs) == ["human", "human-correction"]
    assert sigs[1].weight > sigs[0].weight


def test_trivial_prompts_and_loop_prompts_give_no_signal(tb, tmp_path):
    tb.human("go").human("Continue.").human("<<autonomous-loop-dynamic>>")
    assert find_signals(parse(tb, tmp_path).events) == []


def test_grep_no_match_is_benign_but_real_failure_counts(tb, tmp_path):
    tb.bash("t1", "grep -rn foo src", "", exit_code=1)
    tb.bash("t2", "cargo test", "test failed", exit_code=101)
    sigs = find_signals(parse(tb, tmp_path).events)
    assert sig_kinds(sigs) == ["tool-error"]


def test_error_streak_fires_once_per_run(tb, tmp_path):
    for i in range(5):
        tb.bash(f"t{i}", f"python build.py --step {i}", "Traceback ...", exit_code=1)
    sigs = find_signals(parse(tb, tmp_path).events)
    assert sig_kinds(sigs).count("tool-error") == 5
    assert sig_kinds(sigs).count("error-streak") == 1


def test_self_correction_strong_and_weak(tb, tmp_path):
    tb.agent("Turns out the root cause was the backslash being mangled by MSYS.")
    tb.agent("Unexpectedly slow; surprisingly the cache was stale.")
    tb.agent("Unexpected.")
    sigs = find_signals(parse(tb, tmp_path).events)
    assert sig_kinds(sigs) == ["self-correction", "surprise"]
    assert sigs[0].weight >= 1.5


def test_timeout_interrupt_harness_and_peer(tb, tmp_path):
    tb.bash("t1", "cargo test", "", timedOutAfterMs=600000)
    tb.human("[Request interrupted by user for tool use]")
    tb.raw({"type": "system", "subtype": "x", "content": "safeguards stopped the response", "timestamp": "t", "uuid": "q"})
    tb.human("Another Claude session sent a message: lane A here", origin={"kind": "peer"})
    assert sig_kinds(find_signals(parse(tb, tmp_path).events)) == ["timeout", "interrupt", "harness-error", "peer-message"]


def test_compaction_summaries_only_when_enabled(tb, tmp_path):
    tb.human("This session is being continued from a previous conversation ...", isCompactSummary=True)
    events = parse(tb, tmp_path).events
    assert find_signals(events) == []
    assert sig_kinds(find_signals(events, mine_compact=True)) == ["compact-summary"]


def _session_with_two_incidents(tb):
    tb.human("please build the kernel and run the tests now")
    for i in range(40):
        tb.bash(f"a{i}", f"echo step {i}", "x" * 300)
    tb.human("no, never use rm -rf on target dirs through MSYS, use rd /s /q from a bat file")
    for i in range(40):
        tb.bash(f"b{i}", f"echo step {i}", "y" * 300)
    tb.agent("Turns out the earlier failure was a stale lock file.")


def test_windows_merge_split_and_score(tb, tmp_path):
    _session_with_two_incidents(tb)
    tr = parse(tb, tmp_path)
    sigs = find_signals(tr.events)
    wins = build_windows(tr, sigs, before_chars=2000, after_chars=3000, max_chars=100000, min_score=1.0)
    # three separate incidents, far apart: three windows, each starting with its header
    assert len(wins) == 3
    assert all(w.text.startswith("Session s | project proj") for w in wins)
    assert wins[1].kinds == ["human-correction"] and wins[1].score == 2.5
    # A small cap forces the one merged span to split; parts without a signal are dropped.
    small = build_windows(tr, sigs, before_chars=50000, after_chars=50000, max_chars=4000, min_score=1.0)
    header = len(small[0].text.split("\n\n", 1)[0]) + 2
    assert all(len(w.text) <= 4000 + header for w in small)
    assert {k for w in small for k in w.kinds} == {"human", "human-correction", "self-correction"}


def test_windows_below_threshold_dropped(tb, tmp_path):
    tb.bash("t1", "cargo build", "boom", exit_code=1)  # one tool error = 0.35
    tr = parse(tb, tmp_path)
    assert build_windows(tr, find_signals(tr.events), 2000, 3000, 30000, 1.0) == []


def test_window_ids_are_stable_and_text_hash_tracks_content(tb, tmp_path):
    tb.human("never commit secrets to the repository, ever")
    tb.agent("Understood.")
    tr = parse(tb, tmp_path)
    w1 = build_windows(tr, find_signals(tr.events), 2000, 3000, 30000, 1.0)[0]
    tb.agent("Also adding a pre-commit check now.")
    tr2 = parse(tb, tmp_path, "s.jsonl")
    w2 = build_windows(tr2, find_signals(tr2.events), 2000, 3000, 30000, 1.0)[0]
    assert w1.id == w2.id and w1.text_hash != w2.text_hash


def test_render_compacts_noise():
    read = T.Event(0, "t", T.RESULT, "a\nb\nc", tool="Read")
    edit = T.Event(1, "t", T.RESULT, "The file x has been updated", tool="Edit")
    err = T.Event(2, "t", T.RESULT, "Exit code 2\nboom\nShell cwd was reset to E:\\os", tool="Bash",
                  is_error=True, exit_code=2)
    assert render_event(read) == "[result] (3 lines of file content)"
    assert render_event(edit) == "[result] ok"
    assert render_event(err).startswith("[ERROR exit 2]") and "Shell cwd" not in render_event(err)
    assert render_event(T.Event(3, "t", T.LOOP, "Autonomous-loop wakeup")).startswith("[automatic prompt]")

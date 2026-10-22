from __future__ import annotations

from akn import transcripts as T


def kinds(tr):
    return [e.kind for e in tr.events]


def test_prompt_kinds_are_classified(tb, tmp_path):
    tb.human("please fix the build")
    tb.human("Autonomous-loop wakeup (scheduled by you). Resume work now: first check ...")
    tb.human("&lt;&lt;autonomous-loop-dynamic&gt;&gt;")
    tb.human("Your response above was cut off", isMeta=True)
    tb.human("Another Claude session sent a message: hello", origin={"kind": "peer"})
    tb.human("<task-notification><task-id>x</task-id></task-notification>", origin={"kind": "task-notification"})
    tb.human("This session is being continued from a previous conversation that ran out of context.",
             isCompactSummary=True)
    tb.human("[Request interrupted by user for tool use]")
    tb.human("<command-name>/clear</command-name>")
    tb.human("[orchestrator2] this session was resumed")
    tr = T.parse_file(tb.write(tmp_path / "s1.jsonl"), "claude", "proj")
    assert kinds(tr) == [T.HUMAN, T.LOOP, T.LOOP, T.META, T.PEER, T.NOTIFY, T.COMPACT, T.INTERRUPT,
                         T.COMMAND, T.PEER]


def test_repeated_long_prompt_is_demoted_to_loop(tb, tmp_path):
    template = "Continue lane work on the project; check background tasks first, then pick the next task."
    for _ in range(3):
        tb.human(template)
        tb.agent("ok")
    tb.human("one-off operator question that is long enough")
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    humans = [e for e in tr.events if e.kind in (T.HUMAN, T.LOOP)]
    assert [e.kind for e in humans] == [T.LOOP, T.LOOP, T.LOOP, T.HUMAN]


def test_two_repeats_are_not_a_template(tb, tmp_path):
    msg = "please re-run the full test suite and report what failed"
    tb.human(msg).agent("ok").human(msg)
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    assert [e.kind for e in tr.events if e.kind in (T.HUMAN, T.LOOP)] == [T.HUMAN, T.HUMAN]


def test_tool_calls_results_errors_and_exit_codes(tb, tmp_path):
    tb.bash("t1", "cargo build", "error[E0425]: cannot find value", exit_code=101)
    tb.call("Edit", "t2", file_path="E:/x.rs", old_string="fn a()\n{", new_string="fn b()")
    tb.result("t2", "<tool_use_error>String to replace not found</tool_use_error>", is_error=True)
    tb.bash("t3", "sleep 999", "", timedOutAfterMs=120000)
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    calls = [e for e in tr.events if e.kind == T.CALL]
    results = [e for e in tr.events if e.kind == T.RESULT]
    assert [c.tool for c in calls] == ["Bash", "Edit", "Bash"]
    assert calls[0].detail == "cargo build"
    assert calls[1].detail.startswith("E:/x.rs [replace 8 chars starting 'fn a()'")
    assert results[0].is_error and results[0].exit_code == 101 and results[0].detail == "cargo build"
    assert results[1].is_error and results[1].exit_code is None and results[1].tool == "Edit"
    assert results[2].timed_out and not results[2].is_error


def test_thinking_blocks_skipped_and_text_kept(tb, tmp_path):
    tb.thinking().agent("Found it: the path was mangled by MSYS.")
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    assert kinds(tr) == [T.AGENT]


def test_system_entries_refusals_and_api_errors(tb, tmp_path):
    tb.raw({"type": "system", "subtype": "model_refusal_no_fallback", "content": "", "timestamp": "t",
            "uuid": "s1", "apiRefusalCategory": "reasoning_extraction", "apiRefusalExplanation": "blocked"})
    tb.raw({"type": "system", "subtype": "compact_boundary", "content": "Conversation compacted", "timestamp": "t",
            "uuid": "s2"})
    tb.raw({"type": "assistant", "isApiErrorMessage": True, "uuid": "s3", "timestamp": "t",
            "message": {"content": [{"type": "text", "text": "API Error: overloaded"}]}})
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    assert kinds(tr) == [T.SYSTEM, T.SYSTEM]
    assert "reasoning_extraction" in tr.events[0].text
    assert tr.events[1].text.startswith("(api error)")


def test_sidechains_skipped_unless_requested(tb, tmp_path):
    tb.human("main prompt here")
    tb.human("subagent prompt", isSidechain=True)
    path = tb.write(tmp_path / "s.jsonl")
    assert len(T.parse_file(path, "c", "p").events) == 1
    assert len(T.parse_file(path, "c", "p", include_sidechains=True).events) == 2


def test_malformed_and_bookkeeping_lines_ignored_titles_kept(tb, tmp_path):
    tb.raw({"type": "ai-title", "aiTitle": "First title"})
    tb.human("hello there operator speaking")
    tb.lines.append('{"type": "user", "message": {"content": "trunc')  # a live session mid-write
    tb.raw({"type": "custom-title", "customTitle": "OS F"})
    tb.raw({"type": "queue-operation", "operation": "enqueue", "content": "dup of a prompt"})
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    assert kinds(tr) == [T.HUMAN]
    assert tr.title == "OS F"
    assert tr.cwd == "E:\\proj"


def test_large_results_are_clipped_at_parse_time(tb, tmp_path):
    big = "\n".join(f"line {i}" for i in range(20000))
    tb.bash("t1", "cat big.log", big)
    tr = T.parse_file(tb.write(tmp_path / "s.jsonl"), "claude", "p")
    res = [e for e in tr.events if e.kind == T.RESULT][0]
    assert len(res.text) < T.RESULT_HEAD_CHARS + T.RESULT_TAIL_CHARS + 100
    assert "chars cut" in res.text and res.text.startswith("line 0") and res.text.endswith("line 19999")


def test_discover_filters_projects(tmp_path):
    acct = tmp_path / ".claude-account-b"
    for proj in ("E--os", "D--other"):
        (acct / "projects" / proj).mkdir(parents=True)
        (acct / "projects" / proj / "a.jsonl").write_text("", encoding="utf-8")
    found = list(T.discover([acct], include=["E--*"], exclude=[]))
    assert [(p.name, a, pr) for p, a, pr in found] == [("a.jsonl", "account-b", "E--os")]
    assert list(T.discover([acct], include=[], exclude=["*os"]))[0][2] == "D--other"


def test_account_label():
    assert T.account_label(".claude") == "claude"
    assert T.account_label(".claude-account-c") == "account-c"


def test_summarize_call_shapes():
    assert T.summarize_call("Read", {"file_path": "a.md", "offset": 5, "limit": 9}) == "a.md (offset 5, limit 9)"
    assert T.summarize_call("Write", {"file_path": "b.py", "content": "xyz"}) == "b.py [3 chars]"
    assert "pattern='foo'" in T.summarize_call("Grep", {"pattern": "foo", "path": "src"})
    assert T.summarize_call("Bash", "not a dict") == ""

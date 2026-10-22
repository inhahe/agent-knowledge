"""Read Claude Code JSONL transcripts into a flat, ordered list of events.

Format facts this parser relies on (checked against real transcripts written by
Claude Code 2.x between 2026-06 and 2026-10):

* One JSON object per line. `type` is one of `user`, `assistant`, `system`,
  `attachment`, plus bookkeeping types (`queue-operation`, `ai-title`,
  `custom-title`, `last-prompt`, `atis-latch`, `cost-state`, `mode`, older
  `summary`) that carry no conversation and are skipped, apart from the titles.
* `user` entries hold either a prompt (`message.content` is a string or a list
  of text blocks) or tool output (`tool_result` blocks, paired to the call by
  `tool_use_id`). Human prompts arrive with `promptSource: "sdk"` from the SDK
  front end; injected harness text has `isMeta: true`; background-task
  notifications have `origin.kind: "task-notification"`; messages from other
  sessions have `origin.kind: "peer"`; compaction summaries carry
  `isCompactSummary: true`. Older files lack `promptSource`/`origin`, so the
  text prefixes are checked as well.
* `assistant` entries hold one content block each: `text`, `tool_use`, or
  `thinking` (whose text is empty in these files: only a signature is stored).
* Failed tool calls have `is_error: true`; Bash/PowerShell failures start with
  "Exit code N".

Transcripts are large (a single long session reaches 300 MB, nearly all of it
tool output), so tool output is clipped at parse time to what any later stage
could use: the head and tail of each result.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Event kinds -----------------------------------------------------------------
HUMAN = "human"  # a prompt the operator typed
LOOP = "loop"  # a machine-sent prompt: loop wakeups, launcher templates
AGENT = "agent"  # text the agent showed
CALL = "call"  # a tool invocation
RESULT = "result"  # a tool's output
SYSTEM = "system"  # harness notices: refusals, API errors, informational lines
META = "meta"  # injected text: classifier notices, "continue", hook output
PEER = "peer"  # a message from another Claude session
NOTIFY = "notify"  # a background task finished
INTERRUPT = "interrupt"  # the operator interrupted the agent
COMPACT = "compact"  # a context-compaction summary
COMMAND = "command"  # a slash-command echo (/clear, /model, ...)

RESULT_HEAD_CHARS = 6000
RESULT_TAIL_CHARS = 3000

_EXIT_RE = re.compile(r"^Exit code (-?\d+)")
_INTERRUPT_PREFIX = "[Request interrupted"
_PEER_PREFIX = "Another Claude session sent a message"
_COMMAND_PREFIXES = ("<command-name>", "<command-message>", "<local-command-stdout>", "<local-command-stderr>", "<command-args>")
_META_PREFIXES = ("Caveat: The messages below", "<system-reminder>", "<user-prompt-submit-hook>")
_NOTIFY_PREFIX = "<task-notification>"
_COMPACT_PREFIX = "This session is being continued from a previous conversation"
# Prompts that arrive through the human channel but are sent by a machine: the
# /loop wakeups (5,341 of the 9,064 "human" prompts in the first scan), the loop
# sentinel (sometimes HTML-escaped), and launcher templates.
_LOOP_PREFIXES = (
    "autonomous-loop wakeup",
    "<<autonomous-loop",
    "resume work now:",
    "continue autonomous",
    "continue the fastpy autonomous",
)
_RELAY_PREFIXES = ("[orchestrator2]",)  # relayed by the operator's orchestrator tool
# A prompt of this length or more that recurs this many times in one session is
# a template, not something the operator typed each time.
_TEMPLATE_MIN_LEN = 20
_TEMPLATE_MIN_REPEATS = 3


@dataclass(slots=True)
class Event:
    idx: int
    ts: str
    kind: str
    text: str
    uuid: str = ""
    tool: str = ""
    tool_id: str = ""
    is_error: bool = False
    exit_code: int | None = None
    # For CALL: a one-line summary of the arguments (the command, the file).
    # For RESULT: the same summary, copied from the call it answers.
    detail: str = ""
    timed_out: bool = False
    # For RESULT: the output's length before clipping (what the agent actually received).
    full_len: int = 0


@dataclass
class Transcript:
    path: Path
    account: str
    project: str
    session_id: str
    title: str = ""
    cwd: str = ""
    events: list[Event] = field(default_factory=list)

    @property
    def first_ts(self) -> str:
        return self.events[0].ts if self.events else ""

    @property
    def last_ts(self) -> str:
        return self.events[-1].ts if self.events else ""


def account_label(account_dir_name: str) -> str:
    """'.claude' -> 'claude', '.claude-account-b' -> 'account-b'."""
    name = account_dir_name.lstrip(".")
    return name.removeprefix("claude-") if name != "claude" else name


def discover(accounts: list[Path], include: list[str], exclude: list[str]) -> Iterator[tuple[Path, str, str]]:
    """Yield (jsonl path, account label, project dir name) for every main transcript."""
    for acct in accounts:
        root = acct / "projects"
        if not root.is_dir():
            continue
        for proj in sorted(root.iterdir()):
            if not proj.is_dir():
                continue
            if include and not any(fnmatch.fnmatch(proj.name, pat) for pat in include):
                continue
            if any(fnmatch.fnmatch(proj.name, pat) for pat in exclude):
                continue
            for f in sorted(proj.glob("*.jsonl")):
                yield f, account_label(acct.name), proj.name


def clip_middle(text: str, head: int, tail: int) -> str:
    """Keep the first `head` and last `tail` characters, marking the cut."""
    if len(text) <= head + tail + 40:
        return text
    cut = len(text) - head - tail
    return f"{text[:head]}\n... [{cut} chars cut] ...\n{text[-tail:]}"


def summarize_call(name: str, inp: object) -> str:
    """One line saying what a tool call did, without its bulky arguments."""
    if not isinstance(inp, dict):
        return ""
    if name in ("Bash", "PowerShell"):
        return str(inp.get("command", ""))
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(inp.get("file_path") or inp.get("notebook_path") or "")
        if name == "Read" and (inp.get("offset") or inp.get("limit")):
            return f"{path} (offset {inp.get('offset')}, limit {inp.get('limit')})"
        if name == "Edit":
            old = str(inp.get("old_string") or "")
            first = old.strip().splitlines()[0][:80] if old.strip() else ""
            return f"{path} [replace {len(old)} chars starting {first!r}]"
        if name == "Write":
            return f"{path} [{len(str(inp.get('content') or ''))} chars]"
        return path
    if name == "Grep":
        bits = [f"pattern={inp.get('pattern')!r}"]
        for key in ("path", "glob", "type", "output_mode"):
            if inp.get(key):
                bits.append(f"{key}={inp.get(key)}")
        return " ".join(bits)
    if name == "Glob":
        return f"{inp.get('pattern')} in {inp.get('path') or '.'}"
    if name == "WebFetch":
        return str(inp.get("url", ""))
    if name == "WebSearch":
        return str(inp.get("query", ""))
    if name in ("Agent", "Task"):
        return f"{inp.get('subagent_type', '')}: {inp.get('description', '')}"
    try:
        return json.dumps(inp, ensure_ascii=False)[:400]
    except (TypeError, ValueError):
        return ""


def _text_of(content: object) -> str:
    """Flatten a message/tool_result `content` (string or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def _classify_prompt(rec: dict, text: str) -> str:
    """Decide what kind of `user` text entry this is."""
    origin = rec.get("origin") if isinstance(rec.get("origin"), dict) else {}
    okind = origin.get("kind") if isinstance(origin, dict) else None
    stripped = text.lstrip()
    lowered = stripped.replace("&lt;", "<").replace("&gt;", ">").lower()
    if rec.get("isCompactSummary") or stripped.startswith(_COMPACT_PREFIX):
        return COMPACT
    if okind == "peer" or stripped.startswith(_PEER_PREFIX) or lowered.startswith(_RELAY_PREFIXES):
        return PEER
    if lowered.startswith(_LOOP_PREFIXES) or okind in ("scheduled", "loop", "cron"):
        return LOOP
    if okind == "task-notification" or stripped.startswith(_NOTIFY_PREFIX):
        return NOTIFY
    if stripped.startswith(_INTERRUPT_PREFIX):
        return INTERRUPT
    if stripped.startswith(_COMMAND_PREFIXES):
        return COMMAND
    if rec.get("isMeta") or stripped.startswith(_META_PREFIXES):
        return META
    if rec.get("promptSource") == "system":
        return META
    return HUMAN


def parse_file(path: Path, account: str, project: str, include_sidechains: bool = False) -> Transcript:
    """Parse one transcript file. Malformed lines (a live session mid-write) are skipped."""
    tr = Transcript(path=path, account=account, project=project, session_id=path.stem)
    calls: dict[str, tuple[str, str]] = {}  # tool_use_id -> (tool name, summary)
    events = tr.events

    def add(kind: str, ts: str, text: str, uuid: str, **kw) -> None:
        events.append(Event(idx=len(events), ts=ts, kind=kind, text=text, uuid=uuid, **kw))

    with open(path, "rb") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            if rtype in ("custom-title", "ai-title"):
                tr.title = str(rec.get("customTitle") or rec.get("aiTitle") or tr.title)
                continue
            if rtype not in ("user", "assistant", "system"):
                continue
            if rec.get("isSidechain") and not include_sidechains:
                continue
            ts = str(rec.get("timestamp") or "")
            uuid = str(rec.get("uuid") or "")
            if not tr.cwd and rec.get("cwd"):
                tr.cwd = str(rec.get("cwd"))

            if rtype == "system":
                text = str(rec.get("content") or "")
                category = rec.get("apiRefusalCategory")
                if category:
                    text = f"{text} [refusal: {category}] {rec.get('apiRefusalExplanation') or ''}".strip()
                subtype = str(rec.get("subtype") or "")
                if text and subtype not in ("compact_boundary", "turn_duration"):
                    add(SYSTEM, ts, f"({subtype}) {text}" if subtype else text, uuid)
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")

            if rtype == "assistant":
                if rec.get("isApiErrorMessage"):
                    add(SYSTEM, ts, "(api error) " + _text_of(content), uuid)
                    continue
                blocks = content if isinstance(content, list) else [{"type": "text", "text": _text_of(content)}]
                for i, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    buid = uuid if len(blocks) == 1 else f"{uuid}:{i}"
                    if btype == "text":
                        text = str(block.get("text") or "")
                        if text.strip():
                            add(AGENT, ts, text, buid)
                    elif btype == "tool_use":
                        name = str(block.get("name") or "")
                        tid = str(block.get("id") or "")
                        summary = summarize_call(name, block.get("input"))
                        calls[tid] = (name, summary)
                        add(CALL, ts, "", buid, tool=name, tool_id=tid, detail=summary)
                continue

            # rtype == "user"
            if isinstance(content, str):
                add(_classify_prompt(rec, content), ts, content, uuid)
                continue
            if not isinstance(content, list):
                continue
            tur = rec.get("toolUseResult") if isinstance(rec.get("toolUseResult"), dict) else {}
            text_blocks = []
            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                buid = uuid if len(content) == 1 else f"{uuid}:{i}"
                if btype == "tool_result":
                    tid = str(block.get("tool_use_id") or "")
                    name, summary = calls.get(tid, ("", ""))
                    text = _text_of(block.get("content"))
                    m = _EXIT_RE.match(text)
                    timed_out = bool(tur.get("timedOutAfterMs")) or bool(tur.get("interrupted"))
                    add(
                        RESULT,
                        ts,
                        clip_middle(text, RESULT_HEAD_CHARS, RESULT_TAIL_CHARS),
                        buid,
                        tool=name,
                        tool_id=tid,
                        is_error=bool(block.get("is_error")),
                        exit_code=int(m.group(1)) if m else None,
                        detail=summary,
                        timed_out=timed_out,
                        full_len=len(text),
                    )
                elif btype == "text":
                    text_blocks.append(str(block.get("text") or ""))
            if text_blocks:
                text = "\n".join(text_blocks)
                add(_classify_prompt(rec, text), ts, text, uuid)
    _demote_templates(events)
    return tr


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _demote_templates(events: list[Event]) -> None:
    """Reclassify HUMAN prompts that recur verbatim within the session as LOOP."""
    counts: dict[str, int] = {}
    for e in events:
        if e.kind == HUMAN and len(e.text) >= _TEMPLATE_MIN_LEN:
            key = _normalize(e.text)
            counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, n in counts.items() if n >= _TEMPLATE_MIN_REPEATS}
    if not repeated:
        return
    for e in events:
        if e.kind == HUMAN and len(e.text) >= _TEMPLATE_MIN_LEN and _normalize(e.text) in repeated:
            e.kind = LOOP

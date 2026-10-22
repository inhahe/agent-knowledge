"""Turn signals into bounded excerpts ("windows") the local model can read.

A window is a contiguous run of events around one or more signals: enough
before the signal to see what the agent was trying to do, and more after it to
see how the problem was resolved (the resolution is usually where the lesson
is). Overlapping windows merge; a merged window that grows past the size cap is
split, and each part must still contain a signal worth extracting.

Events are rendered compactly. Tool output is cut to its head and tail, file
contents read by the agent are reduced to a line count (they are never where the
lesson is), and successful edits to "ok". A compaction summary is not inlined
into ordinary windows (it would dominate them); it becomes a window of its own,
because its "errors and fixes" and "user messages" sections are a dense digest
of everything before it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace

from .signals import Signal
from .transcripts import (
    AGENT,
    CALL,
    COMMAND,
    COMPACT,
    HUMAN,
    INTERRUPT,
    LOOP,
    META,
    NOTIFY,
    PEER,
    RESULT,
    SYSTEM,
    Event,
    Transcript,
)

_QUIET_OK_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit", "TodoWrite", "ToolSearch", "ScheduleWakeup", "TaskStop"}


def clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]} ...[{len(text) - limit} chars cut]... {text[-tail:]}"


def clip_lines(text: str, head_lines: int, tail_lines: int, max_chars: int) -> str:
    lines = text.strip().splitlines()
    if len(lines) > head_lines + tail_lines + 2:
        cut = len(lines) - head_lines - tail_lines
        lines = lines[:head_lines] + [f"... [{cut} lines cut] ..."] + lines[-tail_lines:]
    return clip("\n".join(lines), max_chars)


_HARNESS_NOISE = re.compile(r"(?m)^Shell cwd was reset to .*$\n?|^Session cwd remains .*$\n?")


def render_event(e: Event) -> str:
    """One event as compact text for the model."""
    k = e.kind
    if k == RESULT and e.text:
        e = replace(e, text=_HARNESS_NOISE.sub("", e.text))
    if k == HUMAN:
        return f"[human] {clip(e.text, 2500)}"
    if k == AGENT:
        return f"[agent] {clip(e.text, 1800)}"
    if k == CALL:
        return f"[call {e.tool}] {clip(e.detail, 700)}"
    if k == RESULT:
        if e.is_error or e.timed_out:
            tag = "ERROR"
            if e.exit_code is not None:
                tag += f" exit {e.exit_code}"
            if e.timed_out:
                tag += " timed out"
            return f"[{tag}] {clip_lines(e.text, 14, 10, 1800)}"
        if e.tool == "Read":
            return f"[result] ({e.text.count(chr(10)) + 1} lines of file content)"
        if e.tool in _QUIET_OK_TOOLS:
            return "[result] ok"
        if e.tool == "Glob":
            return f"[result] {e.text.count(chr(10)) + 1} paths"
        return f"[result] {clip_lines(e.text, 10, 6, 1200)}"
    if k == SYSTEM:
        return f"[system] {clip(e.text, 600)}"
    if k == META:
        return f"[note] {clip(e.text, 500)}"
    if k == PEER:
        return f"[message from another agent] {clip(e.text, 1500)}"
    if k == NOTIFY:
        return f"[background task finished] {clip(e.text, 300)}"
    if k == INTERRUPT:
        return f"[OPERATOR INTERRUPTED] {clip(e.text, 200)}"
    if k == COMMAND:
        return f"[slash command] {clip(e.text, 200)}"
    if k == LOOP:
        return f"[automatic prompt] {clip(e.text, 160)}"
    if k == COMPACT:
        return "[compaction summary of earlier context: omitted here]"
    return f"[{k}] {clip(e.text, 300)}"


@dataclass
class Window:
    session_id: str
    path: str
    account: str
    project: str
    title: str
    start_idx: int
    end_idx: int
    start_ts: str
    end_ts: str
    score: float
    kinds: list[str]
    anchor_uuids: list[str]
    text: str
    id: str = field(default="")
    text_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.session_id}|{self.start_idx}|{self.anchor_uuids[0] if self.anchor_uuids else ''}"
            self.id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        if not self.text_hash:
            self.text_hash = hashlib.sha1(self.text.encode("utf-8")).hexdigest()


def _header(tr: Transcript, start_ts: str, end_ts: str) -> str:
    title = f' "{tr.title}"' if tr.title else ""
    return (
        f"Session {tr.session_id[:8]}{title} | project {tr.project} | account {tr.account} | "
        f"{start_ts[:16].replace('T', ' ')} .. {end_ts[:16].replace('T', ' ')} UTC"
    )


def build_windows(
    tr: Transcript,
    signals: list[Signal],
    before_chars: int,
    after_chars: int,
    max_chars: int,
    min_score: float,
) -> list[Window]:
    """Group a transcript's signals into scored, rendered windows."""
    events = tr.events
    if not events or not signals:
        return []
    rendered = [render_event(e) for e in events]
    lengths = [len(r) + 1 for r in rendered]
    windows: list[Window] = []

    # 1. Compaction summaries stand alone.
    plain: list[Signal] = []
    for s in signals:
        if s.kind == "compact-summary":
            e = events[s.idx]
            body = clip(e.text, max_chars - 400)
            text = _header(tr, e.ts, e.ts) + "\n(This is the agent's own summary of an earlier part of the session.)\n\n" + body
            windows.append(
                Window(
                    session_id=tr.session_id, path=str(tr.path), account=tr.account, project=tr.project,
                    title=tr.title, start_idx=s.idx, end_idx=s.idx, start_ts=e.ts, end_ts=e.ts,
                    score=s.weight, kinds=[s.kind], anchor_uuids=[e.uuid], text=text,
                )
            )
        else:
            plain.append(s)

    # 2. A span per signal, then merge overlapping spans.
    spans: list[list[int]] = []  # [start, end, first_signal_pos, last_signal_pos]
    for pos, s in enumerate(plain):
        start, budget = s.idx, before_chars
        while start > 0 and budget - lengths[start - 1] > 0:
            start -= 1
            budget -= lengths[start]
        end, budget = s.idx, after_chars
        while end < len(events) - 1 and budget - lengths[end + 1] > 0:
            end += 1
            budget -= lengths[end]
        if spans and start <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], end)
            spans[-1][3] = pos
        else:
            spans.append([start, end, pos, pos])

    # 3. Split spans that render too large, keeping only parts with enough signal.
    for start, end, p0, p1 in spans:
        sigs = plain[p0 : p1 + 1]
        cur_start = start
        while cur_start <= end:
            size, cur_end = 0, cur_start
            while cur_end <= end and size + lengths[cur_end] <= max_chars:
                size += lengths[cur_end]
                cur_end += 1
            cur_end = max(cur_end - 1, cur_start)
            inside = [s for s in sigs if cur_start <= s.idx <= cur_end]
            score = round(sum(s.weight for s in inside), 2)
            if inside and score >= min_score:
                body = "\n".join(rendered[cur_start : cur_end + 1])
                text = _header(tr, events[cur_start].ts, events[cur_end].ts) + "\n\n" + body
                windows.append(
                    Window(
                        session_id=tr.session_id, path=str(tr.path), account=tr.account, project=tr.project,
                        title=tr.title, start_idx=cur_start, end_idx=cur_end,
                        start_ts=events[cur_start].ts, end_ts=events[cur_end].ts, score=score,
                        kinds=sorted({s.kind for s in inside}),
                        anchor_uuids=[events[s.idx].uuid for s in inside],
                        text=text,
                    )
                )
            cur_start = cur_end + 1
    windows.sort(key=lambda w: w.start_idx)
    return windows

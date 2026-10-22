"""Cheap heuristics that point at the parts of a transcript worth an LLM's time.

Reading every token of every transcript through the local model would take days
and would mostly be wasted: over 90% of transcript text is routine tool output.
Lessons cluster around a few kinds of moment, and each has a cheap signature:

* the operator says something (rare, and the richest source: corrections,
  standing instructions, preferences), above all when they correct the agent;
* the operator interrupts the agent;
* the agent says it was wrong, or explains a surprise ("turns out", "root
  cause", "mangled");
* tool calls fail several times in a row (the agent is fighting something);
* a command times out; the harness refuses or errors.

Each signal carries a weight; windows are built around signals and only windows
whose weights sum past a threshold are sent to the model. The weights are
judgement, tuned by reading what they select; `akn scan --show` prints samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .transcripts import AGENT, COMPACT, HUMAN, INTERRUPT, PEER, RESULT, SYSTEM, Event


@dataclass(slots=True)
class Signal:
    idx: int
    kind: str
    weight: float
    note: str = ""


# Operator text that corrects, forbids, or states a standing rule.
_CORRECTION = re.compile(
    r"(?ix)(?<![\w'])("
    r"no[,.!]|nope|don'?t|do\ not|stop\b|wrong|incorrect|why\ (did|are|is|would|do)\ you|"
    r"i\ (told|said|asked)|never|always|instead|not\ what|that'?s\ not|you\ should(n'?t)?|"
    r"please\ don'?t|again[?!.]|revert|undo|mistake|you\ broke|careful|you\ forgot|you\ missed|"
    r"didn'?t\ i|from\ now\ on|in\ (the\ )?future|every\ time|whenever|make\ sure|remember"
    r")"
)
# Operator text with no content worth mining on its own.
_TRIVIAL = re.compile(r"(?i)^\s*(go|continue|ok(ay)?|yes|y|proceed|keep going|do it|sure|thanks?|thank you|resume|next|carry on)\W*$")

# Agent text that admits a mistake or explains a surprise. Strong phrases are
# nearly always a lesson in the making; weak ones only add up.
_STRONG_SELF = re.compile(
    r"(?ix)("
    r"i\ was\ wrong|my\ mistake|i\ made\ a\ mistake|that\ was\ (wrong|a\ mistake)|"
    r"(it\ )?turns\ out|the\ (real|actual|true|root)\ (cause|problem|issue|reason|bug)|root\ cause|"
    r"the\ problem\ (was|is\ that)|mangl\w+|i\ misread|i\ assumed|wrong\ assumption|"
    r"on\ closer\ inspection|false\ (positive|alarm)|gotcha|note\ to\ self|lesson\ learned|"
    r"i\ should\ have|should\ have\ (checked|read|run|looked)|that\ explains|which\ explains|explains\ why|"
    r"i\ was\ (looking|reading)\ at\ the\ wrong|not\ what\ i\ expected"
    r")"
)
_WEAK_SELF = re.compile(
    r"(?ix)("
    r"unexpected(ly)?|surpris\w+|silently|stale|pitfall|workaround|the\ catch\ is|"
    r"actually,|in\ fact,|correction:|wait[,.]|hmm"
    r")"
)

# Exit status 1 from a search/compare command means "no match"/"differs", not failure.
_BENIGN_EXIT1 = re.compile(r"(?i)\b(grep|rg|egrep|fgrep|findstr|diff|cmp|test|git\s+diff|git\s+grep|select-string)\b|\[\s")


def _benign_error(e: Event) -> bool:
    if e.exit_code == 1 and _BENIGN_EXIT1.search(e.detail or ""):
        return True
    return False


def find_signals(events: list[Event], mine_compact: bool = False) -> list[Signal]:
    """Return every signal in one transcript, in event order.

    Compaction summaries are skipped unless `mine_compact`: in long autonomous
    sessions they recur every few hundred events (7,306 in the first scan, 96 M
    characters), each restating the last, and the raw events they summarise are
    already mined directly.
    """
    out: list[Signal] = []
    recent_errors: list[int] = []  # indices of recent non-benign tool errors
    streak_open = False
    results_seen = 0

    for e in events:
        if e.kind == HUMAN:
            text = e.text.strip()
            if not text or _TRIVIAL.match(text):
                continue
            weight = 1.0
            if _CORRECTION.search(text):
                weight += 1.5
            out.append(Signal(e.idx, "human-correction" if weight > 1 else "human", weight))
        elif e.kind == INTERRUPT:
            out.append(Signal(e.idx, "interrupt", 2.5))
        elif e.kind == AGENT:
            strong = len(_STRONG_SELF.findall(e.text))
            weak = len(_WEAK_SELF.findall(e.text))
            if strong:
                out.append(Signal(e.idx, "self-correction", min(1.5 + 0.3 * (strong - 1), 2.5)))
            elif weak >= 2:
                out.append(Signal(e.idx, "surprise", 0.5))
        elif e.kind == RESULT:
            results_seen += 1
            if e.timed_out:
                out.append(Signal(e.idx, "timeout", 1.0, e.tool))
            if e.is_error and not _benign_error(e):
                out.append(Signal(e.idx, "tool-error", 0.35, e.tool))
                recent_errors.append(results_seen)
                # Three failures within eight consecutive tool results: the agent
                # is fighting something. One streak signal per run of failures.
                recent_errors[:] = [r for r in recent_errors if results_seen - r < 8]
                if len(recent_errors) >= 3 and not streak_open:
                    out.append(Signal(e.idx, "error-streak", 1.2, e.tool))
                    streak_open = True
            elif not e.is_error and streak_open and recent_errors and results_seen - recent_errors[-1] >= 8:
                streak_open = False
                recent_errors.clear()
        elif e.kind == SYSTEM:
            low = e.text.lower()
            if "refusal" in low or "safeguard" in low or "api error" in low:
                out.append(Signal(e.idx, "harness-error", 0.8))
        elif e.kind == PEER:
            out.append(Signal(e.idx, "peer-message", 0.6))
        elif e.kind == COMPACT and mine_compact:
            # Mined as a window of its own (see windows.py).
            out.append(Signal(e.idx, "compact-summary", 1.5))
    return out

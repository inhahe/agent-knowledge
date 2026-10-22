"""Ask the local model what reusable lesson, if any, each window teaches.

The reply is constrained by a JSON schema (llama-server compiles it into a
grammar), so it always parses. Property order is deliberate: the model must
quote its evidence before it states the lesson, which keeps it from inventing a
lesson and then looking for support.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import Client, ServerError
from .store import Store, window_text

KINDS = [
    "operator-instruction",  # a standing rule or preference the operator stated
    "operator-correction",  # the operator corrected something the agent did
    "tool-quirk",  # a tool/CLI behaved unexpectedly
    "shell-quoting",  # quoting, escaping, path conversion between shells
    "windows-environment",  # Windows specifics: encodings, paths, processes, drives
    "git-workflow",
    "build-and-test",
    "debugging-method",  # a way of finding a problem that worked, or misled
    "multi-agent-coordination",  # several agents/sessions sharing a repo or machine
    "process",  # how to plan, verify, report, or pace work
    "project-convention",  # a rule specific to one project's codebase or docs
    "other",
]
SCOPES = ["this-machine", "all-projects", "this-project", "one-subsystem"]
SOURCES = ["operator", "agent-discovered", "tool-failure"]

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "title": {"type": "string"},
        "lesson": {"type": "string"},
        "trigger": {"type": "string"},
        "kind": {"type": "string", "enum": KINDS},
        "scope": {"type": "string", "enum": SCOPES},
        "source": {"type": "string", "enum": SOURCES},
        "confidence": {"type": "number"},
    },
    "required": ["evidence", "title", "lesson", "trigger", "kind", "scope", "source", "confidence"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {"lessons": {"type": "array", "items": LESSON_SCHEMA, "maxItems": 3}},
    "required": ["lessons"],
    "additionalProperties": False,
}

SYSTEM = """You read excerpts from the logs of Claude Code sessions: an AI coding agent working on a Windows 11 PC, mostly unattended, supervised by a human operator who checks in occasionally. The agent uses Git Bash (MSYS), PowerShell, git, cargo, Python and file read/edit tools. Several agent sessions often run at once on the same machine and repository.

Your job: find LESSONS in the excerpt. A lesson is knowledge that would have saved the agent time or prevented a mistake, written so that a future agent can act on it.

A good lesson is
- reusable: it will matter again in future sessions, not only in this task;
- specific and actionable: it names the tool, command, file, or situation and says what to do;
- grounded: the excerpt shows it. Quote the evidence verbatim (short).

Look especially for:
1. The operator corrects the agent, forbids something, or states a standing preference or rule. These are the most valuable; record the operator's intent faithfully.
2. A tool, shell, or the OS behaved unexpectedly (quoting, path conversion, encodings, timeouts, Windows quirks) and the cause became clear.
3. The agent took a wrong approach, found out, and changed course: what should it have done first?
4. A checking or debugging method that found a real problem, or one that gave a misleading result.
5. Coordination failures between agents or sessions; git workflow mistakes.

These are NOT lessons: routine progress; facts about the current task's code that will soon change; a one-off typo; a search that found nothing; descriptions of what was done; generic advice such as "write tests" or "be careful".

Fields, and keep every one short: evidence = a verbatim quote from the excerpt, at most 30 words. title = at most 10 words. lesson = 1-2 sentences and at most 50 words, imperative, self-contained (do not refer to "the excerpt" or "this session"). trigger = when the lesson applies, at most 15 words. scope = how widely it applies. source = who supplied the lesson. confidence = 0..1, how sure you are this is a real, reusable lesson.

Most excerpts contain no lesson or one lesson; never more than three. If there is none, return an empty list. Never invent a lesson that the excerpt does not show."""


def user_prompt(excerpt: str) -> str:
    return (
        "Excerpt (events in order. [human] = the operator; [agent] = the agent's text; [call X] = a tool call; "
        "[result] / [ERROR] = tool output; [note], [system] = messages from the harness):\n\n"
        f"{excerpt}\n\nReturn the lessons this excerpt teaches (often none)."
    )


def _clean(lessons: list[dict]) -> list[dict]:
    out = []
    for les in lessons:
        text = str(les.get("lesson", "")).strip()
        if not text:
            continue
        les = {k: (v.strip() if isinstance(v, str) else v) for k, v in les.items()}
        try:
            les["confidence"] = min(max(float(les.get("confidence", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            les["confidence"] = 0.0
        out.append(les)
    return out


def extract_one(client: Client, text: str, thinking: bool) -> tuple[list[dict], int, int, float]:
    """Extract lessons from one window, retrying once with more room if the reply was cut off."""
    max_tokens = 1800
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            res = client.chat_json(SYSTEM, user_prompt(text), SCHEMA, max_tokens=max_tokens,
                                   temperature=0.3, thinking=thinking)
            return _clean(res.data.get("lessons", [])), res.prompt_tokens, res.completion_tokens, res.seconds
        except ValueError as exc:  # truncated or unparseable reply
            last_exc = exc
            max_tokens *= 2
        except ServerError as exc:
            last_exc = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"extraction failed after retries: {last_exc}")


def run(store: Store, client: Client, model_name: str, workers: int, thinking: bool,
        limit: int | None = None, retry_failed: bool = False) -> dict:
    rows = store.pending_windows(limit=limit, retry_failed=retry_failed)
    total = len(rows)
    if not total:
        print("no pending windows", flush=True)
        return {"windows": 0}
    total_chars = sum(r["n_chars"] for r in rows)
    print(f"extracting {total} windows ({total_chars / 1e6:.1f} M chars) with {workers} parallel requests", flush=True)
    t0 = time.time()
    done = failed = n_lessons = p_tok = c_tok = 0
    chars_done = 0
    last_print = t0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, client, window_text(r), thinking): r for r in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                lessons, pt, ct, secs = fut.result()
                store.finish_window(row["id"], lessons, model_name, pt, ct, secs)
                done += 1
                n_lessons += len(lessons)
                p_tok += pt
                c_tok += ct
            except Exception as exc:  # noqa: BLE001 - record and continue; one bad window must not stop the run
                store.fail_window(row["id"], repr(exc))
                failed += 1
            chars_done += row["n_chars"]
            now = time.time()
            if now - last_print >= 30 or done + failed == total:
                last_print = now
                el = now - t0
                rate = chars_done / el if el else 0
                eta = (total_chars - chars_done) / rate / 60 if rate else 0
                print(
                    f"  {done + failed}/{total} windows ({failed} failed), {n_lessons} lessons, "
                    f"{p_tok / el:.0f} prompt tok/s, {c_tok / el:.1f} gen tok/s, ~{eta:.0f} min left",
                    flush=True,
                )
    return {"windows": total, "done": done, "failed": failed, "lessons": n_lessons,
            "prompt_tokens": p_tok, "completion_tokens": c_tok, "seconds": time.time() - t0}

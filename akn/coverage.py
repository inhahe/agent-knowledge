"""Is a lesson already a rule? Compare lessons with the instruction files.

The instruction files (every CLAUDE.md the agents load, the files those import,
plus Claude Code's memory files) are split into rule-sized chunks: a markdown section is cut into its
top-level bullets and paragraphs, each tagged with its heading path. For each
consolidated lesson the nearest chunks are retrieved by embedding, and the
model judges whether any of them already tells an agent the same thing.

A lesson that is covered but still recurring in recent sessions is the most
interesting output of the whole tool: the rule exists and is not working.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from . import gitread, imports
from .config import Config

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_ITEM = re.compile(r"^(\s{0,3})([-*+]|\d+[.)])\s+")
MAX_CHUNK = 1600


@dataclass
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def key(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


def instruction_sources(cfg: Config) -> list[tuple[str, str]]:
    """(label, text) for every instruction file, and every file one imports,
    deduplicated by content."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    skip = [str(Path(s)).lower() for s in cfg.instruction_skip]

    def skipped(p: Path) -> bool:
        return any(fnmatch.fnmatch(str(p).lower(), s) for s in skip)

    def add(label: str, text: str) -> None:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append((label, text))

    for pattern in cfg.instruction_globs:
        for name in sorted(glob.glob(pattern)):
            p = Path(name)
            if skipped(p) or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            add(str(p), text)
            for f in imports.follow(p, text).files:
                if not skipped(f.path):
                    add(str(f.path), f.text)
    for repo, rev, path in cfg.instruction_git:
        text = gitread.show(repo, rev, path)
        if text is None:
            continue
        add(f"{repo}/{path} @ {rev}", text)
        # What it imports is read from the checkout: a relative mention names a file there.
        for f in imports.follow(Path(repo) / path, text).files:
            if not skipped(f.path):
                add(str(f.path), f.text)
    return out


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """Split markdown into top-level bullets and paragraphs under their headings."""
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < 25:
            return
        heading = " > ".join(path)
        while len(body) > MAX_CHUNK:
            cut = body.rfind("\n", 0, MAX_CHUNK)
            cut = cut if cut > MAX_CHUNK // 2 else MAX_CHUNK
            chunks.append(Chunk(source, heading, body[:cut].strip()))
            body = body[cut:].strip()
        if body:
            chunks.append(Chunk(source, heading, body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        if in_code:
            buf.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
            continue
        if not line.strip():
            # A blank line ends a paragraph, but not a list item's continuation.
            if buf and not _ITEM.match(buf[0]):
                flush()
            continue
        item = _ITEM.match(line)
        if item and len(item.group(1)) == 0:
            flush()  # each top-level item is its own chunk
        buf.append(line)
    flush()
    return chunks


def load_chunks(cfg: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for source, text in instruction_sources(cfg):
        for c in chunk_markdown(source, text):
            if c.key not in seen:
                seen.add(c.key)
                chunks.append(c)
    return chunks


COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["covered", "partly-covered", "not-covered"]},
        "best_rule": {"type": "integer"},
        "missing": {"type": "string"},
    },
    "required": ["reasoning", "verdict", "best_rule", "missing"],
    "additionalProperties": False,
}

COVERAGE_SYSTEM = """You compare a lesson that AI coding agents learned with the instructions those agents are already given. Decide whether any of the numbered existing instructions already tells an agent what the lesson says.

covered = an existing instruction states the same advice (wording may differ).
partly-covered = an instruction addresses the same situation but leaves out something the lesson adds.
not-covered = none of them gives this advice.

reasoning: one or two sentences. best_rule: the number of the closest instruction, or -1 if none is relevant. missing: for partly-covered, what the lesson adds; otherwise an empty string."""


def coverage_prompt(lesson: str, chunks: list[Chunk]) -> str:
    rules = "\n\n".join(
        f"[{i}] (from {Path(c.source.split(' @ ')[0]).name}, section: {c.heading or '-'})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return f"LESSON:\n{lesson}\n\nEXISTING INSTRUCTIONS:\n{rules}"

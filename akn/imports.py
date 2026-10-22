"""Follow a CLAUDE.md's `@path` imports, the way Claude Code does.

Claude Code also loads every file an instruction file names as `@path` -- at the
start of a line, after whitespace, or right after a code span or a comment --
unless the mention sits in a code span, a fenced code block or an HTML comment.
`@~/x.md` is read from the home directory; `@x.md`, `@./x.md` and `@../x.md` from
the directory of the file that names it; an absolute path as it stands. An
imported file's own imports are followed too, each file once, four levels deep at
most. A file that is not where its mention says is skipped without a word.

So whatever reads an instruction file to learn the rules a session is given must
read what it imports as well: the rules every account shares live in a file that
each account's CLAUDE.md imports. And because Claude Code skips an unreadable
import in silence, `follow` reports each one, so that a moved or deleted shared
file is noticed by something. Matched against Claude Code 2.1.280's loader.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# Claude Code's loader refuses a file this many imports below the file it started
# from, which is at depth 0: four levels of imports at most.
MAX_DEPTH = 5

# A fence opens with three or more backticks, whose info string holds no backtick,
# or tildes; indented any amount or quoted, as inside a list item or a blockquote.
_FENCE = re.compile(r"[ \t>]*(`{3,}(?=[^`]*$)|~{3,})")
# A code span: a run of backticks, text that holds neither that run nor a blank line, the same run.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?:(?!\1)(?!\n[ \t]*\n)[\s\S])+?\1(?!`)")
_COMMENT = re.compile(r"<!--[\s\S]*?-->")
# A backslash may appear in a mention only to escape a space.
_MENTION = re.compile(r"(?:^|(?<=\s))@((?:[^\s\\]|\\ )+)", re.M)
_PATH_START = re.compile(r"[A-Za-z0-9._-]")
_MSYS_DRIVE = re.compile(r"/([A-Za-z])(?:/|$)")


def _prose(text: str) -> str:
    """`text` with its fenced code blocks dropped, and each code span and HTML comment
    replaced by a space. A space, not nothing: Claude Code matches mentions in each run
    of plain text on its own, so the end of a code span or a comment counts as a start,
    like whitespace. Code spans go first, so a span that shows `<!--` opens no comment."""
    kept: list[str] = []
    fence = ""
    for line in text.split("\n"):
        if fence:
            closing = line.lstrip(" \t>").rstrip()
            if closing.startswith(fence) and closing == fence[0] * len(closing):
                fence = ""
            continue
        opening = _FENCE.match(line)
        if opening:
            fence = opening.group(1)
            continue
        kept.append(line)
    return _COMMENT.sub(" ", _CODE_SPAN.sub(" ", "\n".join(kept)))


def mentions(text: str) -> list[str]:
    """The paths `text` imports, as written there (a `#fragment` dropped, an escaped
    space unescaped), in order, each once."""
    out: list[str] = []
    for m in _MENTION.finditer(_prose(text)):
        path = m.group(1).split("#", 1)[0].replace("\\ ", " ")
        if path and path not in out and (path.startswith("~/") or (path.startswith("/") and path != "/")
                                          or _PATH_START.match(path)):
            out.append(path)
    return out


def resolve(mention: str, base: Path, home: Path | None = None) -> Path:
    """The file `mention` names when it is written in a file in directory `base`;
    `home` is the directory `~` stands for (the user's, unless given)."""
    if mention.startswith("~/"):
        return Path(os.path.normpath((config.HOME if home is None else home) / mention[2:]))
    if sys.platform == "win32":
        drive = _MSYS_DRIVE.match(mention)
        if drive:
            mention = f"{drive.group(1).upper()}:/{mention[drive.end():]}"
    path = Path(mention)
    return Path(os.path.normpath(path if path.is_absolute() else base / path))


def _same_file_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _shown(value: object) -> str:
    """`value` as text for a message, each control character escaped: a mention can hold one."""
    return "".join(c if c >= " " else f"\\x{ord(c):02x}" for c in str(value))


@dataclass
class Imported:
    importer: Path  # the file whose mention named it
    path: Path
    text: str


@dataclass
class Imports:
    files: list[Imported] = field(default_factory=list)  # in the order Claude Code loads them
    unreadable: list[str] = field(default_factory=list)  # each mention that names no readable file, described


def follow(path: Path, text: str, home: Path | None = None) -> Imports:
    """What the instruction file `path`, whose text is `text`, imports -- directly or
    through the files it imports -- in the order Claude Code loads it: depth first,
    each file once. `home` is the directory `~` stands for. A mention whose file
    cannot be read is described in `unreadable` instead."""
    out = Imports()
    seen = {_same_file_key(path)}

    def visit(file: Path, body: str, depth: int) -> None:
        if depth + 1 >= MAX_DEPTH:
            return
        for mention in mentions(body):
            target = resolve(mention, file.parent, home)
            key = _same_file_key(target)
            if key in seen:
                continue
            try:
                target_text = target.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError) as exc:  # ValueError: a path no file can have, holding a NUL
                why = getattr(exc, "strerror", None) or exc
                problem = f"{file} imports @{_shown(mention)}, but {_shown(target)} cannot be read ({why})"
                if problem not in out.unreadable:
                    out.unreadable.append(problem)
                continue
            seen.add(key)
            out.files.append(Imported(file, target, target_text))
            visit(target, target_text, depth + 1)

    visit(path, text, 0)
    return out


def expand(files: list[tuple[Path, str]], home: Path | None = None) -> str:
    """What sessions reading the instruction files `files` -- (path, text) pairs -- are
    told: each distinct text once, under a line naming the files that hold it, and after
    it every file it imports that no earlier text did, under a line naming it and the
    file that imports it."""
    holders: dict[str, list[Path]] = {}
    for path, text in files:
        holders.setdefault(text, []).append(path)
    parts: list[str] = []
    shown: set[str] = set()
    for text, paths in holders.items():
        body = text.strip("\n")
        parts.append(f"(In {', '.join(str(p) for p in paths)})\n\n{body}")
        for f in follow(paths[0], text, home).files:
            key = _same_file_key(f.path)
            if key not in shown:
                shown.add(key)
                body = f.text.strip("\n")
                parts.append(f"(Imported by {f.importer}: {f.path})\n\n{body}")
    return "\n\n".join(parts) + "\n"

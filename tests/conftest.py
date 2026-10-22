"""Shared fixtures: synthetic transcripts in the real Claude Code JSONL shape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TranscriptBuilder:
    """Build JSONL lines the way Claude Code writes them (field names verified on real files)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n = 0

    def _ts(self) -> str:
        self.n += 1
        return f"2026-09-20T10:{self.n // 60:02d}:{self.n % 60:02d}.000Z"

    def _uuid(self) -> str:
        return f"u-{self.n:05d}"

    def raw(self, obj: dict) -> "TranscriptBuilder":
        self.lines.append(json.dumps(obj))
        return self

    def human(self, text: str, **extra) -> "TranscriptBuilder":
        ts = self._ts()
        rec = {"type": "user", "uuid": self._uuid(), "timestamp": ts, "promptSource": "sdk", "cwd": "E:\\proj",
               "message": {"role": "user", "content": text}}
        rec.update(extra)
        return self.raw(rec)

    def agent(self, text: str) -> "TranscriptBuilder":
        ts = self._ts()
        return self.raw({"type": "assistant", "uuid": self._uuid(), "timestamp": ts,
                         "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})

    def thinking(self) -> "TranscriptBuilder":
        ts = self._ts()
        return self.raw({"type": "assistant", "uuid": self._uuid(), "timestamp": ts,
                         "message": {"content": [{"type": "thinking", "thinking": "", "signature": "x"}]}})

    def call(self, tool: str, tid: str, **inp) -> "TranscriptBuilder":
        ts = self._ts()
        return self.raw({"type": "assistant", "uuid": self._uuid(), "timestamp": ts,
                         "message": {"content": [{"type": "tool_use", "id": tid, "name": tool, "input": inp}]}})

    def result(self, tid: str, text: str, is_error: bool = False, **tur) -> "TranscriptBuilder":
        ts = self._ts()
        rec = {"type": "user", "uuid": self._uuid(), "timestamp": ts,
               "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": text,
                                        "is_error": is_error}]}}
        if tur:
            rec["toolUseResult"] = tur
        return self.raw(rec)

    def bash(self, tid: str, command: str, output: str, exit_code: int | None = None, **tur) -> "TranscriptBuilder":
        self.call("Bash", tid, command=command)
        if exit_code:
            return self.result(tid, f"Exit code {exit_code}\n{output}", is_error=True, **tur)
        return self.result(tid, output, **tur)

    def write(self, path: Path) -> Path:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path


@pytest.fixture
def tb() -> TranscriptBuilder:
    return TranscriptBuilder()


def git_repo(path: Path, files: dict[str, str], commit: list[str] | None = None) -> Path:
    """A git repository at `path` holding `files` (name -> text, written with \\n line
    endings), with those named in `commit` (all of them by default) committed."""
    path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text, encoding="utf-8", newline="\n")
    names = list(files) if commit is None else commit
    for args in (["init", "-q"], ["add", "--", *names],
                 ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    return path

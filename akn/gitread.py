"""Read a file as a git revision holds it, or whether git tracks it."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def clean_env() -> dict[str, str]:
    """This process's environment without GIT_* variables. A GIT_DIR or GIT_WORK_TREE
    exported by a hook or a shell outranks `git -C <dir>`, and would have git answer
    for some other repository."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}


def show(repo: str | Path, rev: str, path: str, timeout: float = 60) -> str | None:
    """The text of `path` at revision `rev` of the repository `repo`, or None when git
    cannot give it: no such revision or file, no git, or no answer within `timeout`
    seconds -- the nightly run and the curator run unattended, and a git that hangs
    must not hold them up."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True,
                             timeout=timeout, env=clean_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else None


def tracked(path: Path, timeout: float = 60) -> bool:
    """Whether git tracks the file `path` in the repository its folder is in. False
    when it is in none, or git cannot say within `timeout` seconds."""
    try:
        out = subprocess.run(["git", "-C", str(path.parent), "ls-files", "--error-unmatch", "--", path.name],
                             capture_output=True, timeout=timeout, env=clean_env())
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0

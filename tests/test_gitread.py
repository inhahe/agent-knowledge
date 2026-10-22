"""Reading a file as a git revision holds it."""

from __future__ import annotations

import subprocess

from akn import gitread


def test_show_reads_a_revision_and_answers_none_when_git_cannot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("first\n", encoding="utf-8", newline="\n")
    for args in (["init", "-q"], ["add", "a.md"],
                 ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    (repo / "a.md").write_text("changed since\n", encoding="utf-8", newline="\n")
    assert gitread.show(repo, "HEAD", "a.md") == "first\n"
    assert gitread.show(repo, "HEAD", "missing.md") is None
    assert gitread.show(tmp_path / "not-a-repo", "HEAD", "a.md") is None

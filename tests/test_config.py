"""The paths: everything below the projects folder and the AI folder is
derived from those two, so that moving either is one setting, and nothing
names a drive the next machine may not have. Paths into the projects folder
may be given relative to it, and follow it when it moves."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from akn import config as config_mod
from tests.conftest import git_repo


def test_the_projects_paths_derive_from_projects_dir(tmp_path):
    c = config_mod.Config(projects_dir=tmp_path)
    p = tmp_path.as_posix()

    assert c.workspace_file == tmp_path / "CLAUDE.md"
    assert f"{p}/CLAUDE.md" in c.instruction_globs and f"{p}/*/CLAUDE.md" in c.instruction_globs
    assert c.docs_repo is None and c.instruction_git == [] and c.instruction_skip == []


def test_the_home_patterns_start_from_the_home_folder(tmp_path):
    c = config_mod.Config(projects_dir=tmp_path, instruction_globs=["~/.claude*/CLAUDE.md", "~\\x.md"])

    home = config_mod.HOME.as_posix()
    assert c.instruction_globs == [f"{home}/.claude*/CLAUDE.md", f"{home}/x.md"]


def test_relative_project_paths_resolve_in_the_projects_folder(tmp_path):
    c = config_mod.Config(projects_dir=tmp_path, docs_repo=Path("repo"),
                          instruction_git=[["repo", "origin/main", "CLAUDE.md"]],
                          instruction_skip=["repo/CLAUDE.md", "repo-*/CLAUDE.md"])
    p = tmp_path.as_posix()

    assert c.docs_repo == tmp_path / "repo"
    assert c.instruction_git == [[f"{p}/repo", "origin/main", "CLAUDE.md"]]
    assert c.instruction_skip == [f"{p}/repo/CLAUDE.md", f"{p}/repo-*/CLAUDE.md"]


def test_an_absolute_path_is_kept(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    c = config_mod.Config(projects_dir=tmp_path / "projects", docs_repo=elsewhere,
                          instruction_globs=[f"{elsewhere.as_posix()}/*/CLAUDE.md"])

    assert c.docs_repo == elsewhere
    assert c.instruction_globs == [f"{elsewhere.as_posix()}/*/CLAUDE.md"]


def test_the_model_paths_derive_from_ai_dir(tmp_path):
    c = config_mod.Config(ai_dir=tmp_path)

    assert c.llama_server == tmp_path / "llama.cpp" / "b11344" / "llama-server.exe"
    assert c.llm_model.is_relative_to(tmp_path) and c.embed_model.is_relative_to(tmp_path)


def test_config_toml_moving_the_projects_moves_what_is_relative_to_them(tmp_path):
    toml = tmp_path / "config.toml"
    moved = tmp_path / "projects"
    toml.write_text(f'projects_dir = "{moved.as_posix()}"\n'
                    'docs_repo = "repo"\n'
                    'instruction_skip = ["repo/CLAUDE.md"]\n'
                    'curator_config_dir = "~/.claude-x"\n'
                    'curated_accounts = ["~/.claude-x"]\n', encoding="utf-8")

    c = config_mod.load(toml)

    assert c.projects_dir == moved and c.docs_repo == moved / "repo"
    assert c.instruction_skip == [f"{moved.as_posix()}/repo/CLAUDE.md"]
    assert c.curator_config_dir == config_mod.HOME / ".claude-x"
    assert c.global_files == [config_mod.HOME / ".claude-x" / "CLAUDE.md"]


def test_a_folder_name_with_glob_characters_is_not_a_pattern(tmp_path):
    odd = tmp_path / "work [2026]"
    c = config_mod.Config(projects_dir=odd, instruction_skip=["repo/CLAUDE.md"])

    assert all("[[]2026]" in g for g in c.instruction_skip)
    assert sum("[[]2026]" in g for g in c.instruction_globs) == 2


def test_the_global_section_goes_into_every_accounts_claude_md_unless_they_are_named(tmp_path):
    a, b = tmp_path / ".claude", tmp_path / ".claude-b"
    c = config_mod.Config(accounts=[a, b])
    assert c.global_files == [a / "CLAUDE.md", b / "CLAUDE.md"]

    # Mining one account must not stop the section reaching the other.
    c = config_mod.Config(accounts=[a], curated_accounts=[a, b])
    assert c.global_files == [a / "CLAUDE.md", b / "CLAUDE.md"]


def test_a_malformed_instruction_git_entry_names_the_setting(tmp_path):
    with pytest.raises(ValueError, match="instruction_git: each entry is a list"):
        config_mod.Config(projects_dir=tmp_path, instruction_git=["repo", "origin/main", "CLAUDE.md"])


@pytest.mark.parametrize("line, setting", [
    ('instruction_skip = "os/CLAUDE.md"', "instruction_skip"),  # would be read a character at a time
    ('accounts = "~/.claude"', "accounts"),
    ("docs_tags = [1, 2]", "docs_tags"),
    ("projects_dir = 5", "projects_dir"),
])
def test_a_setting_of_the_wrong_shape_is_an_error_that_names_it(tmp_path, line, setting):
    toml = tmp_path / "config.toml"
    toml.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{setting} is a"):
        config_mod.load(toml)


def test_a_config_path_given_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        config_mod.load(tmp_path / "cofig.toml")


def test_the_projects_folder_is_checked(tmp_path, monkeypatch):
    c = config_mod.Config(projects_dir=tmp_path)
    assert "it has no" in c.check_projects()

    (tmp_path / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
    assert c.check_projects() is None

    monkeypatch.setattr(config_mod, "HOME", tmp_path)  # a clone in the home folder
    assert "it is the home folder" in c.check_projects()


def test_a_projects_folder_whose_claude_md_is_in_git_is_refused(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "repo", {"CLAUDE.md": "# rules\n", "README.md": "x\n"})
    assert "is in git" in config_mod.Config(projects_dir=repo).check_projects()

    loose = git_repo(tmp_path / "loose", {"CLAUDE.md": "# rules\n", "README.md": "x\n"}, commit=["README.md"])
    assert config_mod.Config(projects_dir=loose).check_projects() is None  # in a repository, but not tracked

    # A GIT_DIR exported by a hook would have git answer for its repository instead.
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))
    assert config_mod.Config(projects_dir=loose).check_projects() is None


def test_the_home_folder_is_recognised_through_a_junction(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(home, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi  # a junction needs no privilege, a symbolic link does
        _winapi.CreateJunction(str(home), str(link))
    monkeypatch.setattr(config_mod, "HOME", home)
    assert "it is the home folder" in config_mod.Config(projects_dir=link).check_projects()


def test_the_curator_prompt_names_no_drive():
    prompt = (config_mod.ROOT / "curator" / "PROMPT.md").read_text(encoding="utf-8")

    assert not re.search(r"(?i)\b[A-Z]:[\\/]", prompt)


def test_the_default_projects_folder_is_this_checkouts_parent():
    assert config_mod.Config().projects_dir == config_mod.ROOT.parent


def test_a_config_that_cannot_load_is_logged_for_the_unattended_runs(tmp_path, monkeypatch):
    from akn import __main__ as cli

    bad = tmp_path / "config.toml"
    bad.write_text('instruction_git = ["repo", "origin/main", "CLAUDE.md"]\n', encoding="utf-8")
    monkeypatch.setattr(config_mod, "ROOT", tmp_path)  # where state/config-error.log goes
    with pytest.raises(ValueError):
        cli.main(["--config", str(bad), "nightly"])
    assert "`akn nightly`" in (tmp_path / "state" / "config-error.log").read_text(encoding="utf-8")

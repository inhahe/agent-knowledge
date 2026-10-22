"""Following a CLAUDE.md's `@path` imports the way Claude Code does, and the
coverage check reading the files instruction files import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from akn import config as config_mod
from akn import coverage, imports
from tests.conftest import git_repo


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_a_mention_is_a_path_after_whitespace_outside_code_and_comments():
    text = "\n".join([
        "@~/shared.md",
        "See also @./near.md and @rel/far.md#section but not mail@example.com.",
        "Not `@~/in-a-code-span.md`, nor <!-- @~/in-a-comment.md -->.",
        "  ```",
        "  @~/in-a-fence.md",
        "  ```",
        "An escaped space: @with\\ space.md",
        "Not paths: @ alone, @@twice, @#hash, @/ and @{upstream}.",
        "Named twice: @~/shared.md",
    ])
    assert imports.mentions(text) == ["~/shared.md", "./near.md", "rel/far.md", "with space.md"]


def test_the_end_of_a_code_span_or_a_comment_counts_as_a_start_as_in_claude_code():
    # Claude Code matches each run of plain text on its own, so a mention right after a
    # code span or a comment is at the start of its run.
    assert imports.mentions("x`y`@~/a.md and <!-- c -->@~/b.md") == ["~/a.md", "~/b.md"]


def test_a_code_span_showing_a_comment_opener_hides_nothing():
    text = "Write `<!--` to open a comment.\n@~/real.md\nand `-->` to close it."
    assert imports.mentions(text) == ["~/real.md"]


def test_fences_in_lists_and_quotes_hide_their_lines_and_a_one_line_span_opens_none():
    text = "\n".join([
        "```js```",
        "@~/after-a-span.md",
        "- a",
        "  - b",
        "",
        "    ```",
        "    @~/in-a-list-fence.md",
        "    ```",
        "> ```",
        "> @~/in-a-quoted-fence.md",
        "> ```",
        "~~~",
        "@~/in-a-tilde-fence.md",
        "~~~~",
        "@~/after.md",
    ])
    assert imports.mentions(text) == ["~/after-a-span.md", "~/after.md"]


def test_a_mention_resolves_from_home_the_naming_files_directory_or_as_it_stands(tmp_path):
    home, base = tmp_path / "home", tmp_path / "proj"
    assert imports.resolve("~/a.md", base, home) == home / "a.md"
    assert imports.resolve("./b.md", base, home) == base / "b.md"
    assert imports.resolve("c/d.md", base, home) == base / "c" / "d.md"
    assert imports.resolve("../e.md", base, home) == tmp_path / "e.md"
    assert imports.resolve(str(tmp_path / "abs.md"), base, home) == tmp_path / "abs.md"


@pytest.mark.skipif(sys.platform != "win32", reason="Claude Code reads /c/... as C:/... on Windows only")
def test_an_msys_style_drive_path_names_that_drive():
    assert imports.resolve("/c/Users/x.md", Path("E:/proj"), Path("E:/home")) == Path("C:/Users/x.md")


def test_imports_are_followed_depth_first_each_once_and_unreadable_ones_reported(tmp_path):
    home = tmp_path / "home"
    top = write(tmp_path / "CLAUDE.md", "@~/one.md\n@./missing.md\n@~/two.md\n")
    one = write(home / "one.md", "@./three.md\n@~/two.md\n")
    two = write(home / "two.md", "@~/one.md\n")
    three = write(home / "three.md", "leaf\n")
    got = imports.follow(top, top.read_text(encoding="utf-8"), home=home)
    assert [(f.importer, f.path) for f in got.files] == [(top, one), (one, three), (one, two)]
    assert got.files[1].text == "leaf\n"
    assert len(got.unreadable) == 1 and "@./missing.md" in got.unreadable[0]
    assert str(tmp_path / "missing.md") in got.unreadable[0]


def test_imports_are_followed_four_levels_down_and_no_further(tmp_path):
    files = [tmp_path / f"f{i}.md" for i in range(8)]
    for i, f in enumerate(files):
        write(f, f"@./f{i + 1}.md\n" if i + 1 < len(files) else "end\n")
    got = imports.follow(files[0], files[0].read_text(encoding="utf-8"), home=tmp_path)
    assert [f.path for f in got.files] == files[1:5]
    assert got.unreadable == []


def test_a_mention_no_file_could_have_is_reported_not_raised(tmp_path):
    top = write(tmp_path / "CLAUDE.md", "@bad\x00name.md\n")
    got = imports.follow(top, top.read_text(encoding="utf-8"), home=tmp_path)
    assert got.files == [] and len(got.unreadable) == 1


def test_expand_puts_each_imported_file_under_a_line_naming_it_and_its_importer(tmp_path):
    home = tmp_path / "home"
    top = write(tmp_path / "a" / "CLAUDE.md", "@~/b.md\n@~/d.md\n")
    b = write(home / "b.md", "@./c.md\n- Rule B.\n")
    c = write(home / "c.md", "- Rule C.\n")
    d = write(home / "d.md", "- Rule D.\n")
    text = imports.expand([(top, top.read_text(encoding="utf-8"))], home=home)
    labels = [ln for ln in text.split("\n") if ln.startswith("(")]
    assert labels == [f"(In {top})", f"(Imported by {top}: {b})", f"(Imported by {b}: {c})",
                      f"(Imported by {top}: {d})"]
    assert text.endswith("- Rule D.\n")


def test_expand_shows_each_distinct_text_once_and_each_imported_file_once(tmp_path):
    home = tmp_path / "home"
    shared = write(home / "shared.md", "- Shared rule.\n")
    a = write(tmp_path / "a" / "CLAUDE.md", "# graphify\n\n@~/shared.md\n")
    b = write(tmp_path / "b" / "CLAUDE.md", "@~/shared.md\n")
    c = write(tmp_path / "c" / "CLAUDE.md", "@~/shared.md\n")
    text = imports.expand([(p, p.read_text(encoding="utf-8")) for p in (a, b, c)], home=home)
    labels = [ln for ln in text.split("\n") if ln.startswith("(")]
    assert labels == [f"(In {a})", f"(Imported by {a}: {shared})", f"(In {b}, {c})"]
    assert text.count("- Shared rule.") == 1


def test_the_coverage_check_reads_each_imported_file_once(tmp_path):
    shared = write(tmp_path / "shared.md", "# Shared\n\n- Kill only your own PID.\n")
    a = write(tmp_path / "a" / "CLAUDE.md", "@../shared.md\n\n# A\n\n- Rule A.\n")
    b = write(tmp_path / "b" / "CLAUDE.md", "@../shared.md\n\n# B\n\n- Rule B.\n")
    cfg = config_mod.Config()
    cfg.instruction_globs = [str(tmp_path / "*" / "CLAUDE.md")]
    cfg.instruction_git = []
    cfg.instruction_skip = []
    sources = coverage.instruction_sources(cfg)
    assert [label for label, _ in sources] == [str(a), str(shared), str(b)]
    assert any("Kill only your own PID" in c.text for c in coverage.load_chunks(cfg))


def test_the_coverage_check_follows_the_imports_of_a_file_read_from_git(tmp_path):
    repo = git_repo(tmp_path / "repo", {"CLAUDE.md": "@./extra.md\n\n# Repo\n\n- Rule from git.\n",
                                        "extra.md": "- Rule beside it.\n"}, commit=["CLAUDE.md"])
    extra = repo / "extra.md"
    cfg = config_mod.Config()
    cfg.instruction_globs = []
    cfg.instruction_git = [[str(repo), "HEAD", "CLAUDE.md"]]
    cfg.instruction_skip = []
    labels = [label for label, _ in coverage.instruction_sources(cfg)]
    assert labels == [f"{repo}/CLAUDE.md @ HEAD", str(extra)]

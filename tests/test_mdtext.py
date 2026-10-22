"""Reports and curated rules are plain Markdown: no raw HTML, and text from models or
transcripts that looks like a tag is escaped (a viewer that rendered every HTML
fragment separately printed one "html error ... timed out" line per fragment)."""

from __future__ import annotations

import re

from akn import curate
from akn.mdtext import md_inline, md_quote

TAG = re.compile(r"(?<!\\)</?[A-Za-z][A-Za-z0-9_-]*[^<>\n]{0,40}>")


def raw_html(md: str) -> list[str]:
    """HTML-like tags outside fenced code and code spans, unescaped."""
    no_fences = re.sub(r"(?ms)^(`{3,}|~{3,}).*?^\1\s*$", "", md)
    return TAG.findall(re.sub(r"`[^`\n]*`", "", no_fences))


def test_helpers_escape_tags_and_fold_quotes():
    assert md_inline("run it with <crate> and <pid>") == "run it with \\<crate> and \\<pid>"
    assert raw_html(md_inline("a <details><summary>x</summary></details> b")) == []
    assert raw_html("a <details> b") == ["<details>"]  # the checker itself sees a raw tag
    assert md_quote("line one\nline <two>   three", 12) == "> line one lin …"
    assert md_quote("x <y>") == "> x \\<y>"


def test_curated_rules_may_hold_tags_only_inside_backticks():
    # Curated rules go into CLAUDE.md files that agents read raw, so they are not
    # escaped; instead a rule with tag-like text outside backticks is refused.
    sec = curate.Section("global", [], max_chars=1000, max_rules=5, note="")
    assert curate.check_rules("- **Kill by PID.** Use `taskkill /PID <pid>`, never an image name.", sec) == []
    assert curate.check_rules("- **Kill by PID.** Use taskkill /PID <pid>.", sec) != []
    assert raw_html("- **Kill by PID.** Use `taskkill /PID <pid>`.") == []

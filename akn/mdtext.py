"""Putting model-written and quoted text into Markdown safely.

The reports, the curator's log and the curated rules are read in whatever viewer
the operator opens them with, so they are plain Markdown: no raw HTML at all. Text that comes from a
model or a transcript can still look like HTML -- `<crate>`, `<pid>`,
`<tool_use_error>` -- and a viewer then treats it as a tag. One viewer rendered
every such fragment separately and printed "html error: Offscreen request timed
out after 20000ms" for each (2026-10-03). A backslash before `<` is CommonMark's
escape for it: displayed as `<`, never read as markup.
"""

from __future__ import annotations


def md_inline(text: object) -> str:
    """`text` as Markdown that displays exactly as written: `<` escaped."""
    return str(text).replace("<", "\\<")


def md_quote(text: object, limit: int = 500) -> str:
    """`text` as a one-line blockquote, whitespace folded, cut at `limit` characters."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[:limit] + " …"
    return "> " + md_inline(flat)

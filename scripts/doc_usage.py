"""Measure how agents use the big shared docs of a project (read-only).

Counts tool calls whose arguments name one of the documents, by tool and by
month, and how many characters those calls returned into the agent's context.
Used once to ground the "should the docs get a search index?" decision; kept
because the question will come back.

Usage: python scripts/doc_usage.py --project-glob "*-myproject*" known-issues design.md roadmap.md

The project glob is matched against Claude Code's project folder names (the
project's path with `\\`, `/` and `:` as `-`). Each document name matches where it
stands alone in a tool call's arguments, so `known-issues` does not match
`known-issues-resolved`, and does match `known-issues/1234.md`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akn import config  # noqa: E402
from akn.transcripts import CALL, RESULT, discover, parse_file  # noqa: E402

def doc_pattern(name: str) -> re.Pattern:
    """`name` where it stands alone: not continued by a letter, digit, `_` or `-`
    (so `known-issues` is not found in `known-issues-resolved`), and not preceded
    by one or by `.` (so `roadmap.md` is not found in `old-roadmap.md`)."""
    return re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-glob", required=True, help="Claude Code project folder names to read")
    ap.add_argument("docs", nargs="+", help="the document names to count, e.g. known-issues design.md")
    args = ap.parse_args()
    docs = {name: doc_pattern(name) for name in args.docs}
    cfg = config.load()
    calls: Counter = Counter()  # (doc, tool) -> n
    chars: Counter = Counter()  # (doc, tool) -> chars returned
    empty: Counter = Counter()  # (doc, tool) -> calls that found nothing
    month: dict[str, Counter] = defaultdict(Counter)
    big_reads = 0
    seen_uuids: set[str] = set()
    for path, acct, proj in discover(cfg.accounts, [args.project_glob], []):
        tr = parse_file(path, acct, proj)
        pending: dict[str, list[str]] = {}
        for e in tr.events:
            if e.uuid in seen_uuids:  # copied history: count once
                continue
            seen_uuids.add(e.uuid)
            if e.kind == CALL:
                hit = [d for d, rx in docs.items() if rx.search(e.detail or "")]
                if hit:
                    pending[e.tool_id] = hit
                    for d in hit:
                        calls[(d, e.tool)] += 1
                        month[e.ts[:7]][d] += 1
            elif e.kind == RESULT and e.tool_id in pending:
                for d in pending.pop(e.tool_id):
                    n = e.full_len or len(e.text)
                    chars[(d, e.tool)] += n
                    if n < 40 or (e.exit_code == 1 and n < 200):
                        empty[(d, e.tool)] += 1
                    if e.tool == "Read" and n > 20000:
                        big_reads += 1
    print(f"{'doc':24s} {'tool':12s} {'calls':>7s} {'found nothing':>14s} {'chars returned':>15s} {'avg':>8s}")
    for (d, t), n in sorted(calls.items(), key=lambda kv: -kv[1]):
        c = chars[(d, t)]
        print(f"{d:24s} {t:12s} {n:7d} {empty[(d, t)]:14d} {c:15,d} {c // max(n, 1):8,d}")
    total_calls = sum(calls.values())
    total_chars = sum(chars.values())
    print(f"\nTOTAL {total_calls} calls, {total_chars:,} chars (~{total_chars / 4e6:.1f} M tokens) returned; "
          f"Read calls returning >20k chars: {big_reads}")
    print("\nby month:")
    for m in sorted(month):
        print(f"  {m}  " + "  ".join(f"{d}={n}" for d, n in month[m].most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

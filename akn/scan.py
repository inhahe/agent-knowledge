"""Scan transcripts into windows (no model involved).

Each changed transcript is parsed, its signals found, and its windows stored.
Unchanged transcripts are skipped by (size, mtime), so a re-run costs only the
files that grew since the last one.
"""

from __future__ import annotations

import time
from collections import Counter

from .config import Config
from .signals import find_signals
from .store import Store
from .transcripts import discover, parse_file
from .windows import build_windows


def run(cfg: Config, store: Store, force: bool = False, verbose: bool = False) -> dict:
    t0 = time.time()
    files = list(discover(cfg.accounts, cfg.project_include, cfg.project_exclude))
    totals: Counter = Counter()
    signal_kinds: Counter = Counter()
    event_kinds: Counter = Counter()
    for n, (path, account, project) in enumerate(files, 1):
        try:
            st = path.stat()
        except OSError:
            continue
        if not force and store.file_unchanged(path, st.st_size, st.st_mtime):
            totals["files_unchanged"] += 1
            continue
        tr = parse_file(path, account, project, cfg.include_sidechains)
        sigs = find_signals(tr.events, mine_compact=cfg.mine_compact_summaries)
        wins = build_windows(tr, sigs, cfg.window_before_chars, cfg.window_after_chars,
                             cfg.window_max_chars, cfg.window_min_score)
        counts = store.replace_file_windows(path, wins)
        store.record_file(path, account, project, tr.session_id, tr.title, st.st_size, st.st_mtime,
                          len(tr.events), len(sigs), tr.first_ts, tr.last_ts)
        totals["files_scanned"] += 1
        totals["events"] += len(tr.events)
        totals["signals"] += len(sigs)
        totals["windows"] += len(wins)
        for k, v in counts.items():
            totals[f"windows_{k}"] += v
        signal_kinds.update(s.kind for s in sigs)
        event_kinds.update(e.kind for e in tr.events)
        if verbose or st.st_size > 50e6:
            print(f"  [{n}/{len(files)}] {account}/{project}/{path.name}: {st.st_size / 1e6:.0f} MB, "
                  f"{len(tr.events)} events, {len(sigs)} signals, {len(wins)} windows "
                  f"({counts['new']} new, {counts['changed']} changed, {counts['duplicate']} duplicate)", flush=True)
    totals["seconds"] = round(time.time() - t0, 1)
    return {"totals": dict(totals), "signal_kinds": dict(signal_kinds.most_common()),
            "event_kinds": dict(event_kinds.most_common())}

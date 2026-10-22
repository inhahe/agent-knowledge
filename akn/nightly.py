"""Unattended run for Task Scheduler: scan, extract what is new, write the report.

Measured on 2026-10-02: one machine's projects, six agents at work on one of them,
produced ~86 new excerpts a day (25-148), about 10-15 minutes of extraction, plus a minute of
scanning and a report that, with cached answers, only consults the model for
clusters that changed. So a nightly run is short; the first run over the whole
backlog (~3,900 excerpts) is the long one, and it must finish before nights start.

The run declines to start -- logging why and exiting 0, so Task Scheduler does
not record a failure for a deliberate skip -- when:

* another run holds the lock (`state/nightly.lock`, holding a live PID),
* something already listens on a model port (a manual `extract` or `server`),
* the GPU lacks room: the 27B model needs ~19.5 GB, and if other programs (a
  renderer, a game) hold more than the desktop's usual few GB, loading it would
  either fail or spill into shared memory and crawl for hours.

Everything it prints goes to `state/nightly-<date>.log` (pythonw has no console);
logs older than 14 days are deleted.
"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from .config import Config
from .store import Store

REQUIRED_FREE_VRAM_MB = 19500
LOG_KEEP_DAYS = 14


def pid_alive(pid: int) -> bool:
    """Whether a process exists, without touching it. (On Windows `os.kill(pid, 0)`
    would *terminate* it: os.kill maps every signal but Ctrl-C/Break to
    TerminateProcess.)"""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_vram_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Lock:
    def __init__(self, path: Path):
        self.path = path
        self.held = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return True
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid_alive(pid):
                    return False
                self.path.unlink(missing_ok=True)  # stale: its owner is gone
        return False

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


def _prune_logs(state: Path) -> None:
    cutoff = datetime.now() - timedelta(days=LOG_KEEP_DAYS)
    for p in state.glob("nightly-*.log"):
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                p.unlink()
        except OSError:
            pass


def run(cfg: Config, store: Store) -> int:
    log_path = cfg.state_dir / f"nightly-{datetime.now():%Y-%m-%d}.log"
    log = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the run
    sys.stdout = sys.stderr = log
    print(f"\n=== nightly run {datetime.now():%Y-%m-%d %H:%M:%S} (pid {os.getpid()}) ===")
    lock = Lock(cfg.state_dir / "nightly.lock")
    if not lock.acquire():
        print("another run holds the lock; skipping")
        return 0
    try:
        for port in (cfg.llm_port, cfg.embed_port):
            if port_in_use(port):
                print(f"port {port} is in use (a manual extract or server is running); skipping")
                return 0
        free = free_vram_mb()
        if free is not None and free < REQUIRED_FREE_VRAM_MB:
            print(f"only {free} MB of VRAM free, {REQUIRED_FREE_VRAM_MB} needed; skipping so as not to fight "
                  "whatever is using the GPU")
            return 0
        from . import extract, report, scan
        from .llm import Client, LlamaServer, llm_spec

        t0 = time.time()
        print("scan:", scan.run(cfg, store)["totals"])
        if store.pending_windows(limit=1):
            with LlamaServer(llm_spec(cfg), cfg.state_dir) as srv:
                print("extract:", extract.run(store, Client(srv.base_url), cfg.llm_model.name,
                                              workers=cfg.llm_parallel, thinking=cfg.llm_thinking))
        else:
            print("extract: nothing new")
        # Cards and embeddings for docs_repo's docsearch: incremental (only entries
        # whose text is new), so a night costs minutes after the first full run.
        # Not gated on that checkout having the tools: a checkout can lag its
        # trunk, and doccards finds them in any of its worktrees -- or reports
        # "skipped" when none has them yet.
        if cfg.docs_repo is None:
            print("doccards: off (no docs_repo)")
        else:
            from . import doccards
            print("doccards:", doccards.run(cfg, cfg.docs_repo, cards=True, vectors=True))
        print("report:", report.run(cfg, store))
        st = store.stats()
        print("store:", {k: st[k] for k in ("store_mb", "stored_text_mb", "lessons", "clusters",
                                             "cached_embeddings", "cached_answers")})
        print(f"=== done in {(time.time() - t0) / 60:.1f} min ===")
        return 0
    except Exception:  # noqa: BLE001 - an unattended run must log, not vanish
        traceback.print_exc()
        return 1
    finally:
        lock.release()
        _prune_logs(cfg.state_dir)

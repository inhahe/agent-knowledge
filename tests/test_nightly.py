from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

from akn import config as config_mod
from akn import nightly
from akn.store import Store


def test_pid_alive_for_self_and_for_a_finished_process():
    assert nightly.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=30)
    assert not nightly.pid_alive(p.pid)


def test_pid_alive_does_not_harm_the_process():
    # On Windows os.kill(pid, 0) would terminate the target; pid_alive must not.
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert nightly.pid_alive(p.pid)
        assert p.poll() is None
    finally:
        p.kill()
        p.wait(timeout=30)


def test_lock_excludes_and_takes_over_a_stale_one(tmp_path):
    a = nightly.Lock(tmp_path / "x.lock")
    assert a.acquire()
    b = nightly.Lock(tmp_path / "x.lock")
    assert not b.acquire()  # held by a live pid (ours)
    a.release()
    assert b.acquire()
    b.release()
    # A lock file naming a dead process is stale and is taken over.
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=30)
    (tmp_path / "x.lock").write_text(str(p.pid))
    c = nightly.Lock(tmp_path / "x.lock")
    assert c.acquire()
    c.release()


def test_port_in_use():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        assert nightly.port_in_use(srv.getsockname()[1])
    finally:
        srv.close()


def test_run_skips_when_a_model_port_is_busy(tmp_path, monkeypatch):
    cfg = config_mod.Config()
    cfg.state_dir = tmp_path
    st = Store(cfg.db_path)
    monkeypatch.setattr(nightly, "port_in_use", lambda port: port == cfg.llm_port)
    saved = sys.stdout, sys.stderr
    try:
        assert nightly.run(cfg, st) == 0
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = saved
    log = next(tmp_path.glob("nightly-*.log")).read_text(encoding="utf-8")
    assert "is in use" in log and not (tmp_path / "nightly.lock").exists()


def test_run_skips_when_vram_is_short(tmp_path, monkeypatch):
    cfg = config_mod.Config()
    cfg.state_dir = tmp_path
    st = Store(cfg.db_path)
    monkeypatch.setattr(nightly, "port_in_use", lambda port: False)
    monkeypatch.setattr(nightly, "free_vram_mb", lambda: 5000)
    saved = sys.stdout, sys.stderr
    try:
        assert nightly.run(cfg, st) == 0
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = saved
    assert "VRAM free" in next(tmp_path.glob("nightly-*.log")).read_text(encoding="utf-8")


@pytest.mark.parametrize("docs_repo", ["repo", None])
def test_run_does_every_step_in_order_and_doccards_does_not_need_the_tools_in_that_checkout(
        tmp_path, monkeypatch, docs_repo):
    # A checkout that lags its trunk may never hold scripts/docsearch.py; doccards
    # finds the tools in any worktree, so the nightly must not gate the step on
    # that one checkout. With no docs_repo, the step is left out.
    from akn import doccards, report, scan

    cfg = config_mod.Config()
    cfg.state_dir = tmp_path
    cfg.docs_repo = None
    if docs_repo:
        cfg.docs_repo = tmp_path / docs_repo  # exists, holds no scripts/
        cfg.docs_repo.mkdir()
    st = Store(cfg.db_path)
    steps: list[str] = []
    monkeypatch.setattr(nightly, "port_in_use", lambda port: False)
    monkeypatch.setattr(nightly, "free_vram_mb", lambda: 24000)
    monkeypatch.setattr(scan, "run", lambda cfg, store: steps.append("scan") or {"totals": {}})
    monkeypatch.setattr(st, "pending_windows", lambda limit=1: [])
    monkeypatch.setattr(doccards, "run", lambda cfg, repo, **kw: steps.append(f"doccards:{repo.name}") or {})
    monkeypatch.setattr(report, "run", lambda cfg, store: steps.append("report") or "r.md")
    saved = sys.stdout, sys.stderr
    try:
        assert nightly.run(cfg, st) == 0
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = saved
    assert steps == ["scan", *(["doccards:repo"] if docs_repo else []), "report"]
    log = next(tmp_path.glob("nightly-*.log")).read_text(encoding="utf-8")
    assert "extract: nothing new" in log and "=== done in" in log
    assert ("doccards: off (no docs_repo)" in log) == (docs_repo is None)

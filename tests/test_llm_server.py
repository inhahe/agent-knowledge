"""LlamaServer must never take over, or stop, a server it did not start."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from akn.llm import LlamaServer, ServerError, ServerSpec, _kill_on_close_job


class _Fake(BaseHTTPRequestHandler):
    model_id = "some-other-model.gguf"

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": self.model_id}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def _spec(port: int, model: str = "Qwen3.8-27B-UD-Q4_K_M.gguf") -> ServerSpec:
    return ServerSpec(name="llm", binary=Path("does-not-exist.exe"), model=Path(model), port=port, args=[])


def test_refuses_port_held_by_a_different_model(fake_server, tmp_path):
    srv = LlamaServer(_spec(fake_server.server_port), tmp_path)
    with pytest.raises(ServerError, match="already in use"):
        srv.start()
    assert not srv.owned


def test_reuses_matching_server_but_never_stops_it(fake_server, tmp_path):
    _Fake.model_id = "Qwen3.8-27B-UD-Q4_K_M.gguf"
    try:
        srv = LlamaServer(_spec(fake_server.server_port), tmp_path, reuse=True)
        srv.start()
        assert not srv.owned and srv.proc is None
        srv.stop()  # must be a no-op: the server is not ours
        assert srv._health() == 200
    finally:
        _Fake.model_id = "some-other-model.gguf"


def test_refuses_reuse_when_disabled(fake_server, tmp_path):
    _Fake.model_id = "Qwen3.8-27B-UD-Q4_K_M.gguf"
    try:
        with pytest.raises(ServerError):
            LlamaServer(_spec(fake_server.server_port), tmp_path, reuse=False).start()
    finally:
        _Fake.model_id = "some-other-model.gguf"


def test_missing_binary_is_a_clear_error(tmp_path):
    srv = LlamaServer(_spec(1), tmp_path)  # nothing listens on port 1
    with pytest.raises(ServerError, match="llama-server not found"):
        srv.start()


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_job_object_kills_child_when_handle_closes():
    import ctypes
    import subprocess
    from ctypes import wintypes

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job = _kill_on_close_job(child)
        assert job is not None
        close = ctypes.WinDLL("kernel32").CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(job)  # what process exit does to our handle
        child.wait(timeout=10)  # the job kills the child; this must not time out
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()

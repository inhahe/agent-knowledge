"""Run llama.cpp servers and talk to them.

`LlamaServer` starts one `llama-server.exe` and owns it: it is stopped by the PID
this object recorded, never by image name (other llama.cpp or Python processes on
the machine are not ours to touch). Unless `keep=True`, the server is also placed
in a Windows Job Object with KILL_ON_JOB_CLOSE, so if this tool crashes or is
killed, Windows tears the server down with it instead of leaving 17 GB of model
pinned in VRAM.

If something already answers on the port, `start()` refuses unless `reuse=True`
and the running server reports the expected model file; a reused server is not
ours, so `stop()` leaves it alone.

`Client` speaks the OpenAI-compatible endpoints llama-server exposes:
chat completions with a JSON schema (compiled server-side into a grammar, so the
reply always parses) and embeddings.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np


class ServerError(RuntimeError):
    pass


# --- Windows Job Object (kill the server if this process dies) -----------------

class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def _kill_on_close_job(proc: subprocess.Popen) -> object | None:
    """Put `proc` in a new job that dies when its last handle (ours) closes.

    Returns the job handle, which must be kept alive for as long as the server
    should live. Returns None off Windows or if the API refuses (the server then
    simply lacks the crash safety net; it is still stopped normally).
    """
    if sys.platform != "win32":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
    if ok and k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):  # noqa: SLF001
        return job
    return None


# --- server --------------------------------------------------------------------

@dataclass
class ServerSpec:
    name: str  # "llm" or "embed", used for the log file name
    binary: Path
    model: Path
    port: int
    args: list[str]


class LlamaServer:
    def __init__(self, spec: ServerSpec, log_dir: Path, keep: bool = False, reuse: bool = True):
        self.spec = spec
        self.log_path = log_dir / f"llama-{spec.name}.log"
        self.keep = keep
        self.reuse = reuse
        self.proc: subprocess.Popen | None = None
        self.job: object | None = None
        self.owned = False
        self.base_url = f"http://127.0.0.1:{spec.port}"

    def _health(self) -> int | None:
        try:
            return httpx.get(self.base_url + "/health", timeout=3).status_code
        except httpx.HTTPError:
            return None

    def _served_model(self) -> str:
        try:
            data = httpx.get(self.base_url + "/v1/models", timeout=5).json()
            return " ".join(str(m.get("id", "")) for m in data.get("data", []))
        except (httpx.HTTPError, ValueError):
            return ""

    def start(self, timeout: float = 900) -> "LlamaServer":
        if self._health() is not None:
            served = self._served_model()
            if self.reuse and self.spec.model.name in served:
                print(f"[{self.spec.name}] reusing server already on :{self.spec.port} ({served})", flush=True)
                return self
            raise ServerError(
                f"port {self.spec.port} is already in use by a server serving {served or 'something else'}; "
                "stop it yourself or choose another port (this tool only stops servers it started)"
            )
        if not self.spec.binary.exists():
            raise ServerError(f"llama-server not found at {self.spec.binary}")
        if not self.spec.model.exists():
            raise ServerError(f"model not found at {self.spec.model}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.spec.binary), "-m", str(self.spec.model), "--host", "127.0.0.1",
               "--port", str(self.spec.port), *self.spec.args]
        log = open(self.log_path, "wb")  # noqa: SIM115 - handed to the child for its lifetime
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     creationflags=flags, cwd=str(self.spec.binary.parent))
        self.owned = True
        if not self.keep:
            self.job = _kill_on_close_job(self.proc)
        print(f"[{self.spec.name}] started pid {self.proc.pid}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        while True:
            code = self.proc.poll()
            if code is not None:
                raise ServerError(f"llama-server exited with {code} while loading; log tail:\n{self.log_tail()}")
            if self._health() == 200:
                print(f"[{self.spec.name}] ready after {time.time() - t0:.0f}s", flush=True)
                return self
            if time.time() - t0 > timeout:
                self.stop()
                raise ServerError(f"llama-server not ready after {timeout:.0f}s; log tail:\n{self.log_tail()}")
            time.sleep(2)

    def log_tail(self, n: int = 25) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        """Stop the server if, and only if, this object started it."""
        if not self.owned or self.proc is None:
            return
        if self.keep:
            print(f"[{self.spec.name}] leaving pid {self.proc.pid} running (--keep-server)", flush=True)
            return
        if self.proc.poll() is None:
            self.proc.terminate()  # TerminateProcess on this PID only
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        print(f"[{self.spec.name}] stopped pid {self.proc.pid}", flush=True)
        self.proc = None
        self.owned = False

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def llm_spec(cfg) -> ServerSpec:
    return ServerSpec(
        name="llm", binary=cfg.llama_server, model=cfg.llm_model, port=cfg.llm_port,
        args=["-ngl", "999", "-c", str(cfg.llm_parallel * cfg.llm_ctx_per_slot), "-np", str(cfg.llm_parallel),
              "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--no-webui", "--metrics"],
    )


def embed_spec(cfg, cpu: bool = False) -> ServerSpec:
    """`cpu=True` keeps the 0.6B embedding model off the GPU, for when the 27B model
    already holds nearly all of the VRAM (a nightly run, or a backlog extraction)."""
    return ServerSpec(
        name="embed", binary=cfg.llama_server, model=cfg.embed_model, port=cfg.embed_port,
        args=["-ngl", "0" if cpu else "999", "--embedding", "--pooling", "last", "-c", "16384", "-np", "4",
              "-b", "4096", "-ub", "4096", "--no-webui"],
    )


# --- client --------------------------------------------------------------------

@dataclass
class ChatResult:
    data: dict
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    raw: str


class Client:
    def __init__(self, base_url: str, timeout: float = 1200, transport: httpx.BaseTransport | None = None):
        # `transport` exists for tests (httpx.MockTransport); production uses the default.
        self.http = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10), transport=transport)

    def chat_json(self, system: str, user: str, schema: dict, *, max_tokens: int = 1800,
                  temperature: float = 0.3, thinking: bool = False) -> ChatResult:
        """One chat completion whose reply is constrained to `schema`."""
        payload = {
            "model": "local",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        t0 = time.time()
        resp = self.http.post("/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            raise ServerError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        usage = body.get("usage") or {}
        if choice.get("finish_reason") == "length":
            raise ValueError(f"reply truncated at max_tokens={max_tokens}")
        try:
            data = json.loads(content)
        except ValueError as exc:
            raise ValueError(f"reply is not JSON ({exc}): {content[:300]!r}") from exc
        return ChatResult(data, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                          time.time() - t0, content)

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        """L2-normalised embeddings, one row per text."""
        rows: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = self.http.post("/v1/embeddings", json={"model": "local", "input": chunk})
            if resp.status_code != 200:
                raise ServerError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            rows.extend(np.asarray(d["embedding"], dtype=np.float32) for d in data)
        mat = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def qwen_query(instruction: str, text: str) -> str:
    """Qwen3-Embedding's query format: an instruction line, then the query."""
    return f"Instruct: {instruction}\nQuery: {text}"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

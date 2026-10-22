"""Download one file from a Hugging Face repo with parallel, resumable range requests.

Why this exists rather than `hf download` or plain curl: on a link whose DNS
occasionally fails to resolve the HF CDN host, a single-stream curl dies on the
first hiccup, and `hf download` (Xet) stalled silently at 16 MB when it happened
on 2026-10-02. This downloader:

* splits the file into fixed-size chunks fetched by several connections at once,
* retries every chunk independently with backoff (DNS failures included),
* re-resolves the signed CDN URL when it expires (HTTP 403/410),
* records finished chunks in a sidecar state file, so a killed run resumes where
  it stopped instead of starting over,
* verifies the final size and the SHA-256 that the Hugging Face API publishes
  before renaming the `.part` file into place.

Usage:
    python scripts/fetch_model.py unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf

The destination defaults to `<ai_dir>/<publisher>/<repo>/`, the layout the
models in the AI folder use (`ai_dir`, from akn/config.py and config.toml); pass
a directory to override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from akn import config  # noqa: E402 - needs the project root on the path first

API = "https://huggingface.co/api/models/{repo}/tree/{rev}"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def file_info(repo: str, path: str, rev: str) -> tuple[int, str | None]:
    """Return (size, sha256-or-None) for `path` in `repo`, from the HF tree API."""
    folder = path.rsplit("/", 1)[0] if "/" in path else ""
    url = API.format(repo=repo, rev=rev) + (f"/{folder}" if folder else "")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    for entry in resp.json():
        if entry.get("path") == path:
            lfs = entry.get("lfs") or {}
            return int(entry["size"]), lfs.get("oid")
    raise SystemExit(f"{path} not found in {repo}@{rev}")


def resolve_cdn_url(url: str) -> str:
    """Follow the HF redirect once and return the signed CDN URL."""
    resp = requests.head(url, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    return resp.url


class Downloader:
    def __init__(self, repo: str, path: str, dest_dir: Path, rev: str, connections: int, chunk_mb: int):
        self.repo, self.path, self.rev = repo, path, rev
        self.url = RESOLVE.format(repo=repo, rev=rev, path=path)
        self.final = dest_dir / Path(path).name
        self.part = self.final.with_name(self.final.name + ".part")
        self.state_path = self.final.with_name(self.final.name + ".part.json")
        self.connections = connections
        self.chunk = chunk_mb * 1024 * 1024
        self.size, self.sha256 = file_info(repo, path, rev)
        self.n_chunks = (self.size + self.chunk - 1) // self.chunk
        self.lock = threading.Lock()
        self.cdn_url = resolve_cdn_url(self.url)
        self.done: set[int] = set()
        self.bytes_this_run = 0
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists() and self.part.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            # A state file for a different size/chunking is stale; ignore it rather
            # than trusting chunk indices that no longer mean the same bytes.
            if state.get("size") == self.size and state.get("chunk") == self.chunk:
                self.done = set(state.get("done", []))

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"size": self.size, "chunk": self.chunk, "done": sorted(self.done)}),
            encoding="utf-8",
        )
        os.replace(tmp, self.state_path)

    def _fetch_chunk(self, idx: int) -> int:
        start = idx * self.chunk
        end = min(start + self.chunk, self.size) - 1
        want = end - start + 1
        delay = 2.0
        for attempt in range(1, 61):
            try:
                headers = {"Range": f"bytes={start}-{end}"}
                with requests.get(self.cdn_url, headers=headers, stream=True, timeout=(30, 120)) as resp:
                    if resp.status_code in (401, 403, 410):
                        # Signed URL expired: resolve a fresh one and retry at once.
                        with self.lock:
                            self.cdn_url = resolve_cdn_url(self.url)
                        continue
                    if resp.status_code != 206:
                        raise IOError(f"HTTP {resp.status_code} for chunk {idx}")
                    buf = bytearray()
                    for piece in resp.iter_content(1024 * 1024):
                        buf.extend(piece)
                    if len(buf) != want:
                        raise IOError(f"chunk {idx}: got {len(buf)} bytes, wanted {want}")
                with open(self.part, "r+b") as fh:
                    fh.seek(start)
                    fh.write(buf)
                with self.lock:
                    self.done.add(idx)
                    self.bytes_this_run += want
                    self._save_state()
                return want
            except (requests.RequestException, IOError) as exc:
                print(f"  chunk {idx} attempt {attempt} failed: {exc!s:.160}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 1.6, 60.0)
        raise RuntimeError(f"chunk {idx} failed after 60 attempts")

    def run(self) -> None:
        if self.final.exists() and self.final.stat().st_size == self.size:
            print(f"{self.final} already complete")
            return
        if not self.part.exists():
            # Create it empty rather than truncate() it to full size: on Windows the
            # C runtime extends a file by writing zeros, which cost ~5 minutes for
            # 16 GB on 2026-10-02 before a single byte was downloaded. Chunks are
            # handed out in order, so each write lands near the current end and the
            # gap the filesystem must zero-fill stays small.
            self.part.touch()
        todo = [i for i in range(self.n_chunks) if i not in self.done]
        print(
            f"{self.path}: {self.size / 1e9:.2f} GB, {self.n_chunks} chunks, "
            f"{len(self.done)} already done, {self.connections} connections",
            flush=True,
        )
        t0 = time.time()
        last = t0
        with ThreadPoolExecutor(max_workers=self.connections) as pool:
            futures = [pool.submit(self._fetch_chunk, i) for i in todo]
            for fut in as_completed(futures):
                fut.result()  # re-raise a chunk that exhausted its retries
                now = time.time()
                if now - last >= 15:
                    last = now
                    rate = self.bytes_this_run / (now - t0) / 1e6
                    left = (self.size - len(self.done) * self.chunk) / 1e6
                    eta = left / rate / 60 if rate > 0 else float("inf")
                    print(
                        f"  {len(self.done)}/{self.n_chunks} chunks, {rate:.1f} MB/s, ~{eta:.0f} min left",
                        flush=True,
                    )
        self._verify_and_finish(time.time() - t0)

    def _verify_and_finish(self, elapsed: float) -> None:
        if len(self.done) != self.n_chunks:
            raise SystemExit("not all chunks finished; re-run to resume")
        if self.part.stat().st_size != self.size:
            raise SystemExit(f"size mismatch: {self.part.stat().st_size} != {self.size}")
        if self.sha256:
            print("verifying sha256 ...", flush=True)
            h = hashlib.sha256()
            with open(self.part, "rb") as fh:
                while block := fh.read(16 * 1024 * 1024):
                    h.update(block)
            if h.hexdigest() != self.sha256:
                # Do not delete: the state file lets a re-run refetch selectively
                # once the bad chunk is found; deleting would discard 16 GB.
                raise SystemExit(f"SHA-256 mismatch: {h.hexdigest()} != {self.sha256}")
        os.replace(self.part, self.final)
        self.state_path.unlink(missing_ok=True)
        print(f"done: {self.final} ({self.bytes_this_run / 1e6 / max(elapsed, 1):.1f} MB/s this run, sha256 ok)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="publisher/repo, e.g. unsloth/Qwen3.8-27B-GGUF")
    ap.add_argument("path", help="file path inside the repo")
    ap.add_argument("dest_dir", type=Path, nargs="?", default=None,
                    help="default: <ai_dir>/<publisher>/<repo>")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--connections", type=int, default=6)
    ap.add_argument("--chunk-mb", type=int, default=32)
    args = ap.parse_args()
    dest = args.dest_dir or config.load().ai_dir.joinpath(*args.repo.split("/"))
    dest.mkdir(parents=True, exist_ok=True)
    Downloader(args.repo, args.path, dest, args.rev, args.connections, args.chunk_mb).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare generation speed of one extraction with and without the JSON-schema grammar.

Needs a running llm server (python -m akn server llm). Usage:
    python scripts/bench_extract.py [window_id]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akn import config  # noqa: E402
from akn.extract import SCHEMA, SYSTEM, user_prompt  # noqa: E402
from akn.store import Store  # noqa: E402

cfg = config.load()
store = Store(cfg.db_path)
wid = sys.argv[1] if len(sys.argv) > 1 else None
row = (store.db.execute("SELECT * FROM windows WHERE id=?", (wid,)).fetchone() if wid else
       store.db.execute("SELECT * FROM windows WHERE status='done' ORDER BY n_chars LIMIT 1 OFFSET 5").fetchone())
base = f"http://127.0.0.1:{cfg.llm_port}"


def run(label: str, response_format: dict | None) -> None:
    payload = {
        "model": "local",
        "messages": [{"role": "system", "content": SYSTEM + ("" if response_format else
                      "\n\nReply with JSON only, exactly this shape: {\"lessons\": [{\"evidence\": str, \"title\": str, "
                      "\"lesson\": str, \"trigger\": str, \"kind\": str, \"scope\": str, \"source\": str, "
                      "\"confidence\": number}]}")},
                     {"role": "user", "content": user_prompt(row["text"])}],
        "temperature": 0.3, "top_p": 0.8, "top_k": 20, "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False},
        "timings_per_token": False,
    }
    if response_format:
        payload["response_format"] = response_format
    t0 = time.time()
    r = httpx.post(base + "/v1/chat/completions", json=payload, timeout=600).json()
    dt = time.time() - t0
    t = r.get("timings", {})
    content = r["choices"][0]["message"]["content"]
    try:
        n = len(json.loads(content).get("lessons", []))
        ok = "valid JSON"
    except ValueError:
        n, ok = -1, "INVALID JSON"
    print(f"{label:22s} {dt:6.1f}s  prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second', 0):.0f}/s  "
          f"gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):.1f}/s  {ok}, {n} lessons")


print(f"window {row['id']} ({row['n_chars']} chars)")
run("json_schema grammar", {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": SCHEMA}})
run("json_object (loose)", {"type": "json_object"})
run("no grammar", None)

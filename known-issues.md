# Known issues

Unsolved bugs and technical debt in agent-knowledge. Fix them when found if possible;
this file is for what could not be fixed at the time.

## The nightly has failed every night since 2026-10-18 (found 2026-10-22)

The `agent-knowledge nightly` scheduled task ends with result 1 every night, so no
lessons report has been written since `lessons-2026-10-17-0525.json`. The curator task
went on ending with 0, logging only `skipped: lessons-2026-10-17-0525.json was already
curated` each morning, so nothing told the operator. It now says so (`stale_report`: a
warning in its log and at the top of `curated --log`, and exit code 1).

Two failures, in `state/nightly-2026-10-*.log`:

- **10-18:** `llama-server not ready after 900s`. The server's log shows
  `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`, so it
  was loading the 27B model on the CPU: the scheduled tasks had started running on a
  machine without an NVIDIA GPU.
- **10-19 on:** `model not found at <ai_dir>/unsloth/Qwen3.8-27B-GGUF/...`: the AI folder
  still held llama.cpp and the embedding model, but not the LLM.

**Fix (the operator's):** put the model back (`python scripts/fetch_model.py
unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf`), or point `llm_model` at its new
place in `config.toml`, and run the nightly on a machine with the GPU.

**Fix (here), still open:** the nightly's GPU check (`free_vram_mb`, through
`nvidia-smi`) carries on when `nvidia-smi` fails, so on a machine with no NVIDIA driver
llama-server falls back to the CPU and times out. It should refuse to start there, saying
why. It was not changed blind: `nvidia-smi` can also fail on a machine that has the GPU
(not on the scheduled task's PATH, say), so check what it does under Task Scheduler on
the GPU machine before making its failure a skip.

## A curator test failed once, unexplained (2026-10-03) -- probably fixed

`tests/test_curate.py::test_decide_records_and_removes_from_the_pending_list` failed
once, in a full run of the suite under `python -W error::SyntaxWarning -m pytest -q`,
in the same run as an unrelated failure (a wrong expected value in a new test, since
fixed). Its output was not captured: only the summary line was read. It then passed
in 17 runs: 5 full suites under the same flag, and 12 runs of `test_curate.py` with
`test_mdtext.py`. No code it exercises changed between the failing and the passing
runs. The test was removed later that day, with the proposals mode it tested.

**Probable cause, found the same day:** the rewritten curator's tests hit
`PermissionError: [WinError 5] Access is denied` replacing a file the test had just
written. Windows refuses a rename over a file while another process has it open
without delete sharing, and an antivirus scan of a newly written file does that for
a moment. The removed test saved its state the same way (`tmp.replace(state.json)`).
Every write in `akn/curate.py` now goes through `_atomic_write`, which retries for up
to ~18 s and then writes in place (`test_a_replace_refused_for_a_moment_is_retried...`).

**If a curator test fails once again without explanation:** capture the full output
(`pytest -q tests/test_curate.py -x -vv`). If it is not this, suspect the settings
the `cfg` fixture changes (`accounts`, `projects_dir`, `instruction_git`), or the
date (`_today()`) changing mid-test.

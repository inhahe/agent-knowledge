# agent-knowledge

Mines the Claude Code transcripts on this machine for lessons — operator
corrections, tool traps, mistakes agents made and fixed — using a local LLM, and
writes a report of candidate rules. Each morning a curator (headless Opus) turns
the best of them into rules, within a size budget. It writes them into one section
that it owns, at the end of the global and workspace CLAUDE.md files, and touches
nothing else in any instruction file.

## Set it up

It is written for Windows: Task Scheduler runs it, and llama.cpp's
`llama-server.exe` runs the models.

1. Put this folder in the folder that holds your projects (`projects_dir`), which
   needs a `CLAUDE.md` of its own: the curator writes its *workspace* section there.
   Its *global* section goes at the end of each account's `~/.claude*/CLAUDE.md`.
2. Get llama.cpp (a CUDA build) and the two models into the AI folder (see
   Requirements below), and install the Python packages (`pip install -r requirements.txt`).
3. Copy `config.example.toml` to `config.toml` and set what differs on your
   machine: at least `ai_dir` if your models are elsewhere, and `curator_config_dir`
   if the curator should run on another account than `~/.claude`.
4. Run `python -m akn all` once by hand. The first run reads your whole history.
5. Register the two scheduled tasks: `scripts/register-nightly.ps1` and
   `scripts/register-curator.ps1`.
6. Tell your agents about [CURATED-RULES.md](CURATED-RULES.md) in your own rules, so
   that "drop the curated rule about X" works.

## Run it

From this folder, which sits in the projects folder (`projects_dir`):

```bash
python -m akn scan        # parse transcripts into windows (no model, ~3-5 min, incremental)
python -m akn extract     # local LLM reads pending windows (hours on the first run; minutes after)
python -m akn report      # cluster, consolidate, compare with CLAUDE.md files -> reports/lessons-*.md
python -m akn all         # all three
```

Useful extras:

```bash
python -m akn stats                          # what the store holds
python -m akn extract --limit 50             # try a few windows first (highest-scoring first)
python -m akn sample -n 3 --kind interrupt   # read what the signal heuristics selected
python -m akn server llm                     # start the model for interactive use: chat page at http://127.0.0.1:8091/
python -m akn doccards                       # cards + embeddings for docs_repo's scripts/docsearch.py (new entries only)
python -m pytest -q                          # tests
```

`doccards` serves one kind of repository: one with its own `scripts/docsearch.py`
that searches its shared documents, named by `docs_repo` (unset, nothing runs). It
writes into that repository's shared search cache
(`<git-common-dir>/docsearch-cache.sqlite`): an embedding per document entry, then a
one-line summary card per issue, decision, question, request and spec passage. The first
run took 4.3 hours on 2026-10-02 (16,834 embeddings in ~15 minutes, then 5,146 cards); after
that only new or edited entries are done. Cards are shown by docsearch as summaries and are never written into a document.

`extract` and `report` start the llama.cpp server they need and stop it when done
(`--keep-server` leaves it running).

## Nightly

Task Scheduler runs `python -m akn nightly` at 04:30 (`scripts/register-nightly.ps1`
registered it; `Unregister-ScheduledTask -TaskName 'agent-knowledge nightly'` removes
it). It scans, extracts what is new, refreshes docsearch's cards and embeddings (when
`docs_repo` is set), and writes the report. It skips the night, logging why to `state/nightly-<date>.log`, if
the GPU is busy or a model server is already running. The newest report is always
`reports/latest.md`; lessons first seen since the previous report are marked
**[new since last report]**.

**What a run re-reads.** Transcripts that did not change since the last run (same size
and mtime) are not opened. One that grew is re-parsed whole, which is cheap (the full
11 GB parse takes ~4 minutes), but only *excerpts* whose text is new or changed go to
the model, which is the expensive part: a nightly run sends ~86 excerpts (~10-15 min of
GPU), against ~3,900 for the first, whole-history run. The report caches every model
answer under a hash of its inputs, so only clusters that gained lessons are re-consolidated.

## Curated rules: from report to CLAUDE.md, automatically

You don't need to read the report. At 06:00 a second task ("agent-knowledge curator",
`scripts/register-curator.ps1`) runs `python -m akn curate --run`:

1. It waits for the nightly run to finish.
2. It takes all of the newest report's lessons that no instruction file carries yet, or
   that keep recurring although one does.
3. A headless Opus session sees them all at once and rewrites the one section it owns at
   the end of the global CLAUDE.md (every account's) and of the workspace CLAUDE.md.
   It adds the most valuable rules, sharpens ones agents still miss, and merges or retires
   rules to stay within budget: at most 15 rules / 4,000 characters in the global
   section, and 20 / 5,000 in the workspace section.
   - It runs on the account `curator_config_dir` names, falling back to the other
     accounts if that one is at its usage limit.
   - It may read the projects, to check facts, and write only its draft.
4. Code checks the result before writing it: within budget, at most 8 changes per
   section, and nothing above the section touched.

**Nothing to approve.** Your own rules are never changed. When agents keep missing one of
yours, the log suggests a sharper wording for you to consider.

**Your edits always win.**
- A section you add at the end of the file stays put, and the curated section moves below it.
- An edit you make inside the curated section is copied into every copy of that file, and
  never overwritten. A rule you wrote or reworded is then kept word for word, and one you
  deleted stays deleted.

To see or change what it did, tell any agent:
- "what did the curator change?"
- "drop the curated rule about X": it is removed, and never comes back;
- "undo the last curation".

[CURATED-RULES.md](CURATED-RULES.md) tells the agent how; point your agents to it from
your own rules (mine do it from `~/claude-global.md`, which every account's CLAUDE.md
imports). The history, with diffs, is a git repository in `curated/`.

**A stalled nightly is said out loud.** When the newest lessons report is more than
`stale_report_days` (2) days old by the date in its name, or there is none, the curator
writes a warning in its log, exits with code 1 (Task Scheduler shows the run as failed),
and puts the warning at the top of `curated --log`, so "what did the curator change?"
tells you. Otherwise a nightly that stopped would only look like a run of quiet
mornings. The run also exits 1 when it refuses to run (the projects folder looks wrong),
when a section cannot be written (an account `curated_accounts` names has no CLAUDE.md,
say), or when a draft fails its checks. A `config.toml` that cannot be loaded, or holds a
setting of the wrong shape, is recorded in `state/config-error.log`, since the scheduled
runs have no console.

**The first mornings differ from the rest.** The first reports hold the whole backlog,
which the curator works through strongest first, 120 lessons a morning, for about a week.
The sections fill over the first 2–3 mornings, at most 8 changes per section each. After
that, each morning brings only lessons that are new or still recurring, and usually
changes little or nothing. Once a section is full, a new rule must displace an old one,
and the curator sees how much evidence lay behind each existing rule, and whether its
lesson has kept coming up since. A lesson it has decided on comes back only if it keeps
recurring.

A curation run took 6 minutes on the first real report (117 candidates). It costs about
0.1% of a week's Opus use.

## What it keeps, and how big it gets

| file | what | size |
|---|---|---|
| `state/akn.sqlite` | the excerpts (zlib-compressed; copied history keeps none), lessons, clusters, caches | 44 MB after the whole backlog (147 MB before compression); under 1 MB a night after that |
| `reports/` | `latest.md`, plus the newest 30 dated `lessons-*.md`/`.json` | older ones are deleted by each report |
| `state/nightly-*.log` | one log per night | deleted after 14 days |
| `state/llama-*.log` | the model servers' output | overwritten by each run |
| `<docs_repo>/.git/docsearch-cache.sqlite` | docsearch's cards and vectors | only texts some worktree still holds; the rest is pruned each run |

Cached embeddings and model answers unused for 14 days are deleted at the end of a report
(they are recomputed if ever needed). Reports cluster only the new lessons against clusters
kept from earlier runs, so a report's memory does not grow with every lesson ever extracted;
`python -m akn report --recluster` rebuilds the clusters from scratch. Settings:
`reports_keep`, `cache_max_age_days`, `cluster_batch` in `akn/config.py`.

## Requirements

- Windows, and an NVIDIA GPU with about 20 GB of VRAM free (the nightly run checks, and
  skips the night rather than fight another program for it)
- [Claude Code](https://claude.com/claude-code) for the curator, which runs `claude -p`
  on your account
- the AI folder (`ai_dir`: `D:/ai` where there is one, else `C:/ai`), holding the
  llama.cpp CUDA build at `<ai_dir>/llama.cpp/b11344/` and the models in the
  layout `<ai_dir>/<publisher>/<repo>/<file>`, after the Hugging Face repo each comes from:
  `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` and
  `Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`
  (`python scripts/fetch_model.py <publisher>/<repo> <file>` downloads one into that layout)
- Python 3.12+ with `httpx`, `numpy`, `scikit-learn`, `requests`, and `pytest` for tests
  (`pip install -r requirements.txt`)

Paths and settings live in `akn/config.py`; override any of them in `config.toml`.
See `design.md` for how it works and why.

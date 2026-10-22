# agent-knowledge — design

Local-LLM tooling for one machine's Claude Code sessions. Two tools share the
models and the server plumbing:

- the **lesson miner** (`scan` / `extract` / `report`), over the Claude Code
  session transcripts;
- **doc cards** (`doccards`), which fills the shared cache behind one repository's
  own `scripts/docsearch.py` (`docs_repo`) with a summary card and an embedding per
  document entry.

## Purpose

Agents on this machine keep rediscovering the same things: a shell that eats
backslashes, a build that hangs, an operator preference stated three times. The
operator has been finding these by hand (the "Text that tools mangle" section of
the global CLAUDE.md came from searching every past session for "mangled"). The
miner automates the *finding*. Since 2026-10-03 the curator also automates the
*adopting*, within limits.

**Rules are plain text in the instruction files agents already read, never a
separate memory.** A memory layer that writes its own rules (Hindsight and
similar) was rejected for this machine. Its beliefs are written by an LLM, read by
every agent and reviewed by no one; git cannot diff or revert them, and they ignore
branches. They would be a second source of truth competing with the
version-controlled docs.

The curator answers all of those objections but review. Dropping review was the
operator's choice: approving each proposal was itself too much work.
- Its rules sit in one marked section of the CLAUDE.md files, within a budget.
- Every change is in a git history with its reason, and any run can be undone.
- It never touches the operator's text, or a CLAUDE.md that lives in a
  repository's version control (see "From report to rule" below).

## Pipeline

```
transcripts (~/.claude*/projects/*/*.jsonl, all accounts, all projects)
  │  scan      transcripts.py  parse JSONL → ordered events (prompt, agent text, tool call/result, …)
  │            signals.py      cheap heuristics mark moments worth reading
  │            windows.py      excerpts around signals, rendered compactly, merged/split, scored
  │            store.py        SQLite: windows + status; incremental by file (size, mtime)
  ▼
  │  extract   extract.py      local LLM (llama.cpp) → 0..4 lessons per window, JSON-schema constrained
  ▼
  │  report    cluster.py      embed new lessons (Qwen3-Embedding) and place them among the clusters
  │                            kept from earlier runs (average linkage, incremental); rank by
  │                            distinct sessions × operator boost × recency × breadth
  │            report.py       LLM consolidates each top cluster into one canonical lesson
  │            coverage.py     nearest CLAUDE.md chunks by embedding; LLM judges covered / partly / not
  ▼
reports/lessons-<stamp>.md (for review) + .json (everything)
```

### Signals (what gets read)

Reading every token would take days and mostly waste them (90%+ of transcript text
is tool output). Signals and weights (`signals.py`):

| signal | weight | note |
|---|---|---|
| operator prompt | 1.0 | +1.5 if it corrects/forbids/states a rule |
| operator interrupt | 2.5 | |
| agent admits a mistake / explains a surprise | 1.5–2.5 | "turns out", "root cause", "mangled", … |
| weak surprise words (≥2) | 0.5 | |
| tool error | 0.35 | grep/diff exit 1 is not an error |
| 3 errors within 8 tool results | 1.2 | once per streak |
| timeout | 1.0 | |
| harness refusal / API error | 0.8 | |
| message from another session | 0.6 | |

A window is extracted when its signals sum to ≥ 1.0.

**Machine prompts are not operator prompts.** In the first scan, 6,476 of 9,183
"human" prompts were loop wakeups, the loop sentinel (sometimes HTML-escaped) or
launcher templates — and their wording ("never", "always") tripped the correction
detector. `transcripts.py` classifies them as `loop` by known prefixes, and any
prompt ≥ 20 chars that recurs ≥ 3 times in one session as a template.

**Compaction summaries are not mined** by default (`mine_compact_summaries`): long
autonomous sessions compact every few hundred events (7,306 summaries, 96 M chars
in the first scan), each restating the last, and the raw events are mined anyway.

**Copied history is mined once.** Moving the projects to another drive copied sessions wholesale,
so identical events (same `uuid`) appear in two files. A window whose every
signal event is already claimed by a window in another file is `duplicate`.

### Models and servers

- LLM: Qwen3.8-27B (dense, Aug 2026), unsloth `UD-Q4_K_M` GGUF, 16.5 GB, on the
  RTX 4090 via llama.cpp `llama-server` (CUDA 13 build). Chosen over Gemma 4
  26B-A4B and Qwen3.6-35B-A3B on benchmark strength; the 35B MoE does not fit in
  the ~20 GB the desktop leaves free. Thinking mode off for extraction.
- Embeddings: Qwen3-Embedding-0.6B Q8_0, last-token pooling, instruction-prefixed
  queries.
- Only one server runs at a time (VRAM). Servers are started and stopped by this
  tool **by PID**, never by image name, and sit in a Job Object with
  KILL_ON_JOB_CLOSE so a crash cannot leave 17 GB pinned in VRAM.
- Models live in the AI folder (`ai_dir`: `D:\ai`, or `C:\ai` on a machine
  without a D: drive), where the operator keeps all models, as
  `<ai_dir>\<publisher>\<repo>\<file>` (the Hugging Face repo path); llama.cpp
  sits beside them in `<ai_dir>\llama.cpp\<build>\`. Every other path is
  likewise derived from `projects_dir` (the folder this checkout sits in) or
  `ai_dir` (`Config._derive`), so a moved folder is one setting; and the paths
  `config.toml` gives into the projects folder may be relative to it, so they
  follow it too. Before it runs, the curator checks the projects folder holds
  the workspace CLAUDE.md and is neither the home folder nor a folder whose
  CLAUDE.md git tracks (`Config.check_projects`): a checkout cloned into the
  wrong place must not have it write where every home-folder session reads, or
  into a repository's history.
- Nothing about one machine's projects is in the code. What the operator's
  machine needs beyond the defaults — the account the curator runs on, the
  accounts its global section goes to, a repository whose CLAUDE.md is read
  from git because its checkout lags, the repository doc cards serve and what
  the card writer is told about it, what the curator should know about the
  folder — is in the operator's own `config.toml`. A file that cannot be loaded
  is recorded in `state/config-error.log`, since the scheduled runs have no
  console.

### Incremental clustering

The first design clustered every lesson on every run. Average-linkage clustering
needs an n × n distance matrix, and lessons only accumulate (~8,700 after the
backlog, ~200 a night), so within months a report would have needed tens of GB.
Clusters are now kept in the store as **(member count, vector sum)**, and a run
touches only what changed:

- a lesson that disappeared (its excerpt changed) leaves its cluster, whose sum is
  recomputed from the members left;
- new lessons, in batches of at most `cluster_batch` (10,000), join the cluster
  whose members they are closest to on average, or are clustered among themselves;
- then any two clusters, one of them changed, whose average-linkage distance is
  below the threshold are merged, closest pair first.

This is exact average linkage wherever it can be computed from sums: for unit
vectors, the mean cosine similarity between two groups is `S_a·S_b / (n_a n_b)`.
Only the order of merges differs from clustering everything at once. Measured on
1,200 real lessons, batches of 300 agreed with all-at-once clustering at an
adjusted Rand index of 0.93, with the same number of multi-member clusters; one
batch agrees exactly (1.000). Without the merge step the index was 0.86: clusters
that grew towards each other stayed split, undercounting how often a lesson
recurs, which is what the ranking measures.

### From report to rule: the curator

A report is a list of candidates, and somebody has to turn the good ones into
rules. The operator's decisions:
- 2026-10-02: Opus drafts numbered proposals and the operator approves each one.
- 2026-10-03, replacing that because approving was itself too much work: **Claude
  decides, within a size budget, with no approval step.**

What keeps an unreviewed process safe is its shape:

- **It owns one section and nothing else.** The section is
  "Lessons agents learned (maintained automatically)", last, in each account's
  global CLAUDE.md (`Config.global_files`: identical copies of the section, one
  per account in `curated_accounts`, or in `accounts` when that is unset; the
  rest of each file is its own account's) and in the workspace CLAUDE.md. The
  two lists are apart because `accounts` says whose transcripts are mined: a
  list that limits the mining must not stop the section reaching the others,
  and a copy of a config folder kept beside the real ones must not receive it.
  A project's own CLAUDE.md, in its repository, would need that
  project's way of publishing a change, so the curator never writes one: a rule
  for one project goes under a `### <project>` subheading in the workspace section.
- **Nothing of the operator's is ever destroyed.**
  - Everything outside the section is kept byte for byte, line endings included
    (one account's file may be CRLF and another's LF).
  - The section ends at the next `#` or `##` heading, which its rules may not
    contain. So a section appended to the file later, by the operator or by a
    tool's installer, stays the operator's, and the curated section moves below it.
  - Before every write, `sync` compares each copy with the history:
    - A copy holding text the curator once wrote is **stale**: a write that did
      not reach it, or a restored backup. Each text is stored in the history's
      object store before any copy is written, so git can answer whether the
      curator wrote it. A stale copy, or one with no section yet (a new account),
      is brought up to date.
    - Anything else is a **hand edit**, and it is adopted into every copy. A rule
      written or reworded by hand is kept word for word from then on: the input
      says so, and finalize refuses a draft that changes it. A rule deleted by
      hand counts as dropped.
    - Copies edited differently are left alone until the operator settles it, and
      so is a section edited while the curator runs.
  - Every copy is checked before any is written, so a section is written
    everywhere or nowhere.
  - Each file is replaced through a temporary file, so it is never left
    half-written. The replace is retried while Windows refuses it: an antivirus
    scan of a file just written can do that for a moment, and the tests hit it.
- **A budget, checked by code.** At most 15 rules and 4,000 characters (global),
  and 20 and 5,000 (workspace), since every session pays for them in tokens. There
  are at most 8 changes per section per run, only bullets and `### ` subheadings,
  and tag-like text only in backticks. A section that fails a check keeps
  yesterday's rules, and the run logs why.
- **It sees everything at once.** All candidates (up to 150), the current
  sections, the operator's own rules, the CLAUDE.md files read from git
  (`instruction_git`) and the operator's notes on the projects folder
  (`curator_notes`) go into one session.
  So it can choose the best within the budget, and merge or retire rules to make
  room, rather than accumulate. The operator's rules include every file their
  CLAUDE.md imports, read the way Claude Code reads them (`akn/imports.py`): the
  rules all of the operator's accounts share can live in one file (theirs is
  `~/claude-global.md`) which each account's CLAUDE.md imports. When an import cannot be read (the shared
  file moved or deleted), that section is not written until it can be, and the
  run logs why: Claude Code skips such an import in silence, so its rules are
  gone from every session, and the curator can no longer see what it must not
  repeat.
- **It checks facts.** It may read anything in the projects folder
  (`--add-dir`, `projects_dir`, which `input.md` names to it), and must confirm a rule's claim about a repository before writing
  it. Writing is pinned to its draft by `Edit(state/curator/draft.json)`: tested,
  a write anywhere else is refused. It has no shell, no web and no MCP servers,
  and keeps no transcript, which the miner would otherwise mine.
- **The operator stays in charge.**
  - `curated --drop TEXT` removes a rule everywhere and records it as never to
    come back.
  - `--undo` takes back the newest curator run that changed anything, rule by
    rule:
    - what the run added or reworded goes, and what it removed returns to its
      place;
    - whatever changed since (a drop, a hand edit) is kept;
    - the rules it takes out count as dropped, so they are not simply added back
      the next morning.
  - `--log` reads `curated/CHANGES.md`, where every change has its reason.
  - A stalled nightly is said, not left to look like quiet mornings: when the
    newest report is older than `stale_report_days` (by the date in its name,
    which a copy of `reports/` keeps and its mtime does not), or there is none,
    the run logs a warning and exits 1, and `--log` starts with it
    (`stale_report`). It also exits 1 when it refuses to run, when a section is
    blocked (an account `curated_accounts` names has no CLAUDE.md:
    `missing_named_copies`), or when a draft fails its checks: whatever needs
    the operator is a failed run in Task Scheduler, not a quiet success.
  - Every git call runs without the caller's `GIT_*` variables
    (`gitread.clean_env`): a `GIT_DIR` exported by a hook outranks `git -C`, and
    would have the history committed into, or questions answered by, some
    other repository.
  - The history is a git repository in `curated/`, with commits by
    "agent-knowledge curator".
  - The operator's own rules are never edited: when agents keep missing one,
    the log suggests a sharper wording instead.

Steps (`akn/curate.py`):
1. **prepare** (no model) writes `state/curator/input.md` and the context copies.
2. **The Opus session** writes `state/curator/draft.json`: each section's complete
   new rules, the changes with reasons, skips, and suggestions about the operator's
   rules.
3. **finalize** (no model) checks the draft, writes the sections into every file,
   and commits the history.

It runs on the account `curator_config_dir` names first, falling back to the others
when a run fails without writing its draft (that account's weekly limit, during the
first live test).
Candidates it does not reach are not recorded and come back.

**Backlog, then increments.** Every report ranks the whole history, and the
ranking barely moves from one night to the next. On 2026-10-03 a cluster needed
about 5 sessions to make the top 120 that a report consolidates, and none of the
120 was under a week old. Meanwhile about 10 clusters a day were newly seen in a
second session. Ranking alone would show the curator the same decided lessons
every morning, and a new one only once it had outranked them all. So:
- **The report consolidates the best 120 clusters the curator has not decided
  on** (`curate.settled`, `report.select_open`). A decided cluster returns only
  when it recurs:
  - a rule's lesson after 3 more sessions, since recurring despite the rule means
    the rule is not working;
  - a skipped lesson once it has also grown by half (`RECUR_FACTOR`), since big
    lessons recur at their own steady rate and that alone is no news.
- **The backlog drains in order of strength.** On 2026-10-03 it held 803
  eligible clusters (seen in 2 or more sessions, or stated by the operator). At
  120 a morning it takes about a week, and after that a morning sees only what is
  new or still recurring.
- **The first mornings fill the sections over 2–3 runs**, because of the limit of
  8 changes per section per run (the preview reached 8 in both sections).
  Leftovers are not recorded, so they come back.
- **Once a section is full, every addition displaces a rule.** Judging that needs
  evidence on both sides, so the input shows where each current rule came from:
  when it was written, the sessions behind it, and how many sessions came after
  it (`rule_evidence`, recorded from each change's `rule` and `replaces`). A rule
  whose lesson stopped recurring is probably working, and the prompt says that is
  no reason to retire it.

**First real run** (a preview against copies of the files, 2026-10-03, 117
candidates, 6 minutes):
- 8 global rules (2.9k characters) and 8 workspace rules (3.2k, 4 of them under one project's subheading);
- 64 candidates skipped, mostly the backslash family the operator's own rules
  already cover;
- 2 of the operator's rules flagged as still missed, with suggested wording.

One rule stated a repository fact ("the root `.cargo/config.toml` targets bare
metal"). It was right, and it is why fact-checking access was added before the
first unattended run.

### Size and growth

Measured 2026-10-02, when the store held the whole backlog: 146 MB, of which 128
MB was excerpt text. Now:

- excerpt text is zlib-compressed (3.2× on real excerpts), and duplicate windows
  (copied history, never extracted) keep no text: the store fell to 43 MB, every
  excerpt verified byte-identical by its SHA-1 after the migration;
- cached embeddings and model answers carry a last-used time and are deleted
  after `cache_max_age_days` (14) unused;
- `reports/` keeps the newest `reports_keep` (30) dated reports, and a report's
  JSON lists the 500 best unconsolidated clusters, not all of them;
- docsearch's shared cache is pruned of texts no worktree holds, but only when
  every worktree was read.

What still grows is what should: new excerpts and lessons, under 1 MB a night.

## Doc cards (for a repository's own docsearch)

The repository `docs_repo` names keeps its issues, decisions, questions, requests
and roadmap as one file per entry, and `scripts/docsearch.py` in that repository
searches them: by words
(SQLite FTS5) everywhere, and by meaning when a shared cache holds embeddings.
`akn doccards` fills that cache; the search itself needs nothing from here except
the embedding model file, which it runs on the CPU for the query.

- **What it writes**, per distinct entry text: an embedding (Qwen3-Embedding, of
  the heading and the start of the body) and, for issues, decisions, questions,
  requests and spec passages, a *card* -- a one-sentence summary, the status the
  text states, up to six tags, and two or three questions the entry answers. The
  questions are searched as an extra field, so an entry is found by how someone
  would ask for it, not only by its own wording.
- **Where**: `<git-common-dir>/docsearch-cache.sqlite` of that repository, so
  every worktree shares it. Keyed by the SHA-1 of the entry's text: a card is made
  once per text however many worktrees hold it, an edited entry gets a new one,
  and the two document layouts produce identical text (checked on 2026-10-02:
  every hash difference between an old-layout worktree and the new layout was a
  real edit), so worktrees before and after the cutover share one cache.
- **Whose definitions**: the repository's own `scripts/doc_entries.py` and
  `scripts/docsearch.py` are imported from whichever checkout has them
  (`tools_checkout`) -- what an entry is, and the cache schema, are the
  repository's, never a copy kept here.
- **Order: vectors, then cards.** Vectors take minutes on the GPU. Cards take
  hours on a first run: on 2026-10-02 the 15 worktrees held 16,688 distinct
  entries, 5,118 of them card-worthy, averaging ~4,000 characters of input; at
  the extraction run's measured token rates the estimate was 7-8 hours; the
  first full run (2026-10-02) made all 5,146 cards in about 4 hours (~21 a
  minute), the embeddings in ~15. After that, a night's new and edited
  entries take minutes. docsearch ranks by meaning only once ~90% of the entries being searched
  have a vector (below that, the embedded few crowd out better keyword matches),
  so vectors first makes semantic search available hours sooner, and a run cut
  short still leaves it working.
- **A card is a summary, labelled as one** where docsearch shows it, and is never
  written into a document. The documents stay the only source of truth.

`nightly` runs it after extraction, so new and edited entries get their card and
vector the same night.

## Files

| path | role |
|---|---|
| `akn/config.py` | defaults; `config.toml` overrides (unknown keys are an error; `config.example.toml` lists them) |
| `requirements.txt` | the Python packages |
| `akn/transcripts.py` | JSONL → events; format notes in the module doc |
| `akn/signals.py`, `akn/windows.py` | what to read and how to render it |
| `akn/store.py` | SQLite state (`state/akn.sqlite`) |
| `akn/llm.py` | llama-server lifecycle + OpenAI-compatible client |
| `akn/extract.py` | extraction prompt and schema |
| `akn/cluster.py`, `akn/coverage.py`, `akn/report.py` | grouping, coverage, report |
| `akn/imports.py` | a CLAUDE.md's `@path` imports, followed the way Claude Code follows them |
| `akn/gitread.py` | a file as a git revision holds it, with a timeout (the runs are unattended) |
| `akn/doccards.py` | cards and vectors for `docs_repo`'s docsearch (shared cache) |
| `akn/nightly.py` | the unattended run: lock, GPU/port checks, scan → extract → doccards → report |
| `akn/curate.py`, `curator/PROMPT.md` | the morning curator: candidates, the Opus draft, the curated sections |
| `CURATED-RULES.md` | what an agent does when the operator asks to drop, undo or see curated rules |
| `curated/` (its own git repository, ignored here) | the curated sections' history: `global.md`, `workspace.md`, `CHANGES.md` |
| `state/curator/` | the morning run's working files: `input.md`, `context/`, `draft.json`, `state.json` |
| `akn/mdtext.py` | plain-Markdown helpers: tag-like text escaped, so viewers never see HTML |
| `scripts/register-curator.ps1` | registers the 06:00 curator task |
| `scripts/register-nightly.ps1` | registers the 04:30 Task Scheduler entry |
| `scripts/fetch_model.py` | resumable parallel HF downloader, into `ai_dir` (for links whose DNS drops connections) |
| `scripts/bench_extract.py` | extraction throughput with and without the JSON grammar, thinking on/off |
| `scripts/doc_usage.py` | measures how agents use a project's big docs |
| `tests/` | pytest; synthetic transcripts in the real format |

## Known limits

- Signal weights are judgement, tuned by reading samples (`python -m akn sample`).
- The coverage check sees only instruction files (CLAUDE.md files and the files
  they import, memory files), not
  project docs such as `known-issues.md`; a lesson recorded only there shows as
  "not covered".
- Imports are followed from every instruction file as Claude Code follows them from
  an account's own CLAUDE.md. From a project's CLAUDE.md, Claude Code loads an
  import that leads outside the session's directory only once the operator approves
  it (per project, per account), and it skips files it does not take for text; the
  coverage check models neither, so it may count a rule no session loads. A file
  read from git (`instruction_git`) has its imports read from the
  checkout, not from that revision. No project CLAUDE.md imports anything yet.
- Clustering threshold (0.30 cosine distance) was chosen by inspection.
- The curator's decisions are keyed by cluster id. When two clusters merge, the
  decision on the one that disappears is lost, and the merged cluster is looked at
  afresh. After `report --recluster`, every id is new, and the curator works through
  the whole backlog again. Lessons its own rules cover then show as covered, so
  this costs runs, not wrong rules.

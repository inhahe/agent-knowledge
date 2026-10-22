"""Paths and tunables: defaults, and an optional override file.

The defaults assume Claude Code's own layout and little else: the accounts are
the `~/.claude*` folders that hold transcripts, the projects are in the folder
this tool's own checkout sits in (`projects_dir`), and llama.cpp and the models
are in the AI folder (`ai_dir`). The paths below those two derive from them, so
a moved folder is one setting.

A `config.toml` at the project root overrides any field (config.example.toml
lists them). In it, a path may start with `~`, the home folder; and the paths
that name things in the projects folder -- `docs_repo`, the repositories of
`instruction_git`, the patterns of `instruction_globs` and `instruction_skip` --
may be relative to `projects_dir`, so that they follow it when it moves. Unknown
keys in the file are an error (a typo should fail loudly, not silently fall back
to a default).
"""

from __future__ import annotations

import glob
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import gitread

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()


def _default_projects() -> Path:
    """The folder the projects live in: the one this checkout sits in, which
    holds the workspace CLAUDE.md."""
    return ROOT.parent


def _default_ai() -> Path:
    """Where the models are: D:/ai where there is a D: drive holding it, else
    C:/ai (a one-drive machine)."""
    return next((p for p in (Path("D:/ai"), Path("C:/ai")) if p.is_dir()), Path("C:/ai"))


def _default_accounts() -> list[Path]:
    """Every `~/.claude*` directory that holds a `projects/` transcript tree."""
    return sorted(p for p in HOME.glob(".claude*") if (p / "projects").is_dir())


# Every account's CLAUDE.md, the workspace one and every project's, and each
# account's memory files: relative patterns are in the projects folder.
DEFAULT_INSTRUCTION_GLOBS = ["~/.claude*/CLAUDE.md", "CLAUDE.md", "*/CLAUDE.md",
                             "~/.claude*/projects/*/memory/*.md"]


@dataclass
class Config:
    # --- where transcripts come from -------------------------------------------
    accounts: list[Path] = field(default_factory=_default_accounts)
    # Glob patterns matched against project directory names such as
    # "C--projects-app". Empty include list means "all".
    project_include: list[str] = field(default_factory=list)
    project_exclude: list[str] = field(default_factory=list)
    # Subagent ("sidechain") turns are skipped by default: they repeat work the
    # parent session also reports, and would double-count recurrences.
    include_sidechains: bool = False

    # --- where this tool keeps its own files --------------------------------
    state_dir: Path = ROOT / "state"
    reports_dir: Path = ROOT / "reports"
    # The curator owns one section at the end of every account's CLAUDE.md and
    # of the workspace CLAUDE.md (akn/curate.py). Its history -- a git
    # repository of each section's text, and CHANGES.md -- lives here.
    curated_dir: Path = ROOT / "curated"
    # The accounts (config folders) whose CLAUDE.md the global section is written
    # into; None: every one in `accounts`. Kept apart from `accounts`, which says
    # whose transcripts are mined: a list that limits the mining must not stop
    # the section reaching the others, and a copy of a config folder kept beside
    # the real ones must not receive it.
    curated_accounts: list[Path] | None = None
    # Size budgets: every session pays for these in tokens (~4 characters each).
    curated_global_chars: int = 4000
    curated_global_rules: int = 15
    curated_workspace_chars: int = 5000
    curated_workspace_rules: int = 20
    # At most this many lessons go to the curator in one morning (best first); it
    # sees them all at once, so it can choose the best within the budgets...
    curator_max_candidates: int = 150
    # ...and a lesson already curated or skipped comes back only after recurring in
    # this many more sessions.
    curator_recur_sessions: int = 3
    # The curator is a headless Claude Code session on this account (its config
    # folder), with this model; the other accounts are its fallbacks.
    curator_config_dir: Path = HOME / ".claude"
    curator_model: str = "opus"
    # What the curator should know about the projects folder that it cannot see
    # for itself: which checkout is out of date, which project's rules go under
    # their own subheading. Free text, given to it as written.
    curator_notes: str = ""
    # A lessons report older than this many days means the nightly run has
    # stopped writing them: the curator says so in its log, its exit code and
    # `curated --log`, since otherwise it would only keep finding nothing new.
    stale_report_days: float = 2.0

    # --- where things are ------------------------------------------------------
    # The folder the projects live in, and the AI folder: everything below is
    # derived from these two unless set itself (`_derive`).
    projects_dir: Path = field(default_factory=_default_projects)
    ai_dir: Path = field(default_factory=_default_ai)

    # --- local model servers (llama.cpp) --------------------------------------
    # The AI folder holds the models laid out as <ai_dir>/<publisher>/<repo>/<file>,
    # after the Hugging Face repo each came from.
    llama_dir: Path | None = None    # <ai_dir>/llama.cpp/b11344
    llm_model: Path | None = None    # <ai_dir>/unsloth/Qwen3.8-27B-GGUF/...
    embed_model: Path | None = None  # <ai_dir>/Qwen/Qwen3-Embedding-0.6B-GGUF/...
    llm_port: int = 8091
    embed_port: int = 8092
    # Parallel decoding slots and per-slot context. Qwen3.8-27B keeps a KV cache
    # for only 16 of its 64 layers (the rest are linear-attention), so context
    # is cheap; the binding limit is the ~20 GB of VRAM a 24 GB card leaves free.
    llm_parallel: int = 2
    llm_ctx_per_slot: int = 24576
    # Disable the model's "thinking" phase for extraction: the task is
    # classification and summarisation of text in front of it, and thinking
    # multiplies output tokens several-fold.
    llm_thinking: bool = False

    # --- a repository whose shared documents get summary cards and embeddings --
    # (its own scripts/docsearch.py reads them from the repository's shared
    # cache; see akn/doccards.py). None: there is none, and nightly skips it.
    docs_repo: Path | None = None
    # What the model writing the cards is told about that project and its
    # documents, and the subject tags it is shown as examples.
    docs_project: str = ""
    docs_tags: list[str] = field(default_factory=list)

    # --- windowing -------------------------------------------------------------
    # Characters of context kept before and after each signal, and the cap on a
    # merged window. ~3.5 characters per token for this mix of prose and code.
    window_before_chars: int = 5000
    window_after_chars: int = 9000
    window_max_chars: int = 36000
    # A window is extracted only if its signals sum to at least this weight.
    window_min_score: float = 1.0
    # Mine context-compaction summaries as windows of their own. Off by default:
    # they recur constantly in long sessions and restate events mined directly.
    mine_compact_summaries: bool = False

    # --- how much this tool keeps ------------------------------------------------
    # Dated reports kept in reports_dir (reports/latest.md is always the newest).
    # One run a night, so 30 is about a month of history.
    reports_keep: int = 30
    # Cached embeddings and model answers not used for this many days are
    # deleted at the end of a report: the answer for a cluster that has since
    # changed can never be asked for again, and an embedding that is needed
    # after all is recomputed in milliseconds.
    cache_max_age_days: int = 14
    # New lessons are clustered in batches of at most this many, so clustering
    # memory is bounded by the batch (an n x n distance matrix: ~0.4 GB at
    # 10,000), not by the number of lessons ever extracted. A first report and
    # each night's new lessons fit in one batch, which clusters exactly as
    # clustering everything at once would; measured on 1,200 real lessons,
    # batches of 300 still agreed with it at an adjusted Rand index of 0.93.
    cluster_batch: int = 10000

    # --- coverage check: the files that already hold rules ----------------------
    # Glob patterns; by default DEFAULT_INSTRUCTION_GLOBS.
    instruction_globs: list[str] | None = None
    # Files read from git rather than the working tree, as (repo, rev, path): a
    # checkout that lags its trunk holds an old copy, so the rules are read from
    # the trunk. The curator is also given these files, for reference.
    instruction_git: list[list[str]] | None = None
    # Working-tree files to skip (glob patterns), usually because a git copy
    # above supersedes them: the stale checkout and its worktrees hold older or
    # identical copies, and near-duplicate copies would crowd the nearest-rule
    # lookup.
    instruction_skip: list[str] | None = None

    def __post_init__(self) -> None:
        self._derive()

    def _derive(self) -> None:
        """Fill in each path left unset from `projects_dir` and `ai_dir`, and
        resolve the project paths given relative to `projects_dir` or from `~`.
        Paths go into glob patterns escaped (a folder named `x [2]` is not a
        pattern) and with forward slashes, as the reports print them."""
        ai = self.ai_dir
        if self.llama_dir is None:
            self.llama_dir = ai / "llama.cpp" / "b11344"
        if self.llm_model is None:
            self.llm_model = ai / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q4_K_M.gguf"
        if self.embed_model is None:
            self.embed_model = ai / "Qwen" / "Qwen3-Embedding-0.6B-GGUF" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
        if self.docs_repo is not None:
            self.docs_repo = self.in_projects(self.docs_repo)
        for entry in self.instruction_git or []:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 3
                    and all(isinstance(x, str) for x in entry)):
                raise ValueError(f"instruction_git: each entry is a list [repo, rev, path], not {entry!r}")
        self.instruction_globs = [self._pattern(g) for g in
                                  (DEFAULT_INSTRUCTION_GLOBS if self.instruction_globs is None
                                   else self.instruction_globs)]
        self.instruction_git = [[self.in_projects(Path(repo)).as_posix(), rev, path]
                                for repo, rev, path in (self.instruction_git or [])]
        self.instruction_skip = [self._pattern(s) for s in (self.instruction_skip or [])]

    def in_projects(self, path: Path) -> Path:
        """`path`, from `~` or relative to the projects folder; an absolute one as it is."""
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.projects_dir / path

    def _pattern(self, pattern: str) -> str:
        """A glob pattern, from `~` or relative to the projects folder, whichever
        it starts from; the folder's own name escaped."""
        if pattern == "~" or pattern.startswith(("~/", "~\\")):
            return glob.escape(HOME.as_posix()) + "/" + pattern[2:]
        if Path(pattern).is_absolute():
            return pattern
        return glob.escape(self.projects_dir.as_posix()) + "/" + pattern

    @property
    def global_files(self) -> list[Path]:
        """The files the global section is written into: the CLAUDE.md of every
        account in `curated_accounts`, or in `accounts` when that is unset."""
        accounts = self.accounts if self.curated_accounts is None else self.curated_accounts
        return [Path(a) / "CLAUDE.md" for a in accounts]

    @property
    def workspace_file(self) -> Path:
        """The workspace CLAUDE.md, which every session under `projects_dir` reads."""
        return self.projects_dir / "CLAUDE.md"

    def check_projects(self) -> str | None:
        """Why `projects_dir` is not the projects folder, or None if it is. It
        must hold the workspace CLAUDE.md, and be neither the home folder nor a
        folder whose CLAUDE.md is in git. A checkout placed elsewhere (a clone in
        the home folder, one at the root of another repository) would point the
        curator, and the folder it may read, at the wrong place, and have it
        write into a CLAUDE.md that the home folder's every session loads, or
        that a repository's history holds."""
        here = self.projects_dir
        problem = None
        if not self.workspace_file.is_file():
            problem = f"it has no {self.workspace_file}"
        elif _same_folder(here, HOME):
            problem = "it is the home folder"
        elif gitread.tracked(self.workspace_file):
            problem = f"its {self.workspace_file.name} is in git, so it belongs to a repository"
        if problem:
            return (f"{here} is not the projects folder: {problem}. "
                    "Set projects_dir in config.toml (or create the workspace CLAUDE.md).")
        return None

    @property
    def db_path(self) -> Path:
        return self.state_dir / "akn.sqlite"

    @property
    def llama_server(self) -> Path:
        return self.llama_dir / "llama-server.exe"


def _same_folder(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are one folder, however each is reached: a junction, a
    symbolic link, a `subst` drive or a short 8.3 name all lead to the same file id."""
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


_PATH_FIELDS = {"state_dir", "reports_dir", "curated_dir", "curator_config_dir", "llama_dir",
                "llm_model", "embed_model", "docs_repo", "projects_dir", "ai_dir"}
_PATH_LIST_FIELDS = {"accounts", "curated_accounts"}
# Lists of strings: a bare string here would be read one character at a time.
_STRING_LIST_FIELDS = _PATH_LIST_FIELDS | {"project_include", "project_exclude", "instruction_globs",
                                           "instruction_skip", "docs_tags"}


def load(path: Path | None = None) -> Config:
    """Build a Config from defaults plus `config.toml`, if there is one, or the file
    `path`, which must exist. The file's settings are given to the constructor, so
    that the paths derived from `projects_dir` and `ai_dir` follow an override of
    either. A setting of the wrong shape is an error that names it."""
    if path is None:
        path = ROOT / "config.toml"
        if not path.exists():
            return Config()
    elif not path.is_file():
        raise FileNotFoundError(f"no config file at {path}")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s): {', '.join(unknown)}")
    given = {}
    for key, value in data.items():
        if key in _STRING_LIST_FIELDS and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"{path}: {key} is a list of strings, not {value!r}")
        if key in _PATH_FIELDS and not isinstance(value, str):
            raise ValueError(f"{path}: {key} is a path, as a string, not {value!r}")
        if key in _PATH_FIELDS:
            value = Path(value).expanduser()
        elif key in _PATH_LIST_FIELDS:
            value = [Path(v).expanduser() for v in value]
        given[key] = value
    return Config(**given)

"""Keep a machine-owned section of the instruction files current, from the newest lessons report.

The operator chose (2026-10-03) to let Claude decide which mined lessons become
rules, within a size budget, rather than approve proposals one by one. So the
curator owns exactly one section at the end of each of these files, and nothing
else in them:

* the global CLAUDE.md of every account (`cfg.global_files`: the section in one
  copy per account, kept identical; the rest of each file is its own account's), and
* the workspace CLAUDE.md (`cfg.workspace_file`, in the projects folder), which
  every session working there reads.

Each morning (`curate --run`, Task Scheduler 06:00):

1. `prepare` (no model): bring the files and the history into agreement (`sync`);
   from the newest report, take the lessons no instruction file carries, or carries
   yet still recur; drop those already decided on (unless they have recurred in
   `curator_recur_sessions` more sessions since). Write them to
   state/curator/input.md, with each section's current rules, read-only copies of
   the operator's own rules (with every file their file imports), and the rules the
   operator has dropped.
2. A headless Opus session (curator/PROMPT.md) sees all of that at once and writes
   state/curator/draft.json: each section's complete new rules, within its budget,
   and what changed and why. It may read the projects, and write only the draft.
3. `finalize` (no model): check each section against its budget and shape; write the
   accepted ones into every copy of their files; commit them to the history
   repository in curated/; add the day's changes to curated/CHANGES.md.

Nothing here may destroy the operator's text. Everything outside the section is
kept byte for byte -- including a section the operator (or a tool's installer)
appends to the file later, which the curated section then moves below. A hand edit
inside the section is adopted rather than overwritten (`sync`). And a section is
written into all of its copies or, when any of them cannot take it, into none.

Undo, drop and inspect: `python -m akn curated --log | --undo | --drop TEXT`
(CURATED-RULES.md tells any agent how; the operator's own rules should point
there).
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from . import gitread, imports
from .config import HOME, ROOT, Config

RECURRING_DAYS = 14
MAX_CHANGES_PER_RUN = 8
# A skipped lesson is looked at again once its sessions have grown by half (and by
# at least `curator_recur_sessions`); see reopen_at.
RECUR_FACTOR = 1.5

SECTION_HEADING = "# Lessons agents learned (maintained automatically)"
# The intro is recognised by its first words alone, so rewording the rest of it
# here does not make every copy look edited by hand.
INTRO_PREFIX = "Kept by the morning curator"
SECTION_INTRO = (
    INTRO_PREFIX + f" (`{ROOT}`) from lessons that recurred "
    "across separate sessions. It is rewritten every morning, so do not edit it by hand: to remove a rule for "
    "good, tell any agent \"drop the curated rule about ...\"."
)
# A level-1 or level-2 heading ends the section: its rules never hold one (check_rules).
_SECTION_END = re.compile(r"#{1,2} ")
# Tag-like text outside code spans: viewers treat it as HTML (see mdtext.py).
_RAW_TAG = re.compile(r"(?<!\\)</?[A-Za-z][A-Za-z0-9_-]*[^<>\n]{0,40}>")


@dataclass
class Section:
    key: str               # "global" or "workspace"
    files: list[Path]      # every file the section is written into
    max_chars: int
    max_rules: int
    note: str              # who reads it, for the curator


def sections(cfg: Config) -> dict[str, Section]:
    return {
        "global": Section("global", [p for p in cfg.global_files if p.is_file()], cfg.curated_global_chars,
                          cfg.curated_global_rules,
                          "every session of every account, in every project: only advice that holds anywhere"),
        "workspace": Section("workspace", [cfg.workspace_file] if cfg.workspace_file.is_file() else [],
                             cfg.curated_workspace_chars, cfg.curated_workspace_rules,
                             f"every session working under {cfg.projects_dir}: advice for these projects, "
                             "with a rule for one project only under a `### <project>` subheading"),
    }


def missing_named_copies(cfg: Config) -> str | None:
    """Why the global section cannot be written: an account `curated_accounts` names
    has no CLAUDE.md. None when every one has, or no account is named. A named account
    is one the operator wants the section in, so leaving it out in silence would leave
    that account without the rules for as long as nobody noticed."""
    if cfg.curated_accounts is None:
        return None
    missing = [str(p) for p in cfg.global_files if not p.is_file()]
    if not missing:
        return None
    return (f"curated_accounts names an account with no CLAUDE.md ({'; '.join(missing)}): create the file "
            "(it may be empty), or take the account out of curated_accounts")


def unwritable(cfg: Config, key: str, sec: Section) -> str | None:
    """Why section `key` cannot be written today, besides a conflict between its copies:
    its operator's rules cannot all be read, or a named account's copy is missing."""
    return unreadable_rules(sec) or (missing_named_copies(cfg) if key == "global" else None)


def unreadable_rules(sec: Section) -> str | None:
    """Why the operator's rules in the files of `sec` cannot all be read -- one of them
    imports a file that cannot be read -- or None. Claude Code skips such an import in
    silence, taking its rules out of every session; and the curator must not write the
    section meanwhile, since it can no longer see the rules it must not repeat."""
    found: list[str] = []
    for p in sec.files:
        found += [u for u in imports.follow(p, read_section(p)[0]).unreadable if u not in found]
    return f"its operator's rules cannot all be read: {'; '.join(found)}" if found else None


# --- the section inside a file ------------------------------------------------------------

def tidy(rules: str) -> str:
    """Rules as stored and compared: no trailing spaces, no blank lines at either end.
    An editor that trims trailing spaces on save must not look like a hand edit."""
    return "\n".join(ln.rstrip() for ln in rules.split("\n")).strip("\n")


def split_file(text: str) -> tuple[str, str | None]:
    """(the operator's text, the curated rules) of a file's text, line endings \\n. The
    rules are None when the file has no curated section.

    The section runs from its heading to the next level-1 or level-2 heading, or to the
    end of the file. Its rules never hold such a heading, so one found there was put
    there by the operator, or by a tool's installer appending to the file: from there on
    the text is the operator's, and the next write puts the curated section after it.
    """
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.rstrip() != SECTION_HEADING:
            continue
        end = next((j for j in range(i + 1, len(lines)) if _SECTION_END.match(lines[j])), len(lines))
        body = "\n".join(lines[i + 1 : end]).strip("\n")
        if body.startswith(INTRO_PREFIX):
            body = body.partition("\n\n")[2]
        before = "\n".join(lines[:i]).rstrip("\n")
        after = "\n".join(lines[end:]).strip("\n")
        return "\n\n".join(part for part in (before, after) if part), tidy(body)
    return text, None


def compose(own: str, rules: str) -> str:
    """The operator's text, then the curated section (omitted when it holds no rules)."""
    own = own.rstrip("\n")
    if not rules.strip():
        return own + "\n" if own else ""
    lead = f"{own}\n\n" if own else ""
    return f"{lead}{SECTION_HEADING}\n\n{SECTION_INTRO}\n\n{rules.strip()}\n"


def read_section(path: Path) -> tuple[str, str | None, str]:
    """(the operator's text, the curated rules or None, the file's newline); see split_file."""
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    own, rules = split_file(raw.replace("\r\n", "\n"))
    return own, rules, newline


class SectionError(RuntimeError):
    """A section cannot be written into one of its files; nothing was written."""


def _rewritten(path: Path, rules: str) -> bytes:
    """The bytes `path` should hold with `rules` as its curated section: the operator's
    text, and the file's line endings, exactly as they are."""
    own, _old, newline = read_section(path)
    if any(ln.rstrip() == SECTION_HEADING for ln in own.split("\n")):
        raise SectionError(f"{path} holds the curated section's heading twice; merge the two by hand")
    new = compose(own, rules)
    if split_file(new)[0].rstrip("\n") != own.rstrip("\n"):
        raise SectionError(f"{path}: writing the section would change the operator's text")
    return new.replace("\n", newline).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` without ever leaving it half-written: through a temporary
    file that then replaces it.

    Windows refuses the replace while another process has either file open without
    delete sharing -- typically an antivirus scan of a file just written, for a moment
    (seen in this module's tests) -- so it is retried, for up to ~18 s; if still
    refused, the file is written in place. A hard-linked file is written in place
    from the start, since replacing it would cut the link."""
    if path.exists() and path.stat().st_nlink > 1:
        path.write_bytes(data)
        return
    real = path.resolve()  # through a symbolic link, to the file it names
    tmp = real.with_name(real.name + ".curator-tmp")
    tmp.write_bytes(data)
    for attempt in range(40):
        try:
            os.replace(tmp, real)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    real.write_bytes(data)
    try:
        tmp.unlink()
    except OSError:
        pass  # a leftover temporary file harms nothing, and the next write replaces it


def write_section(path: Path, rules: str) -> None:
    """Replace the curated section of one file (`apply_rules` writes a whole section)."""
    data = _rewritten(path, tidy(rules))
    if path.read_bytes() != data:
        _atomic_write(path, data)


def check_rules(rules: str, sec: Section) -> list[str]:
    """Why `rules` cannot be written into `sec` (empty when it can)."""
    problems = []
    if len(rules) > sec.max_chars:
        problems.append(f"{len(rules)} characters, over the budget of {sec.max_chars}")
    n = sum(1 for ln in rules.split("\n") if ln.startswith("- "))
    if n > sec.max_rules:
        problems.append(f"{n} rules, over the budget of {sec.max_rules}")
    for ln in rules.split("\n"):
        if ln.startswith("#") and not ln.startswith("### "):
            problems.append(f"a heading other than ### : {ln[:60]!r}")
        if ln.rstrip() == SECTION_HEADING:
            problems.append("the section heading appears inside the rules")
    if _RAW_TAG.findall(re.sub(r"`[^`\n]*`", "", rules)):
        problems.append("tag-like text outside backticks (viewers read it as HTML)")
    named = imports.mentions(rules)
    if named:
        problems.append(f"{', '.join('@' + m for m in named)} outside backticks: Claude Code would load the file "
                        "into every session that reads the section")
    return problems


# --- rules as blocks ---------------------------------------------------------------------------

def _rule_blocks(rules: str, keep_other: bool = False) -> list[str]:
    """The rules as blocks: a `- ` line with its indented continuation lines. With
    keep_other, every other line (a heading, a blank line, stray text) is a block of its
    own, so the blocks join back into the text."""
    blocks: list[str] = []
    for ln in rules.split("\n"):
        if ln.startswith("- "):
            blocks.append(ln)
        elif ln.startswith("  ") and blocks and blocks[-1].startswith("- "):
            blocks[-1] += "\n" + ln
        elif keep_other:
            blocks.append(ln)
    return blocks


def _units(rules: str) -> list[str]:
    """What a section is made of, for comparing two versions of it: each rule, and any
    other line of text -- not blank lines or headings."""
    return [b for b in _rule_blocks(rules, keep_other=True) if b.strip() and not b.startswith("#")]


def _key(lead: str) -> str:
    """A rule's lead, normalised for matching: no bold marks, case, final punctuation
    or runs of spaces."""
    return re.sub(r"\s+", " ", lead.replace("*", "").strip().rstrip(".:;!").lower())


def _lead_text(block: str) -> str:
    """A rule's bold lead as written (its first line when it has none)."""
    m = re.match(r"- \*\*(.+?)\*\*", block)
    return m.group(1) if m else block.split("\n", 1)[0]


def _lead(block: str) -> str:
    """A rule's lead, normalised: what still identifies it when its wording changes."""
    return _key(_lead_text(block))


def _lost(old: str, new: str) -> list[str]:
    """The rules of `old` that `new` no longer has in any wording (their lead is gone too)."""
    units = _units(new)
    leads = {_lead(b) for b in units}
    return [b for b in _units(old) if b not in units and _lead(b) not in leads]


def _tidy_headings(rules: str) -> str:
    """Remove `### ` headings left with no rule under them, and runs of blank lines."""
    blocks = _rule_blocks(rules, keep_other=True)
    keep = []
    for i, b in enumerate(blocks):
        if b.startswith("#"):
            rest = next((x for x in blocks[i + 1 :] if x.strip()), None)
            if rest is None or rest.startswith("#"):
                continue
        keep.append(b)
    return tidy(re.sub(r"\n{3,}", "\n\n", "\n".join(keep)))


def _revert(current: str, after: str, before: str) -> str:
    """`current`, with one run's change (`before` -> `after`) taken back: what the run
    added or reworded is removed, wherever it is now, and what it removed returns,
    after whatever preceded it before the run. Changes made since that run (a rule
    dropped, an edit by hand) are kept."""
    seq_after = [b for b in _rule_blocks(after, keep_other=True) if b.strip()]
    seq_before = [b for b in _rule_blocks(before, keep_other=True) if b.strip()]
    added = set(seq_after) - set(seq_before)
    removed = [b for b in seq_before if b not in seq_after]
    out = [b for b in _rule_blocks(current, keep_other=True) if b not in added]
    for i, b in enumerate(seq_before):
        if b not in removed or b in out:
            continue
        anchor = next((seq_before[j] for j in range(i - 1, -1, -1) if seq_before[j] in out), None)
        at = out.index(anchor) + 1 if anchor is not None else 0
        if anchor is not None and anchor.startswith("#"):
            while at < len(out) and not out[at].strip():
                at += 1  # under a heading: after the blank line that follows it
        out[at:at] = ["", b, ""] if b.startswith("#") else [b]
    return _tidy_headings("\n".join(out))


# --- the history repository ------------------------------------------------------------------

def history_dir(cfg: Config) -> Path:
    return cfg.curated_dir


def _git(cfg: Config, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # The history is written by the curator, so its commits carry the curator's
    # name -- and do not depend on a git identity being configured.
    ident = ["-c", "user.name=agent-knowledge curator", "-c", "user.email=curator@localhost"]
    return subprocess.run(["git", "-C", str(history_dir(cfg)), *ident, *args], capture_output=True, text=True,
                          encoding="utf-8", check=check, env=gitread.clean_env())


def _stored(rules: str) -> str:
    """A section's text as its history file holds it."""
    return rules + "\n" if rules else ""


def _blob(cfg: Config, rules: str, store: bool = False) -> str:
    """The id git gives a history file holding `rules`; with store, the text also goes
    into the repository's object store. Bytes on stdin: a text-mode pipe would turn \\n
    into \\r\\n on Windows, and so change the id."""
    r = subprocess.run(["git", "-C", str(history_dir(cfg)), "hash-object", *(["-w"] if store else []), "--stdin"],
                       input=_stored(rules).encode("utf-8"), capture_output=True, check=True,
                       env=gitread.clean_env())
    return r.stdout.decode("ascii").strip()


def _ever_written(cfg: Config, rules: str) -> bool:
    """Whether the curator has ever written exactly this text as a section."""
    return _git(cfg, "cat-file", "-e", _blob(cfg, rules), check=False).returncode == 0


def ensure_history(cfg: Config) -> None:
    d = history_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".git").exists():
        _git(cfg, "init", "-q")
        (d / "README.md").write_text(
            "# Curated instruction-file sections\n\n"
            "Each file here is the current text of one machine-owned section (see akn/curate.py);\n"
            "the git history is what changed, when and why. CHANGES.md is the readable log.\n",
            encoding="utf-8", newline="\n")
        _git(cfg, "add", "-A")
        _git(cfg, "commit", "-q", "-m", "start the history of the curated sections")


def canonical(cfg: Config, key: str) -> str:
    p = history_dir(cfg) / f"{key}.md"
    return tidy(p.read_text(encoding="utf-8")) if p.is_file() else ""


def _show(cfg: Config, rev_path: str) -> str:
    """A history file as of a commit ("" where it did not exist)."""
    r = _git(cfg, "show", rev_path, check=False)
    return tidy(r.stdout) if r.returncode == 0 else ""


def commit_history(cfg: Config, message: str) -> bool:
    _git(cfg, "add", "-A")
    if not _git(cfg, "status", "--porcelain").stdout.strip():
        return False
    _git(cfg, "commit", "-q", "-m", message)
    return True


def _log(cfg: Config, lines: list[str]) -> None:
    """Add an entry at the top of curated/CHANGES.md (newest first)."""
    p = history_dir(cfg) / "CHANGES.md"
    old = p.read_text(encoding="utf-8") if p.is_file() else "# What the curator changed\n\n"
    head, _, rest = old.partition("\n\n")
    entry = "\n".join(lines).strip("\n")
    p.write_text(f"{head}\n\n{entry}\n\n{rest}".rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def apply_rules(cfg: Config, key: str, rules: str) -> None:
    """Make `rules` section `key`, in every copy of its file and in the history: in all
    of them or, when any copy cannot take it (SectionError, raised before anything is
    written), in none."""
    rules = tidy(rules)
    new = {p: _rewritten(p, rules) for p in sections(cfg)[key].files}
    # Stored before any copy is written: should one copy be written and another not, the
    # written one is then recognised as the curator's text (and brought back in line with
    # the history by `sync`), never mistaken for an edit by hand.
    _blob(cfg, rules, store=True)
    for p, data in new.items():
        if p.read_bytes() != data:
            _atomic_write(p, data)
    (history_dir(cfg) / f"{key}.md").write_text(_stored(rules), encoding="utf-8", newline="\n")


# --- state ----------------------------------------------------------------------------------------

def work_dir(cfg: Config) -> Path:
    return cfg.state_dir / "curator"


def state_path(cfg: Config) -> Path:
    return work_dir(cfg) / "state.json"


def load_state(cfg: Config) -> dict:
    p = state_path(cfg)
    state = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    # clusters: cluster id -> the curator's decision on it (status, sessions then, date)
    # rules: rule lead (_key) -> where the rule came from (section, clusters, since)
    for k, empty in (("clusters", {}), ("last_report", ""), ("dropped", []), ("hand_rules", []), ("rules", {})):
        state.setdefault(k, empty)
    return state


def save_state(cfg: Config, state: dict) -> None:
    work_dir(cfg).mkdir(parents=True, exist_ok=True)
    _atomic_write(state_path(cfg), json.dumps(state, ensure_ascii=False, indent=1).encode("utf-8"))


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _dropped(blocks: list[str], key: str) -> list[dict]:
    return [{"text": b.split("\n")[0][:300], "date": _today(), "section": key} for b in blocks]


def _prune(cfg: Config, state: dict) -> None:
    """Forget what no longer applies: rules written by hand, and the records of where
    rules came from, once the rule has left its section (dropped, undone, reworded)."""
    units = {key: _units(canonical(cfg, key)) for key in sections(cfg)}
    leads = {key: {_lead(b) for b in u} for key, u in units.items()}
    state["hand_rules"] = [h for h in state["hand_rules"] if h["text"] in units.get(h["section"], [])]
    state["rules"] = {k: r for k, r in state["rules"].items() if k in leads.get(r.get("section", ""), set())}


def reopen_at(prior: dict, recur_more: int) -> int:
    """The session count at which the curator looks again at a lesson it decided on.

    One made into a rule returns after `recur_more` more sessions: recurring despite
    the rule is a sign the rule is not working. A skipped one must also have grown by
    half (RECUR_FACTOR): a big lesson keeps recurring at its own steady rate, and three
    more sessions of a lesson already seen in thirty are no news."""
    n = int(prior.get("sessions", 0))
    if prior.get("status") == "skipped":
        return max(n + recur_more, math.ceil(n * RECUR_FACTOR))
    return n + recur_more


def settled(cfg: Config) -> dict[int, int]:
    """Cluster id -> the session count at which it is looked at again, for every
    cluster the curator has decided on. The nightly report leaves these out of the
    clusters it consolidates, so that its places go to lessons still undecided:
    otherwise, once the backlog is decided, every report would repeat it, and a new
    lesson would wait until it outranked all of it."""
    try:
        state = load_state(cfg)
    except (OSError, ValueError):
        # Unreadable decisions must not stop the nightly report: it then simply
        # ranks everything, as it did before the curator existed.
        return {}
    out = {}
    for cid, prior in state["clusters"].items():
        try:
            out[int(cid)] = reopen_at(prior, cfg.curator_recur_sessions)
        except (TypeError, ValueError):
            continue  # a malformed record decides nothing
    return out


# --- the files and the history in agreement -------------------------------------------------------

@dataclass
class Synced:
    done: list[str] = field(default_factory=list)              # what was brought into agreement
    adopted: set[str] = field(default_factory=set)             # sections taken over from a hand edit
    conflicts: dict[str, str] = field(default_factory=dict)    # sections left alone, and why


def sync(cfg: Config) -> Synced:
    """Bring every copy of each section and the history into agreement, before anything
    is written. A copy can differ from the history in three ways:

    * it has no section yet (a new account), or holds a version the curator wrote before
      (a write that did not reach it, a backup restored): it is brought up to date;
    * it was edited by hand. The section says not to, but an edit made anyway is the
      operator's decision, and nothing here may undo it, so the edit becomes the section,
      in every copy. A rule written or reworded by hand is the operator's from then on,
      and the curator must keep it word for word; a rule removed by hand counts as dropped;
    * copies were edited by hand in different ways: which edit to keep is the operator's
      call, so the section is left alone everywhere until they agree.
    """
    ensure_history(cfg)
    state = load_state(cfg)
    out = Synced()
    for key, sec in sections(cfg).items():
        canon = canonical(cfg, key)
        edited: dict[Path, str] = {}
        stale: list[Path] = []
        for p in sec.files:
            rules = read_section(p)[1]
            if rules == canon or (rules is None and not canon):
                continue
            if rules is None or (rules and _ever_written(cfg, rules)):
                stale.append(p)
            else:
                edited[p] = rules
        names = ", ".join(str(p) for p in (edited or stale))
        try:
            if len(set(edited.values())) > 1:
                out.conflicts[key] = f"its copies were edited by hand in different ways ({names}); make them agree"
            elif edited:
                new = next(iter(edited.values()))
                mine = [b for b in _units(new) if b not in _units(canon)]
                gone = _lost(canon, new)
                apply_rules(cfg, key, new)
                out.adopted.add(key)
                state["hand_rules"] += [{"section": key, "text": b, "date": _today()} for b in mine]
                state["dropped"] += _dropped(gone, key)
                out.done.append(f"- {key}: adopted the edit made by hand in {names}: {len(mine)} rule(s) written or "
                                f"reworded by hand, kept word for word from now on; {len(gone)} removed, now dropped")
            elif stale:
                apply_rules(cfg, key, canon)
                out.done.append(f"- {key}: brought up to date in {names}")
        except SectionError as exc:
            out.conflicts[key] = str(exc)
    before = (list(state["hand_rules"]), dict(state["rules"]))
    _prune(cfg, state)
    if out.done or before != (state["hand_rules"], state["rules"]):
        save_state(cfg, state)
    if out.done:
        _log(cfg, [f"## {_today()} (the files, checked before writing)", "", *out.done])
        commit_history(cfg, "sync: " + "; ".join(d[2:] for d in out.done))
    return out


# --- 1. prepare -------------------------------------------------------------------------------------

def newest_report(cfg: Config) -> Path | None:
    reports = sorted(cfg.reports_dir.glob("lessons-*.json"))
    return reports[-1] if reports else None


def stale_report(cfg: Config, now: float | None = None) -> str | None:
    """Why the curator has no fresh lessons to work from, or None when it has: there
    is no report at all, or the newest is over `stale_report_days` old. Either means
    the nightly run is not writing reports, which the curator would otherwise only
    show as one more morning with nothing new."""
    report = newest_report(cfg)
    if report is None:
        return "there is no lessons report yet: run `python -m akn all`, and see state/nightly-*.log"
    # Dated by its name, which a copy or a move of reports/ keeps; the file's mtime,
    # which they reset, only for a name that carries no date.
    try:
        written = datetime.strptime(report.name[len("lessons-"):][:15], "%Y-%m-%d-%H%M").timestamp()
    except ValueError:
        written = report.stat().st_mtime
    days = ((time.time() if now is None else now) - written) / 86400
    if days <= cfg.stale_report_days:
        return None
    return (f"the newest lessons report, {report.name}, is {days:.0f} days old: the nightly run has "
            "not written one since; see state/nightly-*.log")


def _recurring(entry: dict, newest: str) -> bool:
    try:
        cut = datetime.fromisoformat(newest.replace("Z", "+00:00")) - timedelta(days=RECURRING_DAYS)
        last = datetime.fromisoformat(str(entry.get("last_ts", "")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return last >= cut


def select_candidates(report: dict, state: dict, recur_more: int, limit: int) -> list[dict]:
    """The report's consolidated lessons worth considering, best first."""
    newest = report.get("newest_lesson_ts", "")
    out = []
    for e in report.get("consolidated", []):
        canon, cov = e.get("canon") or {}, e.get("coverage") or {}
        if canon.get("destination") == "not-worth-recording" or "cluster_id" not in e:
            continue
        verdict = cov.get("verdict", "not-checked")
        if verdict == "covered":
            if not _recurring(e, newest):
                continue
            verdict = "covered-recurring"
        elif verdict not in ("not-covered", "partly-covered"):
            continue
        prior = state["clusters"].get(str(e["cluster_id"]))
        n = len(e.get("sessions", []))
        note = ""
        if prior is not None:
            if n < reopen_at(prior, recur_more):
                continue
            note = f"seen in {n - prior['sessions']} more sessions since it was {prior['status']} on {prior['date']}"
        out.append({**e, "verdict": verdict, "recur_note": note})
    out.sort(key=lambda e: e.get("score", 0), reverse=True)
    return out[:limit]


def _candidate_md(cid: str, e: dict) -> str:
    canon, cov = e["canon"], e.get("coverage") or {}
    lines = [
        f"## {cid} -- {canon.get('title', '')}",
        f"{len(e.get('sessions', []))} sessions, {e.get('mentions', '?')} mentions | score {e.get('score')} | "
        f"verdict: {e['verdict']} | suggested home: {canon.get('destination')}"
        + (" | operator-sourced" if e.get("operator") else ""),
        "",
        f"Lesson: {canon.get('lesson', '')}",
        f"Applies when: {canon.get('applies_when', '')}",
        f"Projects: {', '.join(e.get('projects', []))}; last seen {str(e.get('last_ts', ''))[:10]}",
    ]
    if canon.get("notes"):
        lines.append(f"Consolidation notes: {canon['notes']}")
    rule = cov.get("rule")
    if rule:
        lines.append(f"Nearest existing rule ({rule.get('source', '')} > {rule.get('heading', '')}):")
        lines.extend("> " + ln for ln in str(rule.get("text", "")).splitlines()[:10])
    if cov.get("missing"):
        lines.append(f"What it misses: {cov['missing']}")
    if e.get("recur_note"):
        lines.append(f"Previously: {e['recur_note']}")
    lines.append("Evidence:")
    for m in (e.get("members") or [])[:3]:
        ev = " ".join(str(m.get("evidence", "")).split())[:280]
        lines.append(f'- "{ev}" ({m.get("source", "")}, {str(m.get("ts", ""))[:10]}, {m.get("project", "")})')
    return "\n".join(lines)


def _sessions_by_cluster(cfg: Config, cluster_ids: list[int]) -> dict[int, dict[str, str]]:
    """For each cluster: session id -> the time of the session's first lesson in it,
    read from the miner's store (only read). Empty when there is no store."""
    if not cluster_ids or not cfg.db_path.is_file():
        return {}
    db = sqlite3.connect(str(cfg.db_path))
    try:
        out: dict[int, dict[str, str]] = {}
        for i in range(0, len(cluster_ids), 500):
            ids = cluster_ids[i : i + 500]
            q = ("SELECT lc.cluster_id, l.session_id, MIN(l.ts) FROM lessons l "
                 "JOIN lesson_cluster lc ON lc.lesson_id = l.id "
                 f"WHERE lc.cluster_id IN ({','.join('?' * len(ids))}) GROUP BY lc.cluster_id, l.session_id")
            for cid, sid, ts in db.execute(q, ids):
                out.setdefault(int(cid), {})[str(sid)] = ts or ""
        return out
    finally:
        db.close()


def rule_evidence(cfg: Config, state: dict, key: str, rules: str) -> list[str]:
    """One line per rule of section `key`: who wrote it and when, and the sessions its
    lesson came up in -- in all, and after the rule was written. Once a section is
    full, every new rule displaces an old one, and judging that needs the evidence
    on both sides, not only the candidate's."""
    hand = {h["text"] for h in state["hand_rules"] if h["section"] == key}
    recs = {k: r for k, r in state["rules"].items() if r.get("section") == key}
    seen = _sessions_by_cluster(cfg, sorted({int(c) for r in recs.values() for c in r.get("clusters", [])}))
    out = []
    for b in _rule_blocks(rules):
        name = f"**{_lead_text(b)}**"
        if b in hand:
            out.append(f"- {name} -- written by the operator, by hand")
            continue
        r = recs.get(_lead(b))
        if r is None:
            out.append(f"- {name} -- no record of where it came from")
            continue
        since = str(r.get("since", ""))
        first: dict[str, str] = {}
        for cid in r.get("clusters", []):
            for sid, ts in seen.get(int(cid), {}).items():
                if sid not in first or ts < first[sid]:
                    first[sid] = ts
        if not first:  # its clusters are gone from the store (merged away, or no store)
            out.append(f"- {name} -- written {since}, from a lesson seen in {r.get('sessions', '?')} sessions by then")
            continue
        later = sorted(ts for ts in first.values() if ts[:10] > since)
        line = f"- {name} -- written {since}; its lesson came up in {len(first)} sessions in all, "
        line += (f"{len(later)} of them after the rule was written (the last on {later[-1][:10]})" if later
                 else "none of them after the rule was written")
        out.append(line)
    return out


def prepare(cfg: Config) -> dict:
    synced = sync(cfg)
    report_path = newest_report(cfg)
    if report_path is None:
        return {"skipped": "no report yet"}
    state = load_state(cfg)
    if state.get("last_report") == report_path.name:
        return {"skipped": f"{report_path.name} was already curated"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cands = select_candidates(report, state, cfg.curator_recur_sessions, cfg.curator_max_candidates)
    wd = work_dir(cfg)
    wd.mkdir(parents=True, exist_ok=True)
    for stale in ("input.md", "candidates.json", "draft.json"):
        (wd / stale).unlink(missing_ok=True)
    if not cands:
        state["last_report"] = report_path.name
        save_state(cfg, state)
        return {"skipped": "nothing new to consider", "report": report_path.name}
    secs = sections(cfg)
    blocked = dict(synced.conflicts)
    for key, sec in secs.items():
        why = unwritable(cfg, key, sec)
        if why and key not in blocked:
            blocked[key] = why
    ctx = wd / "context"
    ctx.mkdir(exist_ok=True)
    for old in ctx.iterdir():  # every file: a copy read from git need not end in .md
        if old.is_file():
            old.unlink()
    for key, sec in secs.items():
        if sec.files:  # every copy: each account's file may hold rules of its own
            own = imports.expand([(p, read_section(p)[0]) for p in sec.files])
            (ctx / f"{key}-operator-rules.md").write_text(own, encoding="utf-8", newline="\n")
    # The instruction files read from git, given to the curator for reference.
    refs = []
    for repo, rev, path in cfg.instruction_git:
        text = gitread.show(repo, rev, path)
        if text is not None:
            name = f"{len(refs) + 1}-{Path(repo).name}-{path.replace('/', '-')}"
            (ctx / name).write_text(text, encoding="utf-8", newline="\n")
            refs.append(f"context/{name} ({path} of {repo} at {rev})")
    ids = {f"C{i}": e for i, e in enumerate(cands, 1)}
    out = [f"# Curation input -- {_today()} (from {report_path.name})", ""]
    for key, sec in secs.items():
        current = canonical(cfg, key)
        n = sum(1 for ln in current.split("\n") if ln.startswith("- "))
        out += [f"## Section `{key}` -- read by {sec.note}",
                f"Budget: at most {sec.max_rules} rules and {sec.max_chars} characters "
                f"(now {n} rules, {len(current)} characters). The operator's own rules in that file: "
                f"context/{key}-operator-rules.md.", ""]
        if key in blocked:
            out += [f"This section cannot be written today ({blocked[key]}): leave it out of your draft.", ""]
        out += ["Current rules:", "", "````", current or "(none yet)", "````", ""]
        evidence = rule_evidence(cfg, state, key, current)
        if evidence:
            out += ["Where each current rule came from. A rule reaches only sessions that start, or compact, "
                    "after it is written, and a session can run for days:", "", *evidence, ""]
        hand = [h["text"] for h in state["hand_rules"] if h["section"] == key]
        if hand:
            out += ["The operator wrote or reworded these rules by hand. Keep each one exactly as it is (they "
                    "count toward the budget):", "", *hand, ""]
    out += ["## Dropped by the operator -- never add these again", ""]
    out += [f"- {d['text']} (dropped {d['date']})" for d in state["dropped"]] or ["(none)"]
    out += [""]
    if refs:
        out += ["For reference, the instruction files read from git (do not repeat them): "
                + "; ".join(refs) + "."]
    out += [f"The projects folder, which you may read to check facts: {cfg.projects_dir}."]
    if cfg.curator_notes.strip():
        out += ["", "## Notes on the projects folder", "", cfg.curator_notes.strip()]
    out += ["", "# Candidate lessons", ""]
    out += [_candidate_md(cid, e) + "\n" for cid, e in ids.items()]
    (wd / "input.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    (wd / "candidates.json").write_text(json.dumps({
        "report": report_path.name,
        "candidates": {cid: {"cluster_id": e["cluster_id"], "title": e["canon"].get("title", ""),
                             "sessions": len(e.get("sessions", []))} for cid, e in ids.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    out = {"candidates": len(ids), "report": report_path.name}
    return {**out, "blocked": blocked} if blocked else out


# --- 3. finalize ---------------------------------------------------------------------------------------

def finalize(cfg: Config) -> dict:
    wd = work_dir(cfg)
    cpath, dpath = wd / "candidates.json", wd / "draft.json"
    if not cpath.is_file():
        return {"skipped": "nothing was prepared"}
    cands = json.loads(cpath.read_text(encoding="utf-8"))
    if not dpath.is_file():
        return {"error": "the curator wrote no draft.json"}
    draft = json.loads(dpath.read_text(encoding="utf-8"))
    synced = sync(cfg)  # an edit made by hand while the curator ran is adopted, never overwritten
    state = load_state(cfg)
    today = _today()
    applied, problems, log = {}, [], [f"## {today} (from {cands['report']})", ""]
    for key, sec in sections(cfg).items():
        entry = draft.get(key) or {}
        current = canonical(cfg, key)
        rules = tidy(str(entry.get("rules", current)))
        changes = [c for c in entry.get("changes") or [] if c.get("change") != "kept"]
        if rules == current or not changes:
            continue
        if key in synced.conflicts:
            problems.append(f"{key}: kept as it was -- {synced.conflicts[key]}")
            continue
        unready = unwritable(cfg, key, sec)
        if unready:
            problems.append(f"{key}: kept as it was -- {unready}")
            continue
        if key in synced.adopted:
            problems.append(f"{key}: kept as it was -- it was edited by hand while the curator ran (its lessons "
                            "come back with the next report)")
            continue
        why = check_rules(rules, sec)
        if len(changes) > MAX_CHANGES_PER_RUN:
            why.append(f"{len(changes)} changes, over the {MAX_CHANGES_PER_RUN} allowed in one run")
        if not sec.files:
            why.append("none of its files exist")
        lost = [h for h in state["hand_rules"] if h["section"] == key and h["text"] not in _units(rules)]
        if lost:
            why.append(f"it changes or removes {len(lost)} rule(s) the operator wrote by hand")
        if not why:
            try:
                apply_rules(cfg, key, rules)
            except SectionError as exc:
                why.append(str(exc))
        if why:
            problems.append(f"{key}: kept yesterday's rules -- " + "; ".join(why))
            continue
        applied[key] = len(changes)
        log.append(f"**{key}**")
        for c in changes:
            log.append(f"- {c.get('change')}: {c.get('summary', '')} -- {c.get('reason', '')}")
            used = [cands["candidates"][cid] for cid in c.get("candidates") or [] if cid in cands["candidates"]]
            for cc in used:
                state["clusters"][str(cc["cluster_id"])] = {"status": c.get("change", "curated"),
                                                             "sessions": cc["sessions"], "date": today}
            _record_rule(state, key, c, used, today)
        log.append("")
    for s in draft.get("skipped") or []:
        cc = cands["candidates"].get(s.get("candidate"))
        if cc:
            state["clusters"][str(cc["cluster_id"])] = {"status": "skipped", "sessions": cc["sessions"],
                                                         "date": today, "reason": s.get("reason", "")}
    missed = draft.get("operator_rules_missed") or []
    if missed:
        log.append("**Rules of yours that agents keep missing** (for your consideration; nothing was changed):")
        log += [f"- {m.get('file', '?')}: \"{m.get('rule', '')}\" -- {m.get('suggestion', '')}" for m in missed]
        log.append("")
    if problems:
        log += ["**Not applied:**"] + [f"- {p}" for p in problems] + [""]
    if not applied and not missed and not problems:
        log += ["No changes.", ""]
    _log(cfg, log)
    summary = ", ".join(f"{k}: {n} change(s)" for k, n in applied.items()) or "no changes"
    commit_history(cfg, f"curator {today}: {summary}")
    state["last_report"] = cands["report"]
    _prune(cfg, state)
    save_state(cfg, state)
    return {"applied": applied, "skipped": len(draft.get("skipped") or []), "problems": problems,
            "operator_rules_missed": len(missed)}


def _record_rule(state: dict, key: str, change: dict, used: list[dict], today: str) -> None:
    """Remember where a rule came from: the clusters behind it, and the day it took
    effect. A sharpened or merged rule takes over the clusters of the rules it
    replaces, and dates from today, since its new wording reaches sessions from now."""
    lead = _key(str(change.get("rule", "")))
    if not lead:
        return  # the draft did not say which rule: nothing to record
    if change.get("change") == "retired":
        state["rules"].pop(lead, None)
        return
    clusters = {int(c["cluster_id"]) for c in used}
    sessions = sum(int(c.get("sessions", 0)) for c in used)
    for old in [lead, *(_key(str(r)) for r in change.get("replaces") or [])]:
        prev = state["rules"].pop(old, None)
        if prev:
            clusters.update(int(x) for x in prev.get("clusters", []))
            sessions = max(sessions, int(prev.get("sessions", 0)))
    state["rules"][lead] = {"section": key, "clusters": sorted(clusters), "since": today, "sessions": sessions}


# --- the operator's controls -----------------------------------------------------------------------------

def drop(cfg: Config, text: str) -> dict:
    """Remove the one curated rule containing `text` (case-insensitive) from every copy,
    and never let the curator add it again."""
    synced = sync(cfg)
    needle = text.lower().strip()
    if not needle:
        return {"error": "give a phrase from the rule to drop"}
    hits = [(key, b) for key in sections(cfg) for b in _rule_blocks(canonical(cfg, key)) if needle in b.lower()]
    if len(hits) != 1:
        return {"error": f"{len(hits)} rules contain {text!r}; give a phrase that matches exactly one",
                "matches": [b.split("\n")[0][:120] for _k, b in hits]}
    key, block = hits[0]
    if key in synced.conflicts:
        return {"error": f"the {key} section cannot be written: {synced.conflicts[key]}"}
    new = _tidy_headings("\n".join(b for b in _rule_blocks(canonical(cfg, key), keep_other=True) if b != block))
    try:
        apply_rules(cfg, key, new)
    except SectionError as exc:
        return {"error": str(exc)}
    state = load_state(cfg)
    state["dropped"] += _dropped([block], key)
    _prune(cfg, state)
    save_state(cfg, state)
    first = block.split("\n")[0]
    _log(cfg, [f"## {_today()} (operator)", "", f"- dropped from {key}: {first}"])
    commit_history(cfg, f"operator dropped a rule from {key}")
    return {"dropped": first, "section": key}


def _curator_runs(cfg: Config) -> list[tuple[str, str, list[str]]]:
    """The curator runs that changed a section and have not been undone, newest first:
    (commit, subject, the sections it changed)."""
    entries = [ln.split("\t", 1) for ln in _git(cfg, "log", "--format=%H%x09%s").stdout.splitlines() if "\t" in ln]
    undone = {s.split()[3] for _sha, s in entries if s.startswith("undo curator run ") and len(s.split()) > 3}
    runs = []
    for sha, subject in entries:
        if not subject.startswith("curator ") or sha in undone:
            continue
        changed = _git(cfg, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).stdout.split()
        keys = [k for k in sections(cfg) if f"{k}.md" in changed]
        if keys:
            runs.append((sha, subject, keys))
    return runs


def undo(cfg: Config) -> dict:
    """Take back the newest curator run that changed a section and has not been undone,
    keeping every change made since (a rule dropped, a hand edit adopted). A rule that run
    added or reworded counts as dropped, so the next run does not just add it back; a rule
    it retired or reworded returns as it was."""
    synced = sync(cfg)
    runs = _curator_runs(cfg)
    if not runs:
        return {"error": "there is no curator run to undo"}
    sha, subject, keys = runs[0]
    blocked = [k for k in keys if k in synced.conflicts]
    if blocked:
        return {"error": "; ".join(f"the {k} section cannot be written: {synced.conflicts[k]}" for k in blocked)}
    new = {k: _revert(canonical(cfg, k), _show(cfg, f"{sha}:{k}.md"), _show(cfg, f"{sha}^:{k}.md")) for k in keys}
    try:
        for k, rules in new.items():  # every copy of every section must take it before any is written
            for p in sections(cfg)[k].files:
                _rewritten(p, rules)
    except SectionError as exc:
        return {"error": str(exc)}
    state = load_state(cfg)
    for k, rules in new.items():
        gone = _lost(canonical(cfg, k), rules)
        state["dropped"] += _dropped(gone, k)
        apply_rules(cfg, k, rules)
    _prune(cfg, state)
    save_state(cfg, state)
    _log(cfg, [f"## {_today()} (operator)", "", f"- undid \"{subject}\" ({', '.join(keys)})"])
    commit_history(cfg, f"undo curator run {sha} ({subject})")
    return {"undone": subject, "sections": keys}


def show_log(cfg: Config, n: int = 40) -> str:
    """The newest `n` lines of what the curator changed, after a warning when it has
    had no fresh lessons to work from (`stale_report`)."""
    p = history_dir(cfg) / "CHANGES.md"
    log = "\n".join(p.read_text(encoding="utf-8").split("\n")[:n]) if p.is_file() else "no changes yet"
    stale = stale_report(cfg)
    return f"**Warning: {stale}.**\n\n{log}" if stale else log


def differing_copies(cfg: Config) -> list[str]:
    """Copies whose section differs from the history (reconciled by the next run's sync).
    Read-only: for showing the rules without writing anything."""
    out = []
    for key, sec in sections(cfg).items():
        canon = canonical(cfg, key)
        for p in sec.files:
            rules = read_section(p)[1]
            if rules != canon and not (rules is None and not canon):
                out.append(f"{key}: {p}")
    return out


# --- the morning run ------------------------------------------------------------------------------------

# File permission rules are written Edit(path) whichever tool writes the file
# (Claude Code: "Edit rules cover all file-editing tools"); a Write(path) rule is
# ignored. Tested 2026-10-02 in dontAsk mode: with exactly an Edit(<file>) rule, a
# Write to that file succeeded and a Write to any other file was denied. The bare
# tool name "Write" (or "Edit") must not be listed: it would allow writing anywhere.
CURATOR_TOOLS = ["Read", "Glob", "Grep", "Edit(state/curator/draft.json)"]
CURATOR_DENIED = ["Bash", "PowerShell", "NotebookEdit", "WebFetch", "WebSearch", "Agent", "Task"]


def claude_exe() -> str:
    import shutil

    found = shutil.which("claude")
    if found:
        return found
    fallback = HOME / ".local" / "bin" / "claude.exe"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("claude (Claude Code) is not on PATH or in ~/.local/bin")


def curator_command(cfg: Config, prompt: str) -> list[str]:
    """The headless Opus session: it may read, and write only state/curator/draft.json.
    No MCP servers, no shell, no web, and no transcript kept (the miner would
    otherwise mine the curator's own runs)."""
    # --add-dir lets it *read* the projects, to check a rule's claims about a
    # repository before writing it (nobody reviews its output); it still writes only
    # the draft, which the Edit rule pins.
    return [claude_exe(), "-p", prompt, "--model", cfg.curator_model, "--permission-mode", "dontAsk",
            "--add-dir", str(cfg.projects_dir),
            "--allowedTools", *CURATOR_TOOLS, "--disallowedTools", *CURATOR_DENIED,
            "--strict-mcp-config", "--no-session-persistence", "--output-format", "text"]


def curator_accounts(cfg: Config) -> list[Path]:
    """`curator_config_dir` first, then every other account, in order."""
    primary = Path(cfg.curator_config_dir)
    rest = [Path(a) for a in cfg.accounts if Path(a).resolve() != primary.resolve()]
    return [primary, *rest]


def _nightly_running(cfg: Config) -> bool:
    from .nightly import pid_alive

    lock = cfg.state_dir / "nightly.lock"
    try:
        pid = int(lock.read_text().strip() or 0)
    except (OSError, ValueError):
        return False
    return bool(pid) and pid_alive(pid)


def run(cfg: Config, wait_hours: float = 3.0) -> dict:
    """prepare -> the Opus curator -> finalize, logged to state/curator-<date>.log.
    Waits (up to `wait_hours`) while the nightly run holds its lock, so it curates
    the report that run writes rather than yesterday's."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    log = open(cfg.state_dir / f"curator-{_today()}.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def say(*parts) -> None:
        log.write(f"{datetime.now():%H:%M:%S} " + " ".join(str(p) for p in parts) + "\n")

    try:
        # The projects folder is what the curator writes the workspace section
        # into and may read to check facts: a wrong one is a stop, said aloud.
        problem = cfg.check_projects()
        if problem:
            say(problem)
            return {"error": problem}  # a stop that lasts until the operator acts: exit code 1
        deadline = time.time() + wait_hours * 3600
        while _nightly_running(cfg) and time.time() < deadline:
            time.sleep(60)
        if _nightly_running(cfg):
            say("the nightly run still holds its lock; skipping today")
            return {"skipped": "nightly run still running"}
        prep = prepare(cfg)
        say("prepare:", json.dumps(prep))
        # Said in the log and the result (whose "warning" makes the exit code 1):
        # without it a stalled nightly looks like a quiet morning, every morning.
        stale = stale_report(cfg)
        if stale:
            say("WARNING:", stale)
        warn = {"warning": stale} if stale else {}
        if "candidates" not in prep:
            return {**prep, **warn}
        prompt = (ROOT / "curator" / "PROMPT.md").read_text(encoding="utf-8")
        draft = work_dir(cfg) / "draft.json"
        # The chosen account first; if a run fails without writing its draft (an
        # account at its usage limit, or logged out), the next account tries.
        for config_dir in curator_accounts(cfg):
            env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config_dir)}
            t0 = time.time()
            r = subprocess.run(curator_command(cfg, prompt), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=3600)
            say(f"curator on {config_dir.name} exited {r.returncode} after {time.time() - t0:.0f}s:",
                (r.stdout + r.stderr).decode("utf-8", "replace").strip()[-2000:])
            if draft.is_file() or r.returncode == 0:
                break
        fin = finalize(cfg)
        say("finalize:", json.dumps(fin))
        return {**fin, **warn, **({"blocked": prep["blocked"]} if prep.get("blocked") else {})}
    except Exception as exc:  # noqa: BLE001 - an unattended run must log, not vanish
        import traceback

        say("failed:", traceback.format_exc())
        return {"error": repr(exc)}
    finally:
        log.close()

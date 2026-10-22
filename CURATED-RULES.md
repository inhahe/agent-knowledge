# The curated rules

The last section of each account's global CLAUDE.md, and of the workspace
`CLAUDE.md` in the projects folder (the folder `agent-knowledge` sits in), is titled
**"Lessons agents learned (maintained automatically)"**. It is written by the morning curator (`akn/curate.py`, the
"agent-knowledge curator" task at 06:00) from lessons that recurred across sessions, within a
size budget. The operator handed this job to the curator: there is no approval step.

**Don't edit that section yourself unless the operator asks you to.** It is rewritten every
morning. Everything outside it is the operator's, and the curator never changes it:
- a section added to the end of the file, below the curated one, stays where it is, and the
  curated section moves below it on its next write;
- an edit made inside the section by hand is adopted, never overwritten. The next run copies it
  into every copy of the file. From then on, a rule written or reworded by hand is kept word for
  word, and a rule deleted by hand counts as dropped.

So when the operator asks you to reword a curated rule, or to put a rule of theirs there, edit
the section in any one copy of the file. Their own new rules usually belong outside it: above the section of
their global CLAUDE.md (or in the file of rules it imports, if every account shares
one), or, for the projects folder only, above the section of its workspace `CLAUDE.md`.

Run the commands below in `agent-knowledge`, in the projects folder.

## When the operator says...

**"drop the curated rule about X"**: run `python -m akn curated --drop "<a phrase from that rule>"`.
- It removes that one rule from every copy of its file.
- It records that the curator must never add it again.
- If the phrase matches several rules, it lists them: pick a longer phrase.

**"what did the curator change?"**: run `python -m akn curated --log` and summarise the newest
entries. If it starts with a warning that the newest lessons report is old, say so first: the
nightly run that feeds the curator has stopped, and `state/nightly-*.log` says why. Each entry gives every change and the reason for it. It also lists any of the operator's
own rules that agents keep missing, with a suggested wording. Pass those suggestions on, but never
apply them: the operator's rules are the operator's.

**"undo the last curation"**: run `python -m akn curated --undo`. It takes back the newest
curator run that changed anything (run it again for the one before), and keeps whatever has
changed since, such as a rule dropped or an edit by hand.
- Rules that run added are recorded as dropped, so they do not come straight back.
- Rules it retired or reworded return as they were.

**"show the curated rules"**: run `python -m akn curated`. It also says when a copy of a file
differs from the others: edited by hand, or not written yet.

**"stop the curator"**: `Disable-ScheduledTask -TaskName 'agent-knowledge curator'` in
PowerShell (`Enable-ScheduledTask` turns it back on). The rules already written stay. Remove
them with `--drop`, or with `--undo` once for each run.

The full history, with diffs, is a git repository in `curated/`:
`git -C curated log -p`.

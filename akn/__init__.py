"""agent-knowledge: local-LLM tooling over Claude Code session transcripts.

The first tool is the lesson miner (`python -m akn ...`): it reads every Claude
Code transcript on this machine, finds the places where something went wrong or
the operator corrected an agent, asks a local model (llama.cpp) what reusable
lesson each one teaches, clusters the lessons across sessions, checks each
cluster against the instruction files that already exist (the CLAUDE.md files),
and writes a markdown report for a human to review.

Nothing here writes to a project's documentation. The output is a list of
*candidates*; deciding what becomes a rule stays a reviewed, version-controlled
edit. See design.md for why.
"""

__version__ = "0.1.0"

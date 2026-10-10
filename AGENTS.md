# Rules for this repo

Applies to humans, Hermes sessions, and any delegated coding agent (Codex,
Claude Code, OpenCode, Hermes subagents).

## Language

English everywhere: code, comments, docstrings, docs, commit messages. There is
no user-facing copy in this project, so no PT-BR anywhere.

## What we are building

An implementation of Victor Taelin's OptChat specification (see
`docs/optchat-spec.md`) as a Hermes `ContextEngine` plugin. The spec is
canonical. Hermes' own source is the reference for how to plug in.

## Non-negotiables

- **Implement from the spec, not from prior art.** `ottobunge/hermes-optchat`
  and `fourseven/hermes-optchat` may be read to learn Hermes internals (which
  function to hook, where the engine instance lives). Never copy their code,
  function bodies, names, or structure. Their design is not the reference;
  Taelin's gist is.
- **Bytes, never tokens.** Every size constant is UTF-8 bytes. Never introduce a
  tokenizer or a token estimate.
- **Nothing is ever lost.** The log is append-only. Never edit or delete a
  logged record. Summaries are lossy; raw records are not.
- **Never rebuild the view.** It is persisted and loaded. Rebuilding it
  invalidates every prompt-cache entry.
- **No global state, no `os.environ` reads.** The engine is cloned per agent
  (`clone_for_agent`); key anything durable by the home it is handed.

## Delegate rules (external coding agents)

- **No git.** Delegates edit files only. Never `git add`, `commit`, `push`, or
  rewrite history. Hermes commits.
- **Read order first.** Point every brief at `docs/optchat-spec.md` and
  `CONTRACT.md` before the files it owns.
- **Disjoint file sets.** Two concurrent delegates never touch the same file.
- **Proof required.** Every brief ends with a block demanding the raw tail of
  the test run and a `git diff --stat` limited to the files it owned, plus an
  explicit "assumed / not verified" list.
- **The contract wins.** If code and `CONTRACT.md` disagree, the contract wins;
  the delegate reports the conflict instead of resolving it silently.

## Verification

- Unit tests per module: `tests/test_<module>.py`.
- One end-to-end test that proves the swap: a session where the transcript is
  withheld and the model answers from the view plus `zoom`.
- One cache test: the view's prefix is stable across turns that do not merge.
- `python -m pytest -q` must be green before any commit.

## Commit style

Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). One logical
change per commit. Never hand back a dirty tree.

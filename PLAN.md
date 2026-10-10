# Build plan

**Goal:** Hermes keeps the body; we replace the memory and context subsystem with
OptChat's philosophy, through public seams.

**Architecture:** a Hermes `ContextEngine` plugin (`optchat/`) that swaps the
per-request context for a constant-size view of a permanent log, plus small
generic seams PR'd to `hermes-agent`.

**Budget:** a finite batch of Codex resets (coding labor) plus DeepSeek tokens on
the OpenCode Go route (runtime compaction). One slice per reset, one critical
path.

## Prep (no budget, done by Hermes)

- [x] Freeze the contract (`CONTRACT.md`) and the spec restatement (`docs/optchat-spec.md`).
- [ ] Fork `hermes-agent` under `illanrego`. Add a **git worktree** off the existing checkout (`~/Documents/coding/opthermes-seam`) on branch `feat/context-engine-cache-seam` — reuses the local objects, no 1.2 GB re-download.
- [ ] Create an isolated `HERMES_HOME` (e.g. `~/.hermes/profiles/optchat-dev`) for testing. Never touch the live profile.
- [ ] Confirm the Codex invocation: `$(npm prefix -g)/bin/codex`, `--approve-for-me`, `-m <slug>`, `-o <file>`, `pty=true`.

## Slices

Each slice ends green (`python -m pytest -q`), committed, with the raw test tail
in the report.

### Slice 0 — Spike (gpt-6-reserve)

**Objective:** prove `select_context` actually controls what the model sees.
**Files:** a throwaway engine in the isolated profile; no repo code.
**Method:** engine returns `[system]["<chat>\n0+1|SPIKE-VIEW\n</chat>"][current message]`
and withholds the transcript. Ask the model to state what it can see from earlier
in the session.
**Proof:** raw transcript showing the model cannot see the withheld messages, and
the exact call-site findings (order relative to cache decoration, whether the
turn-local tail survives).
**Done when:** the finding is written into this file under "Findings".

### Slice 1 — The seam (gpt-6-sol, high effort)

**Objective:** the 4-file PR candidate.
**Files:** as in `CONTRACT.md` § The seam.
**Proof:** the two new tests plus the existing context-engine suite, raw tail.
**Done when:** green, and the diff is limited to those 4 files.

### Slice 2 — store + tree (gpt-6-sol, high effort)

**Objective:** the durable substrate.
**Files:** `optchat/store.py`, `optchat/tree.py`, `tests/test_store.py`,
`tests/test_tree.py`.
**Must hold:** append-only with fsync; day-split files; torn-line recovery; one
writer; build-once; free nodes at or under 512 bytes; `needs_summary`.
**Done when:** green, including a test that a re-load never rebuilds an existing node.

### Slice 3 — view + compactor + prompts (gpt-6-sol, high effort)

**Objective:** the constant-size view and the machinery that feeds it.
**Files:** `optchat/view.py`, `optchat/compactor.py`, `optchat/prompts.py`,
`optchat/summarizer.py`, tests.
**Must hold:** `due = (T - last)/2^l`; sawtooth 64 KB to 128 KB; merge only pairs
whose parent is built; queue-driven (never scan); ruler + retry at most 5 keeping
the shortest; the five prompt lines and the rubric verbatim from the spec.
**Done when:** green, including a test that the view's rendered prefix is stable
across an appended message that triggers no merge.

### Slice 4 — engine adapter + tools + shim (gpt-6-sol, high effort)

**Objective:** make it a Hermes context engine.
**Files:** `optchat/engine.py`, `optchat/tools.py`, `optchat/__init__.py`,
`optchat/shim.py`, tests.
**Must hold:** `select_context` returns `[system][view][this turn only]` and never
mutates history; `zoom`/`date` schemas; `should_compress` always False;
`clone_for_agent` shares the durable store and copies only budget state; the shim
re-applies the view's cache marks after the host's decoration pass.
**Done when:** green, and the engine loads in the isolated profile.

### Slice 5 — end-to-end + cache (gpt-6-sol)

**Objective:** prove the whole thing on a real session.
**Proof:** one real session in the isolated profile, a `zoom` round-trip, a
long-message compaction, and a measured cache read share across turns.
**Done when:** the raw numbers are in this file under "Findings".

### Slice 6 — issue + PR (Hermes, no budget)

**Objective:** the upstream contribution.
**Method:** issue first (failure, consumers, the smallest proposed seam, the
catalog-eligibility argument), then the PR from slice 1.

## Parked deliberately

Admission / fail-closed, media and assets, backup and migration UX, catalog
submission, multi-provider cache marks, replay/report tooling. Naming them keeps
the budget off them.

## Findings

_(slice 0, slice 5 write here)_

## Open questions

- Does the host's sanitizer or cache-control pass alter a replaced list in a way
  the shim cannot undo? Slice 0 answers it.
- Is the turn-local tail reliably identifiable from `request_messages` alone, or
  do we need `incoming_message`? Slice 0.
- DeepSeek on OpenCode Go caches automatically; do we need marks at all on that
  route, or is the shim only for mark-based providers? Slice 5 measures it.

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
- [x] Forked `hermes-agent` to `illanrego`, `fork` remote added. Worktree at `~/Documents/coding/opthermes-seam` on `feat/context-engine-cache-seam` (off `origin/main`). No clone, no re-download.
- [x] Slice 0 spike run against the real host functions with the repo's own venv (`~/.hermes/hermes-agent/venv`) — no profile, no model, no tokens.
- [x] Codex confirmed: `codex-cli 0.157.1` at `$(npm prefix -g)/bin/codex`; models `gpt-6-astra` / `gpt-6-sol` / `gpt-6-reserve` / `gpt-6-luna`.
- [ ] Isolated `HERMES_HOME` (`~/.hermes/profiles/opthermes-dev`) — still needed for slices 4 and 5 (live session).

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

### Slice 1 — The seam ✅ done (gpt-6-astra, high effort)

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

### Slice 0 — done, no budget spent (`spike/`)

Run: `PYTHONPATH=~/Documents/coding/opthermes-seam ~/.hermes/hermes-agent/venv/bin/python3 spike/slice0_swap_probe.py`
(the repo's own venv; no profile, no model, no tokens).

- **The swap works.** `_apply_context_engine_selection` honors `select_context`;
  the return replaces the request; persisted history is untouched. A non-overriding
  engine is short-circuited by an identity check; `[]` and exceptions fail open.
- **The replacement is not final.** After the hook the pipeline still runs
  `apply_rejected_thinking_suppression`, `_sanitize_api_messages`,
  `evict_stale_outbound_tool_images`, `_drop_thinking_only_and_merge_users`,
  a whitespace `.strip()` of every content string, tool-call canonicalization and
  surrogate stripping.
- **Adjacent `user` messages always merge** (concatenated with `"\n\n"`), so a
  `user` view is absorbed into the current turn and can never be a stable cache
  prefix. Measured in `spike/slice0_shape_probe.py`:
  - `[system][user:view][user:msg]` -> merged. Rejected.
  - `[system][assistant:view][user:msg]` -> survives. **Chosen.**
  - `[system][system:view][user:msg]` -> survives (second system message:
    provider-dependent, not chosen).
  - `[system][user:view][system:""][user:msg]` -> survives (needs an empty
    message, not chosen).
- The engine instance lives at `agent.context_compressor`; the hook is RFC #36765;
  the suite to extend is `tests/agent/test_context_engine_select_context.py`.

## Open questions

- Does the host's sanitizer or cache-control pass alter a replaced list in a way
  the shim cannot undo? Slice 0 answers it.
- Is the turn-local tail reliably identifiable from `request_messages` alone, or
  do we need `incoming_message`? Slice 0.
- DeepSeek on OpenCode Go caches automatically; do we need marks at all on that
  route, or is the shim only for mark-based providers? Slice 5 measures it.

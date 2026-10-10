# Build plan

**Goal:** Hermes keeps the body; we replace the memory and context subsystem with
OptChat's philosophy, through public seams.

**Architecture:** a Hermes `ContextEngine` plugin (`optchat/`) that swaps the
per-request context for a constant-size view of a permanent log, plus small generic
seams PR'd to `hermes-agent`.

**Sequencing (not a budget):** one brief per Codex run, disjoint file sets, verify
between runs. Nothing here costs money - the point of the discipline is a reviewable
diff and a tree that never breaks. Codex does the module work; Hermes writes the
briefs, reviews the diff, runs the tests, and commits.

## Prep (done by Hermes)

- [x] Freeze the contract (`CONTRACT.md`) and the spec restatement (`docs/optchat-spec.md`).
- [x] Fork `hermes-agent` to `illanrego`, `fork` remote added. Worktree at `~/Documents/coding/opthermes-seam` on `feat/context-engine-cache-seam` (off `origin/main`). No clone.
- [x] Slice 0 spike run against the real host functions with the repo's own venv - no profile, no model, no tokens.
- [x] Codex confirmed: `codex-cli 0.157.1`; models `gpt-6-astra` / `gpt-6-sol` / `gpt-6-reserve` / `gpt-6-luna`.
- [x] Isolated profile `opthermes-dev` created (`hermes profile create --clone`, so it carries its own `.env` without us touching secrets) and configured: `context.engine: optchat`, memory off, whatsapp off.
- [x] Test command for the engine repo: `~/.hermes/tools/uv-*/uv run --no-project --with pytest python -m pytest -q`.

## Slices

Each slice ends green, committed, with the raw test tail verified by Hermes (never
taken on the delegate's word).

### Slice 0 — Spike ✅ done (no model, no tokens)

**Objective:** prove `select_context` actually controls what the model sees.
**Result:** it does, and the view must be an `assistant` message. See "Findings".

### Slice 1 — The seam ✅ done (gpt-6-astra, high effort)

**Objective:** the PR candidate.
**Result:** branch `feat/context-engine-cache-seam`, commit `dd97682821`, 5 files,
+147/-2, pushed to the fork. Issue filed upstream:
https://github.com/NousResearch/hermes-agent/issues/136123

### Slice 2 — store + tree ✅ done (gpt-6-sol, high effort)

**Objective:** the durable substrate.
**Files:** `optchat/store.py`, `optchat/tree.py`, `tests/test_store.py`,
`tests/test_tree.py`. Commit `e31d54a`.

### Slice 3 — view + compactor + prompts ✅ done (gpt-6-sol, high effort)

**Objective:** the constant-size view and the machinery that feeds it.
**Files:** `optchat/view.py`, `optchat/compactor.py`, `optchat/prompts.py` and their
tests. Commit `8431a31`. (`summarizer.py` moved to slice 4b - it needs the host.)

### Slice 4a — engine adapter + tools (gpt-6-sol, high effort) — running

**Objective:** make it a Hermes context engine.
**Files:** `optchat/engine.py`, `optchat/tools.py`, `optchat/__init__.py`,
`tests/test_engine.py`, `tests/test_tools.py`.
**Must hold:** `select_context` returns `[system + the engine's lines][assistant: view]
[this turn only]`, never mutates history, never returns `[]`; the five behavioural lines
reach the model idempotently; `zoom`/`date` schemas; `should_compress` always False;
`clone_for_agent` shares the durable store and copies only budget state; ingestion is
idempotent across overlapping message lists.

### Slice 4b — summarizer + shim + live load (gpt-6-sol)

**Objective:** wire the real model route and run on stock Hermes.
**Files:** `optchat/summarizer.py`, `optchat/shim.py` and their tests.
**Must hold:** the summarizer routes through the host's auxiliary model for task
`optchat` (discovered from the checkout, never guessed), never fabricates, never logs
prompt text; the shim is idempotent, restorable, and no-ops safely if the host's private
function is gone; the engine loads and completes one turn in `opthermes-dev`.

### Slice 5 — end-to-end + cache (gpt-6-sol)

**Objective:** prove the whole thing on a real session.
**Proof:** one real session in `opthermes-dev`, a `zoom` round-trip, a long-message
compaction, and a measured cache read share across turns.
**Done when:** the raw numbers are in this file under "Findings".

### Slice 6 — PR (Hermes, no budget)

**Objective:** the upstream contribution.
**Method:** the issue is filed; open the PR from slice 1 once there is a signal, and
reference the issue.

## Parked deliberately

Admission / fail-closed selection, media and assets, backup and migration UX, catalog
submission, multi-provider cache marks, replay/report tooling, concurrency in the
compactor (v1 is sequential).

## Findings

### Slice 0 — the swap works, and the view's role is forced

Run: `PYTHONPATH=~/Documents/coding/opthermes-seam ~/.hermes/hermes-agent/venv/bin/python3 spike/slice0_swap_probe.py`

- **The swap works.** `_apply_context_engine_selection` honours `select_context`; the
  return replaces the request; persisted history is untouched. A non-overriding engine is
  short-circuited by an identity check; `[]` and exceptions fail open.
- **The replacement is not final.** After the hook the pipeline still runs
  `apply_rejected_thinking_suppression`, `_sanitize_api_messages`,
  `evict_stale_outbound_tool_images`, `_drop_thinking_only_and_merge_users`, a whitespace
  `.strip()` of every content string, tool-call canonicalization and surrogate stripping.
- **Adjacent `user` messages always merge** (concatenated with `"\n\n"`), so a `user` view
  is absorbed into the current turn and can never be a stable cache prefix. Measured in
  `spike/slice0_shape_probe.py`:
  - `[system][user:view][user:msg]` -> merged. Rejected.
  - `[system][assistant:view][user:msg]` -> survives. **Chosen.**
  - `[system][system:view][user:msg]` -> survives (provider-dependent, not chosen).
  - `[system][user:view][system:""][user:msg]` -> survives (needs an empty message).
- The engine instance lives at `agent.context_compressor`; the hook is RFC #36765; the
  suite to extend is `tests/agent/test_context_engine_select_context.py`.

### Slice 1 — the seam, and two review catches

- The delegate's tests were good but one had **no teeth**: the "hook mutates then raises"
  case threw `TypeError` on its first statement (`messages[0]["content"]` is the string
  `"sys"`, so `[0]["text"]` fails), so the `deepcopy` protection was never exercised.
  Repaired, then proven by removing the protection and watching the test fail.
- Boundary worth raising upstream: the hook only fires where the host's own caching is on
  (`agent._use_prompt_caching`). Pinned by a test. It does not bite our DeepSeek route,
  which caches automatically.
- The delegate's pytest route used ephemeral uv-cache paths that were already gone when
  we checked, so its run was not reproducible. Ours is.

### Slices 2-3 — the substrate and the view

- Store: day-partitioned JSONL, fsync per write, `flock` single writer, dense ids, torn
  *tail* line skipped; an id **gap** raises (ids are the identity everything keys on).
- Tree: build-once nodes, incremental ready queue (no tree scan), `free_text` for nodes
  needing no model call. Nothing created level-0 roots - caught in review, closed by
  adding `Compactor.seed()` as the only source of roots.
- View: `due` measured from the pair's **last** message, most-due-then-oldest merge, and a
  test asserting equivalence with Taelin's rollback `push` at every count through 2000.
- Compactor: at most 5 attempts, shortest line kept, never fabricates; a failed summary
  leaves the node unbuilt.
- Repairs made in review: `view.py` no longer reaches into `Tree._nodes` (new
  `Tree.nodes()`); the prompt wording was rewritten to the spec's canonical text and the
  tests updated to match.

## Open questions

- ~~Does the host's sanitizer or cache-control pass alter a replaced list in a way the
  shim cannot undo?~~ **Answered (slice 0):** it does run after us; the `assistant` view
  role is what survives.
- ~~Is the turn-local tail reliably identifiable from `request_messages` alone?~~
  **Answered (slice 0):** yes - everything from the last `user` message onward.
- DeepSeek on OpenCode Go caches automatically; do we need marks at all on that route, or
  is the shim only for mark-based providers? Slice 5 measures it.
- Should an engine be able to opt in to cache marks when the host's own caching is off?
  Raised in the issue for the maintainers.

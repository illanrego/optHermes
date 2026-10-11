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
  tests were updated to match.

### Slices 4a-4b — the engine is live, and the cache is real

Live in the isolated profile `opthermes-dev`, real calls on the user's
opencode-go / `deepseek-v4-flash` route:

- **The engine registers.** It ingests turns, builds free and model-written nodes, and
  persists `main/`, `tree/` and `view.json` under `<profile>/optchat/`. No fallback to
  the built-in compressor.
- **Compaction runs on the host's auxiliary route.** `state.db` shows
  `task='optchat'`, `provider='opencode-go'`, `model='deepseek-v4-flash'` - inherited
  from the main model with no config.
- **The cache works, inside one session.** One session's main turn made two API calls:
  ~16,363 input tokens total, of which **16,000 were cache reads**. The second call paid
  for roughly 200 new tokens. That is the whole economic bet, measured, and it needs no
  cache marks on this route (DeepSeek caches by prefix).
- **It does NOT work across separate sessions.** Two one-shot runs minutes apart both
  reported `cache_read_tokens = 0`: the host's per-session prompt block differs, so the
  prefix never matches. The benefit needs a *continuous* session - the intended shape
  (one endless chat) - and the README should say so plainly.
- **The compactor is expensive on a reasoning model.** One compaction call produced
  **9,483 output tokens** (mostly reasoning) to write a single 512-byte line. Pin
  `auxiliary.optchat` to a cheaper or low-effort model. Config recommendation, not a bug.
- Also measured: the host's own prompt is ~16k tokens before our view. The engine
  replaces the *history*, not the system prompt or the tool schemas, so the saving scales
  with how much history the session would otherwise carry.

Two review catches worth keeping: the engine's `register()` could not read the Hermes home
from the context-engine loader (it passes a minimal collector with no plugin manager), so
registration silently fell back to the built-in compressor - fixed with `default_root()`
and an optional `root`; and `on_turn_complete` had to gate on the host's
`platform="subagent"/"cron"` signal, or a `delegate_task` child would have written its own
turns into the chat.

## Audit vs the original gist (2026-10-10)

The **memory** half is faithful; the **navigation** half was thin. Everything found is now
fixed except the two items that are features rather than fixes (per-subagent logs, media).
Suite: 79 tests.

Findings and their state:

### Was broken (fixed)

- **Tool names.** The prompt named `zoom`/`date`; the tools register as `optchat_zoom` /
  `optchat_date`, so navigation could not work at all. Fixed; a test now reads the names out
  of the engine's schemas and asserts each appears in the prompt.
- **The prompt was a paraphrase.** It omitted the view's line format (`id+n|text`), the
  kinds, the `(not summarized yet: zoom it)` rule, "the latest word is the truth" and
  "zoom is your only allowed mechanism". Fixed.

### Was missing (fixed in revision 2)

- Compaction calls shaped like turns: the same system prompt and tools, plus a `<chat>`
  context view - so they read the turns' cache entry, and so the merge template's
  "`<chat>` may hold their messages in more detail" stops referring to nothing.
- Admission: delivered by the end-of-turn background pass, bounded at 8 model nodes per
  turn. Honest limit: the inline settle can only clear free work, and the turn path must not
  start a model call, so a turn can still see a placeholder for a long message the
  background has not reached.
- The `work` and `note` kinds.
- Long text split into several messages, never cut.
- Compaction concurrency: up to 8 calls at once.

### Different by necessity (documented, defensible)

- The view is an `assistant` message - forced by Hermes merging adjacent `user` messages
  (measured in slice 0).
- The original has one prompt for turns and compactions; we split it into `SYSTEM_PROMPT`
  and `COMPACTION_SYSTEM`. Same information, different shape.
- The prompt names the registered tools, not the gist's bare `zoom`/`date` (namespacing is
  deliberate: a bare `zoom` is collision-prone).

### Still parked, with the reason

- **Per-subagent logs and `zoom("Name")`.** The spec keeps each subagent's steps in its own
  log; we gate children out of writing entirely, so there is no child log to open. Doing it
  properly means giving each child its own root plus a `work` record that can resolve back
  to it - a feature, not a fix.
- **Media and images** (`zoom(id, 1)` "gives message id whole, with its images"): needs the
  host's media plumbing.
- Catalog submission; multi-provider cache marks.

### Wrongly parked as a "feature" - being implemented (revision 3)

- **Per-subagent logs and `zoom("Name")`.** These are the design, not an add-on:
  `zoom("Name") gives an agent's whole chat` is in the canonical tool list, and the spec says a
  subagent's *"own steps stay in its own log (`zoom("Name")`), out of the chat"*. They were
  lost to a shortcut of ours - we gated children out of writing entirely instead of giving each
  child its own log, keeping "out of the chat" and deleting "its own log". Being implemented: a
  child writes to `<root>/agents/<key>/`, the master keeps one `work` record, and
  `optchat_zoom` takes a string to open that child's chat.
- **Media.** `zoom(id, 1) gives message id whole, with its images` is the design too. Being
  implemented: a message's media is archived under `<root>/assets/` and returned by
  `zoom(id, 1)`; the view and the compactor stay text-only.

### Not ours

Catalog submission (distribution) and multi-provider cache marks (host-side).

### Faithful, verified

Log rules (append-only, one writer, fsync, day-split, thoughts never logged, tool output
clipped head+tail 30k) · 512-byte nodes, build-once, free short nodes · `due` from the
pair's last message (push-equivalence through 2000) · the 64-128 KB sawtooth · persisted
view, never rebuilt · no dates in the view · the placeholder line · the ruler and the at
most 5 attempts keeping the shortest · the rubric's priority order · the five behavioural
lines · `zoom(id, n)` semantics · the cache discipline (16,000 of ~16,363 input tokens read
from cache inside one session) · **autonomous zoom, observed**: with the tools reachable the
model opened a folded 32-message line down to the messages itself, unprompted.

### Host-side blocker: the engine's tools are gated behind a toolset flag

`_inject_context_engine_tools` (`agent/agent_init.py`) injects an engine's
`get_tool_schemas()` into `agent.tools` only when `agent.enabled_toolsets` contains
`context_engine`. That list comes from `_get_platform_tools(cfg, "cli")`, which auto-adds
`context_engine` when `_context_engine_active(config)` is true (i.e. `context.engine` is not
`compressor`). In the one-shot path (`chat -q`) the entry does not survive resolution, so
**the engine runs while its tools never reach the model**: `select_context` is honoured and
the log/view are written, but `optchat_zoom` / `optchat_date` are invisible.

Measured in `opthermes-dev`:

- `hermes -p opthermes-dev chat -q "list tools containing optchat"` -> the model answers
  `NONE`.
- `hermes -p opthermes-dev -t context_engine chat -q "..."` -> the model has both tools, and
  calls `optchat_zoom(0, 32)` then `optchat_zoom(16, 16)` on its own.

Workaround: pass `-t context_engine`. The real fix is host-side - an active engine's tools
should be injected because the engine is active, not because a toolset entry survived
resolution. It is the same family as the cache seam (the engine needs first-class
reachability) and belongs in the conversation around the PR.

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

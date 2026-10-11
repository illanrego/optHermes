# Using optHermes

## The shape of it

Hermes keeps the body - tools, providers, skills, gateway, delegation. The memory and context
subsystem is the engine: one append-only log per profile, a binary summary tree over it, a
constant-size view injected in place of the transcript, and `zoom`/`date` navigation. Hermes' own
memory files are off and its lossy compression never fires.

## Setup

- Link the engine into a profile: `ln -s ~/Documents/coding/opthermes/optchat <profile>/plugins/optchat`
- Profile config: `context.engine: optchat`, `memory.memory_enabled: false`,
  `memory.user_profile_enabled: false`.
- Sizes live in `<profile>/optchat/config.json` (`view_min_bytes`, `view_max_bytes`, `node_bytes`).
- **Run with `-t context_engine` until the host toolset gate is fixed.** Without it the engine runs
  - memory works - but the model has no `optchat_zoom`/`optchat_date`, so it can read the view and
  never open it. Ask the model to list tools containing `optchat`: if it answers `NONE`, that gate
  bit.

## How to live in it

- **One profile = one chat, forever.** Don't make a profile per topic; that is the fragmentation
  you are trying to leave. If you want a separate stream (work vs personal), that is a profile
  decision, made once.
- **Prefer a continuous session** - a gateway bot (WhatsApp) or a long-lived CLI/TUI session. That
  is where the cache lives: inside one session the prefix is reused (measured 16,000 of ~16,363
  input tokens); across separate `chat -q` invocations the *memory* continues but the cache does
  not.
- **Don't `/new` to clear context.** There is nothing to clear - the view is the context and the
  log is the memory. `/new` only opens a fresh window onto the same chat.
- **Trust the fold; ask for zoom only when you want detail.** The prompt tells the model to zoom
  when a line is too vague to act on, so you should not have to say it. When it answers from a
  summary and you want the exact words: "zoom into that".
- **Compaction cost.** Keep the compaction on the same model as your turns
  (`auxiliary.optchat.provider: auto`, the default) and lower
  `auxiliary.optchat.reasoning_effort` to `minimal`/`low` instead of switching models - a
  different model silently forfeits the shared cache.
- **Don't shrink the view budget for real use.** A small `view_max_bytes` folds fast, which is
  useful for testing zoom and wrong for daily work.

## How to test it

- `scripts/tour.sh` - three turns, then the log, the tree, the view and the usage. `--state`
  prints the state with no model calls.
- By hand: `<profile>/optchat/main/YYYY-MM-DD.jsonl` (the log), `tree/` (the nodes), `view.json`
  (the ids the view covers).
- A zoom round trip: ask for a detail from an old turn, then check the log for a `tool` record
  named `optchat_zoom`.
- Cache, per call: `sqlite3 <profile>/state.db "select task, api_call_count, input_tokens,
  cache_read_tokens from session_model_usage order by last_seen desc limit 8"` - `task='optchat'`
  is a compaction. Only meaningful within one continuous session.
- Unit tests: `~/.hermes/tools/uv-*/uv run --no-project --with pytest python -m pytest -q`.
- **The failure mode to watch for:** if the log stops growing, the engine is not in the path and
  Hermes is managing context itself. `select_context` returning `None`, or raising, makes the host
  send its own transcript silently.

## The working process, versus regular Hermes

Short version, since the principles are already clear:

| | regular Hermes | optHermes |
|---|---|---|
| the unit | the session | the log |
| context | grows until it is compressed, lossily | a constant-size view, always |
| what is lost | the middle of long work | nothing; detail fades and is recovered by zoom |
| cost profile | climbs with the session | flat, if the session stays continuous |

What that changes in practice: you stop managing sessions - no "which chat?", no `/new` hygiene -
and long work stops losing its middle. What it does not change: Hermes still owns tools,
providers, skills, gateway and delegation. **Skills stay your procedural memory; the log is your
episodic memory.** Don't move skills into the log.

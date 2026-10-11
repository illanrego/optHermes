# Plan: the next batch (2026-10-11)

From the two independent analyses: mine (`docs/ANALYSIS.md`) and Codex's
(`~/.hermes/cache/scratch/codex-ANALYSIS-last.md`). Codex found more than I did, including a
regression I introduced in WP9. Both are treated as claims until verified in source - the two
sharpest were verified before this plan was written.

Ordered by the bar: irreversible loss first, then silent misrepresentation, then request correctness,
then economics.

## P1 - durability and raw loss (the irreversible class)

- **P1a · Recovery must reconcile the view independently of root construction.** `_recover()` skips a
  record whose level-0 node already exists, so when the tree save succeeded and the view save did not,
  the view is never extended - and a later append can fail on the id it expects. Extend the view for
  every record at or beyond its end, whether or not the node exists.
- **P1b · The checkpoint must stop overclaiming, and identity must be an occurrence, not a content
  hash.** A crash between a record's append and the checkpoint's advance replays that row's records -
  the code comment "never a skipped or duplicated row" is false. Keep duplicate-over-skip (the design
  prefers it), say so, and use the host's durable message identity where the host provides one instead
  of a content hash that cannot tell two identical messages apart.
- **P1c · Semantic corruption must be loud.** A valid-JSON record with an invalid schema or size is
  silently skipped today, and if it is the last record its id can be reused. Make it loud and make id
  reuse impossible.
- **P1d · Raw content must not be dropped.** A remote media reference is rejected at archival and
  skipped while ingestion advances; a delegate payload with `status="dispatched"` produces no records
  even when the host attaches completed `inline_results` to it. Record the reference; ingest the
  report.

## P2 - the turn boundary against the real host

- **P2a · Match the incoming message on stable fields, not the whole dict.** The host stamps
  persistence fields, strips them from the API copy, and hands us an unstripped clone: identity and
  equality both fail, and we fall back to the last user row - the steering bug WP1 fixed. Match on
  the stable subset (role, content, tool_calls, tool_call_id), scanning backwards for the newest
  match, and add a test with stamped rows.

## P3 - the view lifecycle

- **P3a · Freeze the view served for the current turn.** `on_message` ingests the current turn, so the
  live view now contains it and selection serves it *and* the same tail: the current turn appears
  twice, and the view is no longer rendered before the message is logged. Serve the view as it stood
  before this turn's rows.
- **P3b · Reconstructing a missing `view.json` must be loud.** Loading from the tree's nodes yields a
  different view, which kills every cache entry. Do it, say so, and do not let it pass as a normal
  load.

## P4 - the compaction's request identity (the cache's precondition)

- **P4a · A compaction must send the shape a turn sends:** `[system][assistant: view][user: task]`.
  Today it sends `[system][user: view + task]`, so the prefixes diverge before tools or markers can
  matter. Until this is fixed, no amount of tool or mark work can produce the reuse the design
  requires.
- **P4b · Then measure.** With P4a and WP9 in place, the milestone's turn→compaction leg becomes
  measurable on the real wire; compaction→compaction follows.

## P5 - admission and scheduling honesty

- **P5a · Stop calling it a background pass.** `on_turn_complete` drains model work and waits on
  futures, inside a host hook that runs during finalization. Either make it detached or describe it
  as it is.

## P6 - child navigation integrity

- **P6a · Read-only navigation must not write.** Scoped navigation opens the child's `Tree`, whose
  loader can seal a JSONL tail - a reader writing another chat without its lock. Use the non-sealing
  reader there.
- **P6b · A duplicate name must not make a report unreachable.** The first mapping is kept, but the
  new report carries the same name, so zooming its label opens the old child. Make the conflict
  visible to the model, not just to the log.

## P7 - document truth

- The read-only `Log` signature is documented inverted in CONTRACT revision 9.
- "Verbatim" overstates two task templates that collapse the spec's line breaks.
- The tracker's preamble says everything below is unimplemented while WP1-WP9 are marked done.
- `AGENTS.md` forbids environment reads and global state; `default_root()` reads `os.environ` and the
  shim installs a process-global wrapper.
- Two spec ambiguities to resolve: short messages as their own nodes vs "the view never contains
  whole messages"; and the contract citing gist requirements the local restatement omits.

## What this plan does not do

It does not attempt settled admission (a genuine blocking outcome needs a host change plus scheduling
that can make progress), does not fetch remote media, and does not touch the host's own fallback.
Those stay recorded as can't-own.

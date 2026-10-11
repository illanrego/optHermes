# Plan: the next batch (revision 2, 2026-10-11)

Revision 1 was challenged by Codex (`~/.hermes/cache/scratch/codex-PLAN-last.md`). This revision
takes its cuts: backward content matching is not identity, "rebuild and warn" still rebuilds, and
documenting a prohibited behaviour is not compliance. The first two are replaced by the design below;
the third becomes real work.

Ordering rule from the challenge: **contract decisions and the identity design come first**, because
the repository says the contract wins over code, and because P1b, P2a and P3a need one shared account
of occurrence, turn boundary, ingestion progress and the served view.

## Order 0 - decisions and design (no code)

**D1 · The target host.** The engine's guarantees hold only on a host that calls the seams. State it,
and make the failure paths say which host the engine is running on.

**D2 · Turn identity is `message_uid`.** The host mints a durable occurrence id (`message_uid`,
distinct from the physical `_row_id`) at insert, copies it back to the live row, and preserves it
through structural clones. It is **stripped from the API copy**. Therefore:

- locate the turn's row by UID in `conversation_messages`, which still carries it;
- map that row into the already-built request by the host's own invariant (current-turn rows are an
  ordered suffix, one API row per source row, system/prefill at the front) - **verify that invariant
  per host version, or ask the host for an explicit boundary**;
- never reconstruct the API tail from raw rows; its transformations belong to the host;
- where no UID exists yet (fresh assistant and tool rows at `on_message` time, which precede
  persistence), identity is **provisional** and is reconciled at the turn boundary.

**D3 · What "the view for this turn" means.** The **rendered bytes actually served**, captured on the
first request of the logical turn *before* its rows are ingested, and reused for the whole tool loop.
A copied pair list is not enough (an unresolved pair renders as a placeholder and changes when its
node appears) and neither is a frozen end (a mid-turn merge changes the representation).

**D4 · The recovery protocol.** A crash between a record's append and the checkpoint's advance still
replays. Specify how recovery recognizes the records already committed for one occurrence, including
a partially written multi-record row - a durable ingestion intent, not a heuristic. Never repair a
duplicate by deleting or rewriting logged records.

**D5 · Missing view state.** An empty new chat and missing state in an existing chat are different
things. The first may build a view; the second gets **exact-state recovery or an explicit
unavailable-state outcome**. Automatic reconstruction is cut.

## Order 1 - stop the silent corruption and the foreign writes

- **P1c** semantic corruption is loud, in the log **and the tree** (invalid schema or size, suppressed
  insertion failures), and an id can never be reused.
- **P6a** every foreign-tree reader is non-writing: scoped integer navigation, string-name zoom, and a
  child opening its master. All three load through a sealing reader today.
- **P1a** recovery reconciles view coverage **independently of root construction**: extend the view
  for every record at or beyond its end, whether or not its node exists.
- **P3b (rewritten)** per D5.

## Order 2 - the durability protocol

- **P1b** per D4, plus **recovery from a lost ingestion boundary**: today both hooks stop and the
  checkpoint is never repaired, so later turns stay stalled.
- **P1d** media failures must not be swallowed while the checkpoint advances; a failed source
  reference is preserved and retried. Recording a remote reference prevents losing the reference, not
  recovering the bytes - the contract's archived-media promise stays open. `View.save()` gains the
  directory sync `atomic_json()` already has.

## Order 3 - the turn boundary and the frozen view

- **P2a** per D2, covering steering, API transformations, restart and children.
- **P3a** per D3, including restart (recovery cannot reproduce a pre-turn rendering) and children
  (they hold the master's view object).

## Order 4 - truthful, usable retrieval

- **P6b** a duplicate name must leave the new report addressable - a durable, unambiguous label, not
  just a warning.
- The navigation gate (#16): an engine whose tools are unreachable hands the model a view it cannot
  open.
- A real transcript-swap acceptance test: a model choosing zoom and answering, not a test calling
  `handle_tool_call()` itself.

## Order 5 - the compaction's request identity, then measure

- **P4a** `[system][assistant: view][user: task]`, and amend the contract that documents the current
  envelope. Matching roles is necessary, not sufficient: the compaction's view advances and merges
  independently, so its first summary lines still differ - and an earlier shared system/tools prefix
  may still be reusable. Do not overclaim in either direction.
- **P4b** compare actual prefixes and serialization first: A->B and A->B->A, including a
  compaction-view merge, not repeated calls for one target. Then measure all three reuse
  relationships.

## Throughout

- **P5a** terminology: `on_turn_complete` drains and waits; it is not a detached background pass.
- **P7** factual corrections as their owning work lands: the inverted read-only `Log` doc, the
  "verbatim" overclaim, the tracker's status lines, the spec's two ambiguities.
- **P7 (executable)** `default_root()` reads `os.environ` and the shim installs a process-global
  wrapper and latch. AGENTS.md forbids both. Violations to fix, not to document.
- **Undersized nodes**: the config accepts budgets below the spec's 512 floor. Separate test-only
  budgets from supported production behaviour.

## Deferred, explicitly

Detached scheduling (needs its own ownership and lifecycle design), genuine blocking admission (needs
a host change plus scheduling that can progress), pagination, lifetime performance, durable
remote-media bytes, original event timestamps, long-message grouping.

# Decisions (Plan Order 0)

Written to unblock orders 2, 3 and 5. The contract wins over code, so these go in before the
implementations that depend on them.

## D1 - The target host

The engine's guarantees hold only on a host that calls the seams: `select_context(tools=...)`,
`ContextSelectionBlocked`, and `on_message`. That host is the seam worktree
(`feat/context-engine-seams`, plus the #16 gate fix), not stock Hermes. On a stock host the engine
still runs, but it is limited to what `cache_plan` and `on_turn_complete` provide - so every failure
path must say which host it is on rather than implying the full guarantee. Order 4's #16 fix is what
keeps an engine's navigation tools reachable once it is running at all.

## D2 - Turn identity is `message_uid`

The host mints `message_uid` at insert (distinct from the physical `_row_id`), copies it back to the
live row, and preserves it through the structural clones the engine receives. It is **stripped from
the API copy**, which is why the current whole-dict equality fails.

- Locate the turn's row by UID in `conversation_messages`, which still carries it.
- Map that row into the already-built request by the host's own invariant: current-turn rows are an
  ordered suffix, one API row per source row, with system and prefill inserted at the front.
  **Verify that invariant per host version** (it is not an ABC guarantee); if it ever fails to hold,
  the honest answer is to ask the host for an explicit boundary rather than guess.
- Never reconstruct the API tail from raw rows: its substitutions (`api_content`, composed context,
  sanitized tool calls) belong to the host.
- Where no UID exists yet - fresh assistant and tool rows at `on_message` time, which precede
  persistence - identity is **provisional** and is reconciled at the turn boundary, never invented
  onto the host's clone.

## D3 - "The view for this turn" is the rendered bytes

Capture the **rendered view bytes actually served**, on the first request of the logical turn and
*before* that turn's rows are ingested, and reuse that exact rendering for the whole tool loop.

- A copied pair list is not enough: an unresolved pair renders as a placeholder and its text changes
  when the node appears.
- A frozen end is not enough: a mid-turn merge changes the representation below it.
- The live persisted view may keep advancing and merging; only what was served is frozen.
- The snapshot stays navigable, because merging replaces view pairs without deleting tree nodes.
- Restart and children need an intentional boundary of their own: recovery cannot reproduce a
  pre-turn rendering, and a child holds the master's view object.

## D4 - The recovery protocol

The crash window is between a record's append (durable, fsynced) and the checkpoint's advance. A
content fingerprint cannot close it, because it identifies content rather than occurrence, and
because a partially written multi-record row is a real state.

The protocol: a **durable in-flight marker** in `ingest.json`, written *before* a row's records and
cleared when the checkpoint advances.

- The marker holds the row's identity and the log's record count *before* that row.
- Recovery reads the log's count - which is the durable truth - and skips exactly
  `count_now - count_before` records for that row before continuing. A row that wrote two of its
  three records resumes at the third; a row that wrote all three is skipped whole.
- Replay therefore never happens, and nothing is ever repaired by deleting or rewriting logged
  records - the log stays append-only.
- When no fingerprint matches at all (a compression rewrote the transcript), ingestion must not
  stall forever: record the failure once, and let a later turn retry from the newest matching row
  rather than freezing on an unchanged checkpoint.

## D5 - Missing view state

Already implemented in Order 1 (P3b): silent for an empty chat; for a chat with records, a loud
warning naming the consequence, a rebuild from the tree's saved nodes only, and a `"recovered"`
marker that later loads read. Nothing further to decide.

---

# Revision 2 - after the challenge (2026-10-11)

Codex challenged D1-D5. The verdicts that change them:

**D5 is REVERSED.** The paragraph above is wrong, and so is Order 1's implementation of it:
reconstructing from the tree's saved nodes **is reconstruction**, which AGENTS.md and plan revision 2
both forbid, and it does not recover the previous pair selection, batch state, or retained contexts.
**P3b must be redone**: exact-state recovery, or an explicit unavailable-state outcome - never
automatic reconstruction - and CONTRACT revision 12's clause must be amended.

**D2 needs a host-owned identity or a correlation mechanism.** End-of-turn reconciliation alone
reopens duplication: ingest a UID-less row provisionally, the host persists it under a UID, crash
before reconciliation, restart presents the UID while the engine knows only the provisional one -
ingesting it duplicates the occurrence, and matching by content can suppress a distinct identical
one (the host itself warns that distinct messages can share content, timestamp and tool-call ids).
The suffix invariant is supported by this assembly implementation but **not guaranteed by the
interface**: canonicalizing the live suffix, filtering rows, or inserting an ephemeral row breaks it,
and it is already false as a statement about the final wire request, because sanitization and
user-merge run *after* selection. So: pursue an explicit host-supplied boundary or source-to-request
mapping, keep version-specific conformance checks, and never claim to have proven the host's anchor
correct.

**D4's count delta is unsound as the whole protocol** - see the steer sent to the implementation:
`Log.count()` counts id positions including refused ones, and another append path (`notes.py`) can
interleave inside the range, attributing somebody else's record to this row and losing one of ours.
It must become a validated transaction: occurrence identity, a validated contiguous prefix, provable
range ownership, and an explicit recovery **failure** on a refused position, a mismatch, or an
impossible delta. "Replay never happens" is unsupported until then.

**D4's lost-boundary clause**: resume after the last **proven contiguous committed occurrence
boundary**, never after the newest content match - a later, distinct row with identical text can skip
intervening rows and the new occurrence. With no evidence, preserve the unresolved interval and
report unavailable recovery.

**D3**: keep the rendered bytes, but one snapshot per **logical turn and reading scope** -
reassembly, provider retry, failover and steering do not create a new one. Recompute the request
boundary and the current tail on each assembly while reusing the frozen view bytes; persist the
snapshot before current-turn ingestion if a restart must resume that same turn; a missing snapshot on
same-turn recovery is an explicit unavailable state; and a child captures its **own immutable
snapshot** of the master's view rather than holding the live object. Call it "selected for the turn",
not "actually served" - selection precedes sanitization, cache decoration and sending.

**D1**: describe capabilities and failure behaviour **per route**, not by branch name. The seam host
still sends the original transcript after two refusals and fails open on ordinary selection
exceptions, which can violate the view-only continuity law.

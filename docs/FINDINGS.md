# Findings tracker

Source: two adversarial review passes of this engine against the canonical design and the host.
Raw reports: `~/.hermes/cache/scratch/codex-REVIEW-last.md` (20 findings) and
`codex-REVIEW2-last.md` (triage challenge + merges + a new finding).

The review validated the core (tree arithmetic, `due` ordering, integer zoom, the log's
lock/append/fsync, the seam's position and precedence) and found defects elsewhere. Everything
below is **not yet implemented** as of this file.

**Status legend:** TODO · RUNNING · DONE (commit)

## Work packages

| WP | findings | subject | model | status |
|---|---|---|---|---|
| WP1 | #9, reasoning-summary, #18 | turn boundary, a reasoning-only summary, the host home | sol @ high | DONE (9e3efed) |
| WP2 | #13, #19 (attribution) | kinds and attribution, before more history is built | sol @ high | DONE (53f2c2b) |
| WP3 | #2, #3, #4 | compaction request identity and routing (cache) | astra @ high | DONE (39d5dfb) |
| WP4 | #5, #8 | maintained views, scheduling, admission (cache) | astra @ high | DONE (07865b0) |
| WP5 | #1, #7 | provider-aware cache planning (cache) | astra @ high | DONE (e103aa2) |
| WP5b | (WP5 follow-up) | group the view's cache blocks 4 lines at a time | sol @ high | DONE (af62f01) |
| WP6 | #10, #11, #12, tree tail | ingestion identity and recovery | by hand (codex limit) | DONE (acebbcd) |
| WP7 | #14, #15 | child identity and navigation | sol @ high | DONE (030fc2e) |
| WP8 | #6, #17, #20 | prompt volatility, media envelope, host fallback | sol @ high | RUNNING |
| WP9 | (seam wiring) | use the three host seams: tools for compactions, refusal for admission, per-message for live logging | TODO | TODO |

Order and why: **WP1** first (the turn-boundary bug is the most urgent active-turn defect and
the other two are tiny); **WP2** before sustained compaction (a node is immutable, so unlabelled
history can never be repaired later); **WP3 → WP4 → WP5** is the cache chain in the order the
review derived (identity and routing, then the views and scheduling that produce stable content,
then the marks that make it cacheable); **WP6** is a release blocker for durable use; **WP7** and
**WP8** follow.

## WP1 - turn boundary, reasoning summary, host home

- **#9** `select_context` anchors the retained tail to the last `user` row. Hermes inserts a
  steering message as another user row after a tool result, so the original task and the current
  tool work are dropped from the request before they were ever ingested. The host passes the real
  turn-start message as `incoming_message`; anchor to it, falling back to the last user row.
- **reasoning as a summary** `summarizer.py` uses `extract_content_or_reasoning`, so a
  reasoning-only reply can be persisted as a tree line - the design never logs thoughts. Require
  ordinary content and raise when there is none.
- **#18** `default_root()` reads `os.environ`; the host supplies `hermes_home` on
  `on_session_start`. Use the supplied home, keeping an explicit constructor root for standalone
  use.

## WP2 - kinds and attribution

- **#13** the kind is stored in the log but never reaches the tree: `seed` receives only text, so a
  root's compaction input has no kind and attribution is lost (the rubric requires it). Carry the
  kind with the source, include it in free-node text and in the compaction input, and count its
  bytes.
- **#19 (attribution half)** the canonical root task carries `{kind}: {message}`, and merges name
  the child range. Restore that wording.

## WP3 - compaction request identity and routing

- **#2** the turn's outgoing tools are captured only inside the host's cache-enabled branch, so on
  automatic-cache routes a compaction keeps just our two navigation schemas and cannot share the
  turn's prefix. Capture unconditionally.
- **#3** a different compaction model or provider silently forfeits sharing. Bind compactions to
  the turn's route by default and detect/report an incompatible override or fallback.
- **#4** the compactor's thread pool loses the host's routing context (a ContextVar), so a
  compaction can route by configuration instead of the active runtime and loses cache scope and
  session affinity. Propagate the context into each job.

## WP4 - maintained views, scheduling, admission

- **#5** `context_view` copies the live view and re-merges on every call; the result can include
  later messages and placeholders. Maintain a separate compaction view with its own batch state,
  ending at the target node (before the source message for a root; through the final source
  message for a merge).
- **#8** "eight concurrent calls" became "eight nodes per turn", free nodes count toward the
  limit, an interrupted merge batch is forgotten, and the view stops shrinking when a parent is
  missing - so the sawtooth cannot hold and the constant-size bound is not guaranteed. Keep
  scheduling until the required state is ready, and remember an unfinished batch.

## WP5 - provider-aware cache planning

- **#1** our plan marks the whole view as one text block including the closing `</chat>`, so
  appending a line invalidates it. On **explicit-breakpoint** routes mark complete view blocks
  with the breakpoint on the last one, keep a reusable tools/system boundary and a request-end
  mark, and stay inside the marker budget (Anthropic looks back only 20 block positions).
  On **automatic-prefix** routes add **no** marks and keep the accepted string form - OpenCode
  Zen's relay 400s on block-array content, which is exactly what marks produce
  (`agent_runtime_helpers.py:1880-1886` in the host).
- **#7** the stock shim installs one global wrapper bound to a captured engine instance and can
  exceed the marker budget. Dispatch through the request's active engine.

## WP6 - ingestion identity and recovery

**Completed by hand after the Codex usage limit stopped the delegated run mid-write.** Its partial
helper, `optchat/persistence.py`, was reviewed and integrated (the tail rule and the durable-rename
pair). Done in four passes - the torn tail and duplicate ids, the tree's counterpart, recovery of
missing roots and the unapplied suffix, and the durable ingestion checkpoint - each committed
separately, the suite green after every pass, and every new test verified to fail against the old
code first.

- **#10** ingestion runs only at end-of-turn (the host calls it best-effort), and dedup is a single
  counter: a restart duplicates or skips, and an equal-length transcript can be skipped entirely.
  Persist an ingestion checkpoint keyed by session and message identity, covering **all** records
  of one message.
- **#11** long root sources live only in memory, so a restart strands them; a crash between the log
  write and the view save leaves the log ahead of the view and wedges the next append. Recover
  missing roots from raw records and reconcile only the unapplied suffix.
- **#12** a parseable JSON tail without its newline makes the next append reuse that id, and on
  reload the first occurrence wins - the new record disappears. Define recovery, and reject
  duplicate ids. The **tree** has the same tail/duplicate path.

## WP7 - child identity and navigation

- **#14** a child renders the master's view but integer `zoom`/`date` resolve against the child's
  own log, so a master id returns a child record or "does not exist". String `zoom` returns a
  compressed view rather than a navigable child scope. Make the namespaces explicit.
- **#15** a delegate result carries no child identity, and the mapping is written only at turn
  completion, so a just-returned report cannot be zoomed in the same turn. Register on arrival and
  refuse to overwrite a name with a different child.

## WP8 - the remainder

- **#6** the host's system prompt carries dates, session id, model and platform, which the design
  forbids in the prompt; relocate runtime metadata after the view. This is a **third deviation**
  beyond the two we had listed.
- **#17** `zoom(id, 1)` returns archive references, not media; the host recognises real multimodal
  tool results through a structured envelope. Use it when supported; resolve supported remote
  references.
- **#20** if our engine cannot load, the host falls back to the built-in compressor; enabled idle
  compaction also touches `summary_target_ratio`, which we do not define. Host-side, but the
  engine should fail loudly rather than silently lose its guarantee.

## Upstream (built locally, not PR'd)

Three additive seams are implemented on branch `feat/context-engine-seams` in the seam worktree
(commit `cf1743d63a`), all in the same shape as the `cache_plan` seam:

- **tools to `select_context`** (#2's host half) - a deep copy of the request's outgoing schemas, so
  a compaction can be assembled like a turn on a route where no cache-plan hook fires.
- **`ContextSelectionBlocked`** (#8's host half) - an engine can refuse a request it cannot serve
  yet; the host re-assembles once, then sends the original messages with a warning.
- **`on_message`** (#10's host half) - a per-append observation hook, so an engine can log as things
  happen instead of waiting for the turn boundary.

Nothing is PR'd: on the user's instruction, nothing goes upstream until it has been tested on his
terms.

Still upstream, not built:

- **The toolset gate (#16)** an active engine whose navigation tools are unreachable: the host adds
  `context_engine` to `enabled_toolsets` only when resolution keeps it, and in the one-shot path
  it does not. Workaround `-t context_engine`. Worth its own defect issue.
- **The auxiliary path** does not run the conversation-loop planner, so a compaction cannot share
  a turn's plan even with identical intent.
- **Child identity in results (#15, host half)** include the child session id in every completed
  result entry.
- **The multimodal envelope (#17, host half)** let engine tools return media. As of this host the
  executor path already accepts the `_multimodal` envelope from an engine tool (the engine now
  uses it); what remains host-side is the `ContextEngine.handle_tool_call` docstring, which still
  says it must return a JSON string.
- **The silent fallback (#20, host half)** an engine that fails to import, register or start is
  logged at debug level and the session continues on the built-in compressor, so the guarantee
  disappears with no user-visible signal. The host should surface the fallback (a warning or a
  status line) or fail closed when `context.engine` names an engine that cannot load. OptChat now
  names its own failures loudly, but the host's messages remain the last word.

## The verification milestone

Measure, on the real wire and separately: **turn→turn**, **turn→compaction** and
**compaction→compaction** reuse, including a growth burst and a merge boundary. The recorded
16,000/16,363 measurement covers two calls inside one turn only. Restart-safety claims
additionally require WP6.

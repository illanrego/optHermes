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
| WP8 | #6, #17, #20 | prompt volatility, media envelope, host fallback | sol @ high | DONE (b7c185a) |
| WP9 | (seam wiring) | use the three host seams: tools for compactions, refusal for admission, per-message for live logging | by hand | DONE (bc0c2df, merged 4070515) |

The plan's orders (`docs/PLAN-next.md`, revision 2) run on top of these:

| Order | items | subject | status |
|---|---|---|---|
| Order 1 | P1c, P6a, P1a, P3b | loud corruption with no id reuse, non-writing foreign readers, view recovery independent of roots, explicit missing-view state | DONE (07d5fa8, merged f548c0d; P3b redone per D5's reversal in 8fa8cb9, CONTRACT revision 15) |
| Order 4 | P6b, #16, acceptance test | truthful, usable retrieval | DONE (0fa9f67 engine, host gate 81af99c011 in the seam worktree) |
| Order 0 | D1-D5 | the decisions: target host, `message_uid` identity, the frozen view, the recovery protocol, missing-view policy | written (dc243aa), challenged twice (revisions 2-3: 34cccbe, 675ae08); the D2/D4 verdicts are applied |
| Order 2 | P1b, P1d | the durability protocol | DONE (6dcfa92, merged 4a7d925; hardened in 56e857c, Order 2b, and dbd4633, Order 2c; CONTRACT revisions 16-17) |
| Order 3 | P2a, P3a | the turn boundary by UID, and the frozen served view | DONE (b27e94c; CONTRACT revision 18) |
| Order 5 | P4a, P4b | the compaction's request shape, then the prefix comparison and the cache measurements | P4a + engine-side P4b DONE (609b527; CONTRACT revision 19); the wire measurement rides the live run |

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

## WP9 - the three host seams wired

The seams built in the seam worktree (`cf1743d63a`) are in use - the refusal is the exception: it
is deliberately NOT used. (Uncommitted at the time of writing.)

- **tools to `select_context`** (#2's host half): captured exactly as `cache_plan` captures them
  (one shared rule: deep copy, latch, update the compactor), so a compaction sends the turn's tools
  on **any** route, including one where no cache-plan hook fires. `None` keeps today's navigation
  fallback and its one-time warning; nothing is invented, and an explicitly empty list is a
  capture.
- **`on_message`** (#10's host half): each appended message is ingested as the turn unfolds, with
  the same classification `on_turn_complete` uses (one shared walk). Ingestion starts at the
  checkpoint boundary, so the turn's user row and unnotified synthetic rows come with it, and the
  appended row is located in the list it is handed, never assumed to be the last row.
  `on_turn_complete` stays the reconciler; the checkpoint makes it a no-op for rows already
  ingested. A restart resumes at the boundary, and a crash mid-turn cannot skip a row.
- **`ContextSelectionBlocked`** (#8's host half) - deliberately not raised. `select_context`
  already settles for free before rendering (`compactor.settle(...)`, `model_calls=False` by
  default), which is the most admission possible without starting a model call on the turn path.
  Refusing would make the host re-assemble once and then send its own original request - the full
  transcript plus Hermes' own compression - which is strictly worse than serving a view that still
  carries placeholders: a stable view prefix is what compactions read, and a transcript swap
  destroys that. A turn served with an unsettled view logs one debug line instead, so the
  condition is visible when it happens and silent when it does not.

## Upstream (built locally, not PR'd)

Three additive seams are implemented on branch `feat/context-engine-seams` in the seam worktree
(commit `cf1743d63a`), all in the same shape as the `cache_plan` seam:

- **tools to `select_context`** (#2's host half) - a deep copy of the request's outgoing schemas, so
  a compaction can be assembled like a turn on a route where no cache-plan hook fires.
- **`ContextSelectionBlocked`** (#8's host half) - an engine can refuse a request it cannot serve
  yet; the host re-assembles once, then sends the original messages with a warning.
- **`on_message`** (#10's host half) - a per-append observation hook, so an engine can log as things
  happen instead of waiting for the turn boundary.

The engine now uses the tools and per-message seams; the refusal seam is deliberately unused - see
the WP9 section for why.

A fourth addition, the logical-turn boundary (`turn_boundary`, carrying a `TurnBoundary` record),
landed on the same branch at `202e30e78f`; the engine consumes it in Plan Order 3.

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

First live pass (2026-10-12, `opthermes-dev` on the opencode-go route, the seam build at
`202e30e78f` via the checkout's own launcher): one continuous session, four turns, a 26-node merge
wave. turn→turn: 9,099 new input + 9,216 cache-read on the session's main calls. turn→compaction: 7,936 cache-read across 8 calls (~597/call -
the shared system/tools prefix; the compaction views had merged independently, so only the earlier
prefix was reusable, matching the engine-side prediction in CONTRACT revision 19). Every turn
bound a host `TurnBoundary`; the selected view was frozen, served and retired each turn;
`optchat_zoom` and `optchat_date` round-tripped live. compaction→compaction reuse (same target)
still to measure on the wire; the engine-side harness shows a fresh request for the same target is
byte-identical. Earlier: the recorded 16,000/16,363 measurement covered two calls inside one turn
only.

Interactive follow-up (same day, the user's own two real turns on `LOGGED-3`): turn 1 was fully
engine-served, then a mid-session model switch (deepseek-v4-flash → deepseek-v4.1-flash) rebuilt
the runtime identity and the engine stepped aside for the rest of the session — "Chat log is
locked": a second `Log` instance (flock, exclusive, nonblocking) could not acquire the chat while
the session's original instance held it. Every later hook failed loudly, the host answered with
unmodified requests, nothing was corrupted, and the exit cleanup hit the same lock. Suspect: the
Suspect: the model-switch runtime rebuild re-creates the engine without closing the live one. Follow-up: repro
with consecutive interactive turns, find the rebuild path, reuse the instance (host-side fix).
Session wire totals after the ride: turns 6 calls / 13,437 new / 9,216 cache-read; compactions 12
calls / 32,847 input / **22,784 cache-read (69% cached)**.

Second interactive finding (same ride, the note-merge wave): the engine's post-turn drain (batched
`run_pending` model calls) keeps the turn thread alive and the turn lease held after the reply is
delivered, so the user's next message waits on "another Hermes process" that is the session's own
finishing turn (self-nesting; the holder pid is the same live window). Interrupting aborts the
wait but leaves the thread/lease in place; re-sending re-arms the wait and the message is not
processed. An exit during this window leaves the turn-lease row until its TTL. Operator mitigation:
clear the spent/dead-holder row (registry prune + targeted row delete). Fix direction: the drain
must not keep turn fencing open once the response is delivered, or the wait must recognize its own
session's draining turn. The delivery path itself stayed correct throughout: log, checkpoint and
view all consistent; nothing dropped from the log except unprocessed sends (which never existed).

Third interactive finding (the cascade, same ride): the checkpoint recorded a durable ``lost``
frontier ({rows: 44, committed_count: 137, observed_rows: 21, reason: "recovery unavailable:
ordered occurrence frontier absent"} - the engine was shown a 21-row history that did not contain
its last eight committed row-identities). By design ``_resume_row`` refuses to advance across it,
so ingestion froze at 137 records while the session grew to 116 messages. On the next turn the
selection then hit "no saved render predates the admitted occurrence" (03:37:49; same family as
02:32:41) and the engine stepped aside stickily for the process; the host fell back to the full
transcript, whose requests grew in=63,101 -> in=94,215 (cache 96-99%) toward Hermes' compression
threshold. Nothing was compressed; log, checkpoint and view never corrupted. Separately, the
compaction summarizer's HTTP call has no timeout: a hung call parked a thread for the full 600s
until the turn-liveness watchdog force-killed the turn (thread dump in agent.log). Diagnosis next:
(1) which host call site handed the engine a history missing its committed frontier, or what
churned the identity chain; (2) the ingestion-vs-selection ordering behind the capture refusals;
(3) bound the summarizer call. The stream is preserved as evidence; re-ride on a fresh root after
the fixes, as the test bed.

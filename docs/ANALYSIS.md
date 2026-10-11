# Analysis: are we still the design? (2026-10-11)

Written by the assistant after WP1-WP8 landed, to be read against Codex's independent pass over the
same question (`~/.hermes/cache/scratch/brief-ANALYSIS-goals.md`). The yardstick is
`docs/optchat-spec.md`; the evidence is the code and the tests, not the intent.

## 1. Where we are faithful

    log        day-partitioned JSONL, {i, kind, text, size, date}, the six kinds, one writer with a
               flush per record, and recovery the spec never asked for (torn tail, duplicate ids)
    tree       node(l,i) merges its children, build-once, a short source is its own node with no
               model call, 512 as floor and target, kind carried as "{kind}: {text}"
    view       the 64-128 KB sawtooth, persisted, loaded at start, never rebuilt - recovery only
               extends it
    compactions  a call like a turn, the turn's system prompt, the spec's task verbatim, the ruler,
               retry <=5 keeping the shortest, a 16-32 KB view with no placeholder, <=8 model calls
               at once, queue-driven, never scanning
    the turn   [tools][system][view][turn], fresh per request; the view rendered before the new
               message is logged; nothing carries over; zoom(id,n), zoom("Name"), date(id); subagents
               with their own log, one work line, master never waits
    prompt     the five lines, verbatim
    mistakes   #1,#2,#3,#4,#5,#6,#7,#8,#9,#10,#11,#12 all avoided. #9 was live until WP1, #10 until
               WP8.

## 2. Where we diverge

### (a) Deliberate, documented, same information

- The view is an `assistant` message. Forced by the host's adjacent-user merge; measured, not chosen.
- A compaction carries the spec's instructions in the task string instead of a second system prompt.
- The byte limit and ruler are parameterised rather than hard-coded 512.
- The compaction view ends at its target (stricter than the spec's "the chat's view merged further")
  so a compaction can never see a message that came after what it is summarizing.

### (b) Gaps not built yet

- Nothing outstanding. WP9 wired the seams: the turn's tools are captured on every route (so a
  compaction can read a turn's cache entry where no cache-plan hook fires), and the log is written
  as the turn unfolds through `on_message`, with the turn boundary as the reconciler. The refusal
  seam is deliberately unused, with the reason recorded.

### (c) A plugin cannot own these

- Freshness is emulated: the host replays a transcript and we replace it per request.
- The system prompt is the host's, so "never changes" holds only once WP8's relocation is in force.
- The wire is the host's: our plan is an input to its adapters, not the bytes.
- The tool list is the host's, and it is dynamic (toolsets, MCP, deferred tools) - the design forbids
  state in tools.
- The transcript exists beside the log. It is our input, not the model's memory, but it is a second
  store the design does not have.

## 3. Implementation vs design - the sharpest inconsistencies

1. **A compaction is not always "a call like a turn".** FIXED in WP9: the tools now arrive through
   `select_context` on every route, so a compaction sends what a turn sends. Still unproven on the
   wire - which is the milestone.
2. **We do not wait for the fold.** DECIDED, not fixed. Selection settles for free before rendering,
   which is the most admission possible without starting a model call on the turn path; refusing
   would make the host send its own transcript plus compression, which is strictly worse. A debug
   line makes an unsettled view visible when it happens. The law is unmet by choice, with the reason
   recorded in CONTRACT revision 11.
3. **We log at the turn boundary, not as things happen.** FIXED in WP9: `on_message` ingests each
   append, and the turn boundary reconciles rather than duplicating.
4. **Two stores, one of them unreachable.** The log is the memory; the host's transcript is a second
   copy that the model could reach until we removed `session_search` from the profile's tools. The
   removal is a config fact, not a code guarantee - a profile without that edit silently reopens the
   second door.
5. **The tool list is state in the tools.** Our navigation tools are merged into a dynamic host list;
   if that list changes between sessions, the cached prefix changes with it. Stable per profile, not
   guaranteed.

## 4. Documents vs reality

- `docs/FINDINGS.md` still carries the verification milestone as unmeasured: turn->turn is measured,
  turn->compaction and compaction->compaction are not. The economics of the whole design rest on
  them, and we have not proved them.
- The toolset gate (#16) is a live usability defect: an engine can be active and replacing the
  transcript while its navigation tools are unreachable in the one-shot path.
- Everything else is consistent: CONTRACT revisions 1-10, the tracker, and the code agree.

## 5. What to fix first

1. **The verification milestone** - measure turn->compaction and compaction->compaction on the real
   wire. WP9 is what makes the first one measurable; until it is measured, "it is cheaper" is an
   argument, not a result.
2. **The toolset gate (#16)** - an engine without its tools is worse than no engine: the model gets a
   view it cannot open.
3. **A live run** - WP1-WP9 are verified by tests, not by use. The engine has not been exercised
   inside a real host since WP4.

## 6. Blind spots

- The wire: no real-provider measurement of anything except turn->turn cache reads.
- Concurrent children: a delegate result carries no child session id, so two at once stay unlinked.
- A reader's torn-tail seal while a writer appends: named, not exercised.
- Recovery cost on a long log: O(records) at every session start, unbenchmarked.
- The shell: `terminal`, `file` and `code_execution` can read the log. Accepted, recorded, not
  prevented.

# Frozen contract

Version 1, frozen 2026-10-10. Changes require a dated note at the bottom of this
file. **If code and this file disagree, this file wins** until it is amended.

## Two workstreams

- **A. Engine** (this repo): the `optchat/` plugin package. Ships as its own repo.
- **B. Seams** (fork of `hermes-agent`, branch `feat/context-engine-cache-seam`):
  small generic additions to the `ContextEngine` ABC, PR'd upstream. Upgrades,
  not prerequisites.

## Byte budgets (UTF-8 bytes, never tokens)

| Name | Value | Meaning |
|---|---|---|
| `NODE_BYTES` | 512 | Target size of one summary line |
| `FREE_NODE_MAX_BYTES` | 512 | A source at or under this is its own node, no model call |
| `VIEW_MIN_BYTES` | 64_000 | Batch merges down to this |
| `VIEW_MAX_BYTES` | 128_000 | Batch merges trigger above this |
| `COMPACT_VIEW_MIN_BYTES` | 16_000 | Compaction view floor |
| `COMPACT_VIEW_MAX_BYTES` | 32_000 | Compaction view ceiling |
| `TOOL_CLIP_CHARS` | 30_000 | Tool output clipped head+tail |
| `COMPACT_CONCURRENCY` | 8 | Max concurrent compaction calls |
| `COMPACT_MAX_RETRIES` | 5 | Oversize retries; keep the shortest line |

## Module map (`optchat/`)

| File | Owns |
|---|---|
| `__init__.py` | `register(ctx)` exposing the engine (directory discovery also works) |
| `store.py` | the append-only log |
| `tree.py` | nodes, build-once, queue of buildable |
| `view.py` | the view: `due`, batch merge, render, persist |
| `compactor.py` | scheduling, concurrency, summarizer calls, retry |
| `prompts.py` | system prompt, compaction templates, the rubric |
| `summarizer.py` | the auxiliary model route |
| `engine.py` | the `ContextEngine` adapter |
| `tools.py` | `zoom`, `date` handlers |

## Interfaces

```python
# store.py
Record = dataclass(id: int, kind: str, text: str, size: int, date: str)
class Log:
    def append(self, kind: str, text: str) -> int      # returns id; fsync; day-split
    def get(self, i: int) -> Record | None
    def range(self, lo: int, hi: int) -> list[Record]   # inclusive
    def count(self) -> int
    # one writer, process-lifetime lock; a torn TAIL line is skipped at load

# tree.py
Line = dataclass(l: int, i: int, text: str, size: int)
class Tree:
    def put(self, l: int, i: int, text: str) -> None    # build once; raises if built or children missing
    def get(self, l: int, i: int) -> Line | None
    def children(self, l: int, i: int) -> tuple[tuple[int, int], tuple[int, int]]
    def free_text(self, l: int, i: int) -> str | None   # text needing NO model call, else None
    def buildable(self, count: int | None = None) -> list[tuple[int, int]]  # ready nodes; with count, only those covering <= count messages
    def needs_summary(self, l: int, i: int) -> bool     # unbuilt and free_text is None
    def save(self) -> None
    def load(self) -> None

# view.py
def due(l: int, i: int, T: int) -> float                # (T - last) / 2**l, last = (i+2)*2**l - 1
class View:
    def add_root(self, i: int) -> None
    def merge_batch(self, T: int) -> None               # down to VIEW_MIN_BYTES
    def lines(self) -> list[Line]
    def render(self) -> str                             # "<chat>...</chat>"
    def save(self) -> None
    def load(self) -> None

# compactor.py
Summarizer = Callable[[str], str]                       # injected: prompt -> one line

class Compactor:
    def __init__(self, tree: Tree, summarize: Summarizer, *,
                 node_bytes: int = NODE_BYTES, max_retries: int = COMPACT_MAX_RETRIES)
    def seed(self, i: int, text: str, kind: str) -> None   # register message i as node(0, i)
    def submit(self, l: int, i: int) -> None
    def run_pending(self, limit: int = COMPACT_CONCURRENCY) -> int   # nodes built; sequential in v1
```

**Load strictness (store).** A torn *tail* line is skipped and costs nothing. A gap in
the record ids is a different thing: `Log` raises `ValueError` rather than silently
re-indexing, because ids are the identity every other module keys on. Mid-file
corruption should be loud.

**`seed` is the only way roots appear.** `Tree` starts empty; `Compactor.seed(i, text, kind)`
registers message `i`, putting it as `node(0, i)` when its kind-prefixed text (`{kind}: {text}`,
the spec's level-0 `<input>` form) fits in `NODE_BYTES` (free, no model call) and leaving it
unbuilt when it does not (the model compresses it). Nothing
else creates level-0 nodes.

### `select_context` (the crux)

Returns, for this request only:

```
[system prompt] [assistant: <chat> view] [user: messages of THIS turn only]
```

"THIS turn" = everything from the message the host passes as `incoming_message`
onward, located in `request_messages` by identity and then by equality. When that
message is absent or cannot be located, it falls back to the last user row. (rev 4:
the host inserts steering messages as later user rows, so anchoring to the last user
row dropped the original task and the current tool work from the request before they
were ever ingested.) Every prior turn is dropped; the view stands in. Returns `None`
when there is nothing to change. Never mutates `conversation_messages`.

**The view MUST be an `assistant` message, not a `user` message.** Measured in
slice 0 (`spike/slice0_shape_probe.py`): the host runs
`_drop_thinking_only_and_merge_users` after this hook, and it merges adjacent
`user` messages by concatenation with `"\n\n"`. A `user` view would be absorbed
into the current turn's user message, which changes every turn, so the view could
never be a stable prompt-cache prefix and the design's economics break. Shapes
that survive: `assistant` view (chosen); a second `system` message (provider-
dependent, not chosen); a `user` view separated by an empty `system` message
(needs an empty message, not chosen).

```python
class OptChatEngine(ContextEngine):
    name = "optchat"
    def select_context(self, request_messages, *, conversation_messages=None,
                       incoming_message=None, budget_tokens=0) -> list | None: ...
    def get_tool_schemas(self) -> list: ...
    def handle_tool_call(self, name, args, **kwargs) -> str: ...
    def on_turn_complete(self, messages, usage=None, **kwargs) -> None: ...
    def on_session_start(self, session_id, **kwargs) -> None: ...
    def on_session_reset(self) -> None: ...
    def should_compress(self, prompt_tokens=None) -> bool: return False
    def clone_for_agent(self) -> ContextEngine: ...
    # once the seam lands; the ABC default is a no-op
    def cache_plan(self, messages, tools, **kwargs): return None
```

## The seam (workstream B)

```
File 1  agent/context_engine.py
        ContextEngine.cache_plan(self, messages, tools, **kwargs) -> Any | None
        """Return a cache plan for THIS request, or None to use the host's."""
        return None

File 2  agent/conversation_loop.py :: _redecorate_prompt_cache_for_provider
        after the built-in plan is applied, call the engine hook through the
        existing _engine_overrides_hook(engine, "cache_plan") gate; if it returns
        a plan, use that instead. Non-implementing engines stay byte-identical.
        (exact wiring settled in slice 1 against the real function body)

File 3  tests/agent/test_context_engine.py
        default no-op is byte-identical; an engine returning a plan wins

File 4  docs page context-engine-plugin: one paragraph
```

The engine instance lives at `agent.context_compressor`. `select_context` is
gated by `_engine_overrides_hook` and called from
`_apply_context_engine_selection` (conversation_loop.py).

## Config shape

```yaml
context:
  engine: optchat
memory:
  memory_enabled: false
  user_profile_enabled: false
auxiliary:
  optchat:
    provider: <provider>
    model: <cheap model>
```

`$HERMES_HOME/optchat/config.json`:

```json
{"view_min_bytes": 64000, "view_max_bytes": 128000, "node_bytes": 512,
 "compact_enabled": true, "compact_concurrency": 8}
```

## Out of scope for v1 (parked)

Admission / fail-closed selection, media and asset archiving, backup and
migration UX, catalog submission, multi-provider cache marks, replay/report
tooling. With admission parked, a turn may start before the view settles: v1
accepts a slightly stale view.

---

# Revision 2 - fidelity fixes (2026-10-10)

An audit against the original gist found the **memory** half faithful and the
**navigation** half thin. These supersede the interfaces above where they differ.

## Tool names

The registered tools are `optchat_zoom` and `optchat_date` - namespaced on purpose, since
a bare `zoom`/`date` is collision-prone. **The prompt must name them exactly as
registered.** The canonical prompt's `zoom(id, n)` wording is adapted, not copied. This
was broken: the prompt named `zoom`/`date` and navigation could not work at all.

## The turn prompt

`SYSTEM_PROMPT` must carry what the original puts in the model's head:

- the view's line format - `id+n|text` means the `n` messages from `id` on, oldest
  first, inside `<chat>`, newlines as spaces;
- the kinds list (`user`, `unii`, `tool`, `echo`, `work`, `note`);
- that an unsummarized message shows as `(not summarized yet: zoom it)`;
- `optchat_zoom(id, n)` opens line `id+n` into the two lines under it;
  `optchat_zoom(id, 1)` gives the message whole; `optchat_date(id)` gives its time;
- "The view is your memory, and its latest word on a thing is the truth";
- "Never grep or search memories manually; `optchat_zoom` is your only allowed mechanism
  to navigate the tree";
- the five behavioural lines the spec names, verbatim.

Missing any of these means the model cannot navigate the view.

## Compaction calls are shaped like turns

A compaction is a call shaped like a turn, so it reads the turns' cache entry.

```python
@dataclass(frozen=True)
class CompactionRequest:
    system: str    # the same system prompt a turn gets
    context: str   # the <chat> context view, merged further (COMPACT_VIEW_* bytes)
    task: str      # the compaction task
    tools: list    # the same tool schemas a turn sends (never called)

Summarizer = Callable[[CompactionRequest], str]
```

- `View.context_view(min_bytes=COMPACT_VIEW_MIN_BYTES, max_bytes=COMPACT_VIEW_MAX_BYTES) -> str`
  returns the chat's view merged further, rendered as `<chat>...</chat>`.
- The engine supplies the last assembled turn system text and the host's tool schemas.
- `make_host_summarizer` sends them through the aux route: the system as a system
  message, the context and task as the user message, and `tools` passed through.

## Admission (settle) - as built, and its honest limit

A view is *settled* when no line is a placeholder, which is true as soon as every message
has a node. A short message's node is free and is put at seed time; a long message's node
needs a model.

So the inline settle (`Compactor.settle`, `model_calls=False`) can only clear **free** work,
and the turn path must never start a model call: `select_context` runs synchronously inside
the host's request, and a summarizer call already in flight cannot be interrupted by the
deadline. **v1's admission is therefore delivered by the end-of-turn background pass**
(`on_turn_complete` -> `run_pending(limit=8)`), which leaves the next turn settled unless
more than 8 nodes were pending. A turn can still see a placeholder for a long message the
background has not reached - the honest gap.

Stock Hermes has no fail-closed sentinel, so the seam's `ContextSelectionBlocked` remains
the real upgrade: it would let the engine refuse a request rather than serve a stale view.

## Kinds

- `work`: a delegated child's report. Detected from a tool result whose call name is
  `delegate_task` (track tool-call id -> name while ingesting).
- `note`: memories imported from before the chat.
  `optchat.import_notes(log, compactor, view, texts)` appends each as a `note` record, so
  a new chat can start with the past.

## Long text

Text longer than `NODE_BYTES` is **split** into consecutive records of at most
`NODE_BYTES` (at a whitespace boundary when one exists), never cut. Tool results keep
their head+tail clip instead.

## Concurrency

`Compactor.run_pending` may run up to `COMPACT_CONCURRENCY` summarizer calls at once,
preserving build-once and the children-before-parent rule.

## Still parked

Media and asset archiving; per-subagent logs and `zoom("Name")` (children do not write, so
there is no child log to open); catalog submission; multi-provider cache marks.

---

# Revision 3 - the design, not an optimisation (2026-10-10)

Two items I had labelled "parked features" are part of the original design, and both were lost
to a shortcut of ours. The canonical tool list says
`zoom(id, 1) gives message id whole, with its images` and
`zoom("Name") gives an agent's whole chat`; spec section 6 says a subagent's *"own steps stay in
its own log (`zoom("Name")`), out of the chat"*.

The shortcut: when a `delegate_task` child's engine clone appended its own turns into the
master's chat, children were gated out of writing entirely. The design's answer is different -
the child writes to **its own log**, and only its final report reaches the chat as one `work`
message. We kept "out of the chat" and deleted "its own log", which is exactly the log
`zoom("Name")` opens.

## Per-subagent logs and `zoom("Name")`

- A child (`platform="subagent"`) **reads** the master's view (its first message is the view)
  and **writes** to its own chat root, `<root>/agents/<key>/`, with its own Log, Tree, View and
  Compactor. It writes nowhere else.
- The master's chat gets exactly one `work` record per child, starting `[Name]`.
- `optchat_zoom` accepts a **string** id as well as an integer. A string resolves to a child's
  chat and returns it (that child's rendered view - its whole chat); an integer keeps the
  halving behaviour.
- The name-to-key mapping is recorded at `<root>/agents/index.json` when the master logs the
  `work` record, so the name the model reads in the record is the name it can zoom by.
- `[Name]` is whatever the delegate result provides, else the child's key.

## Media

- `zoom(id, 1)` returns the message whole **with its media**: image/audio/file parts are
  archived under `<root>/assets/` at ingestion and referenced from the returned payload.
- Media never enters the view or a compaction prompt as base64. The view and the compactor stay
  text-only, and a fabricated placeholder never stands in for content.

## Not ours

Catalog submission (distribution) and multi-provider cache marks (host-side).

---

# Revision 4 - the turn boundary is the host's, not the last user row (2026-10-11)

Three defects, fixed in WP1.

## The turn boundary

The host inserts a **steering** message as another `user` row after a tool result. Anchoring "this
turn" to the last user row therefore dropped everything before it - the original task and the
current tool work - from the request *before it was ever ingested*, so the model lost the task
mid-turn. The host already passes the real boundary as `incoming_message`
(`conversation_loop._apply_context_engine_selection`, computed at `current_turn_user_idx` in
`turn_request_assembly`). The tail now starts at that message, located by identity first and by
equality second; the last-user rule remains the fallback when it is absent or cannot be located.

## A reasoning-only reply is never a summary

`summarizer.py` used the host's `extract_content_or_reasoning`, whose reasoning fallback let a
reasoning blob be persisted as a tree line - and the design never logs thoughts. Only ordinary
content can become a line now; a reasoning-only reply raises, and the node stays unbuilt.

## The home comes from the host

`default_root()` read `HERMES_HOME` from the environment. The engine now defers to the
`hermes_home` the host supplies at session start, keeps an explicit constructor root as the
override, and falls back to the environment only when neither exists. `register()` therefore
passes **no root** when there is no manager home, instead of freezing the ambient one.

---

# Revision 5 - compaction request identity and routing (2026-10-11)

- `cache_plan` captures the host's outgoing tools, including an explicitly empty list,
  and immediately updates the compactor. Startup and selection preserve that capture.
  Until a capture occurs, compactions keep the two navigation schemas and the engine
  emits one `RuntimeWarning` before its first end-of-turn compaction pass. On no-mark
  routes, the host's cache-policy gates the only hook carrying tools; matching the
  turn's tools requires a host seam passing them to `select_context`.
- `on_session_start(model=...)` records the turn model. The engine reads the profile's
  `auxiliary.optchat` route through the host task-config reader at session start and
  before end-of-turn compaction. Any explicit provider or model other than `auto` or
  unset produces one override warning per engine: compactions may not share the turn's
  cache entry. Routing remains host-owned and unchanged. Without the host reader,
  this check is unavailable.
- Every model job submitted by `run_pending` runs inside its own caller ContextVar
  snapshot, captured at submission. This preserves host route pins and accounting
  context without changing batching, admission, or `run_pending` / `settle` signatures.

These changes do not prove provider cache reuse; auxiliary cache decoration and the
missing no-mark tool seam remain host work.

---

# Revision 6 - maintained compaction views and scheduling (2026-10-11)

This supersedes revision 2's recomputed `context_view`, per-pass node limit, and
the claim that more than eight pending nodes necessarily leave the next turn stale.

## Compaction boundaries and stable requests

`View.context_view(min_bytes=16_000, max_bytes=32_000, *, end=None, target=None)`
returns a maintained compaction prefix, never a copy of the current live view.
`target=(l, i)` computes its exclusive boundary as `i` for a root and
`(i + 1) * 2**l` for a merge. An explicit `end` must agree with that boundary.
Without a target, `end` defaults to the live view's message count. In all cases,
the prefix stops at the first unbuilt root; no placeholder enters a compaction.

Each budget pair has an independent working list of pairs and unfinished-batch
flag. Built roots extend this prefix in order. It merges above its ceiling down
to its floor, resuming interrupted batches even after falling below the ceiling.
Historical boundaries split straddling nodes into built descendants, so no line
includes a source after the requested boundary. They never use the live pairs.

The first request for each target retains the exact built pairs used for its
context; later calls and retries reuse them even if earlier holes are filled or
more parents become available. Different targets sharing an end have independent
snapshots. Calls without a target retain a snapshot per explicit/default end.
The working view can finish a batch without changing an already issued request.
Snapshot metadata grows with requested targets; it stores pairs, not duplicate
summary text. Node immutability preserves the rendered bytes.

`View.save/load` now writes/reads a version-2 object in `view.json`, containing
the live pairs and unfinished-batch flag, working compaction states, and retained
request pairs. Legacy list-only files load their original pairs unchanged and
upgrade on save. Save the tree before the view; saved compaction pairs must refer
to built nodes. Selection saves its settled view as well as its tree.

## Scheduling and admission

`run_pending(limit=8, *, deadline=None, model_calls=True) -> int` keeps its
signature and returns the number of nodes built. `limit` now bounds concurrent
model jobs, capped at eight, rather than the total nodes built. Zero remains a
no-op. A pass drains ready work, including newly ready parents; free nodes consume
no model slot. Queues drive scheduling, without scanning the tree for work.

A root can start only when fewer than eight earlier roots are unbuilt, including
failed or in-flight roots. Merges require both children. Failed model jobs are
requeued for the next pass, while nodes waiting for readiness or admission are
reconsidered after progress. Each model job retains its own caller ContextVar
snapshot. Completion is applied in queue order.

Both live and working compaction batches retain their pending flag until reaching
their minimum byte size. The scheduler resumes them after each wave of builds.
If no parent exists that can shrink a view to its minimum, the batch stays pending.

`settle(max_nodes=64, timeout_s=20.0, *, model_calls=False)` keeps its signature
and its separate total-node and deadline bounds. Its default and the synchronous
selection path still never start a model call. A running model call cannot be
interrupted by the deadline. Provider failures, missing sources, or a deadline can
still leave a live placeholder; host fail-closed admission remains outside WP4.

---

# Revision 7 - host-owned cache policy and shared view planning (2026-10-11)

- `cache_plan(messages, tools, **kwargs)` still returns a host `PromptCachePlan`
  or `None`. It layers onto the host's explicit plan, preserving system and tool
  markers and their TTLs. With no assistant `<chat>` view, it returns `None`.
- `optchat.cache.plan_view_cache(messages, tools)` is the shared, request-local
  planner. It returns copied `(messages, tools)` sections or `None`; it never
  changes stored messages, the persisted view, or the caller's plan. The engine's
  `_plan_cache` captures outgoing tools before calling it, for both integrations.
- The request view is split into an opening-tag block, one block per group of
  four complete newline-terminated lines, and a closing-tag block. The last
  complete group gets a breakpoint, so that boundary stays byte-identical until
  another group fills; appending lines preserves the previous blocks and
  boundaries. The closing tag is excluded from that reusable prefix. A merge or
  a changed placeholder invalidates the affected prefix as before.
- The planner retains or adds a request-end mark at the last legal cacheable
  message. Tool-result eligibility and top-level versus part placement come from
  the host's existing marks. System/tools marks are protected; excess conversation
  marks are removed oldest first, keeping at most four breakpoints in total.
  The host layouts reserve at most two prefix marks. A plan already reserving
  four prefix marks is left to the host (`None`); with three, request end takes
  priority over the view mark.
- A view breakpoint is added only with fewer than 20 following content/tool-call
  block positions, including the closing tag. Longer current-turn tails keep
  the host's recent conversation marks. A growth burst beyond the provider's
  lookback window and a batch merge can still miss the preceding view cache.
- The seam call itself is the explicit-cache signal. The stock shim mirrors the
  host's `_use_prompt_caching` branch and prepared-MoA bypass, dispatching through
  the request's `agent.context_compressor`, including clones and later engines.
  Automatic-prefix requests keep the selection's string view; no provider-name
  detection or automatic-route marking is added. Upstream-seam detection,
  installation/uninstallation identity checks, and the missing-hook warning
  remain unchanged.

Provider cache-hit measurements and auxiliary-route cache decoration remain
unverified host work; local planning tests do not prove turn/compaction reuse.

---

# Revision 8 - durability and recovery (2026-10-11)

WP6, four passes. The log is the source of truth; everything else is derived and recoverable.

## The torn tail

`optchat/persistence.py` (`read_jsonl`, `atomic_json`, `sync_directory`) is the shared rule for both
append-only files. A line missing only its newline is a **complete record**: it is indexed and the
file is made whole. An unparseable tail fragment is **sealed** with NUL + newline, so it can never
parse later and is skipped on every subsequent load. Ordinary mid-file corruption **raises**, because
record ids are the identity every other module keys on.

## Duplicate ids

`Log` rejects two records with the same id instead of keeping the first. First-wins was a silent
loss: the later record was unreachable and the next append reused its id. `Tree` rejects a node
written twice with **different** text (a build-once violation) and tolerates an identical duplicate,
which is exactly what a crash between writing a node and dequeuing it produces.

## Recovery at session start

`OptChatEngine._recover()` reconciles the derived state with the log: every record without a level-0
node is re-seeded **from the record itself** (so a long source is never stranded in memory), and the
view is extended to cover every record beyond its end. The view is never rebuilt - a rebuilt view
renders differently and kills every cache entry. Recovery does free work only; starting a session
never waits on a model call.

## The ingestion checkpoint

`<chat_root>/ingest.json` = `{session, rows, fingerprints}`. The boundary is found by **row
identity** (a hash of role, content, tool_calls, tool_call_id), not by a counter: a counter resets on
restart (re-ingesting the whole transcript) and skips a transcript that happens to be the same
length. `CHECKPOINT_ROWS` recent fingerprints are kept, so a rewritten tail is still findable. The
checkpoint advances only after **all** records of a row are durable, so a crash mid-row re-ingests
that row instead of skipping its remaining records. When no recorded row can be found - a
compression rewrote the transcript - the engine ingests nothing and warns once, rather than guessing
a boundary that could duplicate history.

---

# Revision 9 - child identity and navigation (2026-10-11)

WP7. Navigation namespaces are explicit, and a child report is addressable in the turn it returns.

## The navigation scope

`optchat_zoom` and `optchat_date` accept an optional `scope` (a subagent's name):

- Without `scope`, an integer `id` resolves against the log whose view the caller is reading: the
  master's log/tree for the master and for a child (a child renders the master's view). A child's
  own chat is reached only with an explicit scope.
- With `scope="Name"`, the name is resolved through `<root>/agents/index.json` and `id` resolves
  against that child's chat (`<root>/agents/<key>/`). A string `id` opens its own child chat and
  must not be combined with `scope`.
- An unknown, empty, or unmapped scope is an error; navigation never falls back to another
  namespace silently. The scoped log opens read-only, because the child's own engine may still
  hold the chat's lock.
- Every zoom/date payload carries `scope` (the name, or null for the reading view); a child chat
  returned by `optchat_zoom("Name")` names its scope so its ids can be opened with
  `optchat_zoom(id, n, scope="Name")`. The system prompt states the rule.

`Log(root, read_only=False)` is the lock-free read handle for opening another chat's log while its
owner holds the exclusive lock. It never writes: no lock, no directory creation, no tail sealing,
and `append` raises. `read_jsonl(path, seal=False)` is its non-writing reader.
`optchat.tools.child_chat_root(root, name)` resolves a name to a safe chat directory or `None`
(the mapped key must be a single safe path component).

## Registration on arrival

`name -> child key` is recorded when a delegate result first becomes visible to the engine - during
selection of the running turn's request, not only at `on_turn_complete` - so the report is zoomable
by name in the same turn. The row's arrival resolution is reused at ingestion, so the logged
`[Name]` always matches the registered mapping. A name already mapped to a different child keeps
its first mapping and warns once (per name, per engine); re-registering the same pair is
idempotent. The host half (a delegate result carrying `child_session_id`) remains an upstream ask.

---

# Structural rules - laws enforced by absence (2026-10-11)

A law that must hold is enforced by **absence**, not by instruction. A prompt line is a preference
that decays under pressure; a tool that is not in the list cannot be called at all. The prompt's
"never grep or search memories manually; optchat_zoom is your only allowed mechanism to navigate the
tree" stays as belt and braces, but nothing depends on it.

Corollary: whenever the engine is enabled in a profile, **audit the tool surface for memory-shaped
tools** - including any MCP server that stores or reads notes - and remove them rather than
instructing against them. For the `opthermes-dev` profile this meant deleting two entries from
`platform_toolsets.cli`:

- `session_search` - a second door to the past: it searches the host transcript, bypasses the fold,
  and can read raw rows the host's own compression already discarded. That contradicts "the view is
  your memory, and its latest word on a thing is the truth".
- `memory` - the built-in memory tool. `memory_enabled` and `user_profile_enabled` are false in this
  profile, so it could only ever refuse; a dead tool is an invitation plus a wasted schema in the
  cached prefix.

Known and accepted: `terminal`, `file` and `code_execution` remain, and the log is a plain file
(`<profile>/optchat/main/*.jsonl`), so an agent with a shell can read or grep its own log. Closing
that means removing the shell, which would gut the body Hermes is kept for. It is accepted because it
takes a deliberate act rather than an offered, discoverable tool - and it is recorded here rather
than left as a footnote.

---

# Revision 10 - the runtime block moves; media rides the envelope (2026-10-11)

WP8. Two documented interfaces change, and registration/startup failures become loud.

## The system prompt carries no state

`select_context` now lifts the host's runtime block (date, session id, model, platform) out of the
system message and returns it verbatim as a `user` message **after the view and before this turn's
messages**, so the system text is byte-stable across sessions and the stable prefix a compaction
reads is stable. The block is the host's own marker-delimited render
(`RUNTIME_ENVIRONMENT_HEADING` / `RUNTIME_ENVIRONMENT_END`, read from `agent.prompt_builder`); only
that block moves and the rest of the system text is byte for byte. A host without the markers, or a
text without the block, is unchanged. The host's merge pass folds the relocated message into the
turn's first user row, exactly as it delivers its own per-turn notes.

Return shape: `[system] [assistant: <chat> view] [user: runtime block] [user: this turn's
messages]` when the block is present; `[system] [assistant: <chat> view] [turn]` otherwise.

## Media rides the host's multimodal envelope

`optchat_zoom(id, 1)` returns the host's multimodal tool-result envelope where the host recognises
it (`agent.tool_dispatch_helpers._is_multimodal_tool_result`):

```json
{"_multimodal": true,
 "content": [{"type": "text", "text": <today's JSON payload>},
             {"type": "image_url", "image_url": {"url": <data: URL or remote URL>}}],
 "text_summary": <today's JSON payload>}
```

The text part and the summary are today's JSON payload exactly, and each archived image reference is
resolved to an `image_url` part: a local reference becomes an inline data URL, an http(s) reference
is passed through (the provider owns remote-URL validation). Audio and file entries stay in the JSON;
their part shapes are not demonstrated for tool results. Where the host has no such predicate, and
on every path without images, the return is today's JSON string. `handle_tool_call` therefore
returns `str | dict`. Media still never enters the view or a compaction prompt.

## Loud failure

Every registration/startup failure path logs one clear error line naming the lost guarantee
("OptChat is not active: ... the host will ... compress it lossily ..."); `on_session_start` logs
once per engine and re-raises. The host's silent fallback to its built-in compressor is unchanged -
it is an upstream ask (docs/FINDINGS.md).

---

# Revision 11 - the three host seams are wired (2026-10-11)

WP9. The seams built in the seam worktree (`feat/context-engine-seams`, commit `cf1743d63a`) are
now used. Two interfaces change; one refusal is deliberately not taken.

## The turn's tools reach the compactor on every route

`select_context` accepts `tools=` - the host's deep copy of the request's outgoing schemas, `None`
when unavailable - and captures it with the same rule `cache_plan` uses (one helper,
`_capture_turn_tools`: a deep copy into `_turn_tools`, `_has_turn_tools` latched, and the
compactor's tools updated at once). On a route where no cache-plan hook fires this is the only
carrier, so a compaction now sends the same tools a turn sends, mark-based route or not. An
explicitly empty list is a capture, exactly as in `cache_plan`. `None` means unavailable and
changes nothing: the navigation schemas remain the fallback and the existing one-time warning is
unchanged.

## The engine does not refuse requests (`ContextSelectionBlocked` stays unused)

The seam lets an engine refuse a request so the host re-assembles once; the engine deliberately
never raises it. `select_context` already runs `compactor.settle(...)` before rendering - free
work only, `model_calls=False` - which is the most admission possible without starting a model
call on the turn path (a summarizer call already in flight cannot be interrupted by the
deadline). Refusing would make the host re-assemble once and then send its own original request -
the full transcript plus Hermes' own compression - which is strictly worse than serving a view
that still carries placeholders: the design's economics (a stable prefix a compaction can read)
survive placeholders, and they do not survive a transcript swap. A turn served with an unsettled
view logs one debug line, so the condition is visible when it happens and silent when it does not.

## The log is written as the turn unfolds (`on_message`)

`on_message` ingests each appended message immediately, with the same classification
`on_turn_complete` uses - one shared walk (`_ingest_rows`), so a row's kind never depends on which
hook ingested it. Ingestion starts at the checkpoint boundary, so rows the host appended without a
notification (the turn's user row, synthetic nudges) are ingested too, and the appended row is
located by equality against the clone the host hands over, never assumed to be the last row.
`on_turn_complete` stays the reconciler: it runs the same walk from the same boundary, so rows
already ingested are a no-op. The checkpoint still advances once per row, after every record of
that row is durable, so a restart resumes at the boundary and a crash mid-turn cannot skip a row.
A delegate result is resolved once (at ingestion or at selection, whichever comes first) and its
name is registered from that resolution.

---

# Revision 12 - refusal, non-writing foreign reads, and explicit view recovery (2026-10-11)

WP10 (PLAN-next Order 1). Four behaviours change: the log and the tree refuse semantic
corruption loudly, foreign readers stop writing, and a missing view state is an explicit
outcome instead of a silent rebuild.

## Refused ids and coordinates

A log record or tree node that parses but does not validate is refused with one
`RuntimeWarning` naming the file, byte offset and reason - never a silent skip.

- `Log` keeps the refused id in its index: `count()` counts id positions, `get` returns
  `None` for a refused id, `range` may contain `None`, and `append` never reuses the id.
  The gap rule still applies to ids never seen; a refused id is a known position, not a
  gap.
- `Tree` records a refused coordinate: `put` raises for it (it is never rewritten), and
  `Tree.refused(l, i)` reports it. A valid node whose saved children are missing is kept
  with a warning - its line was written, so it is not treated as absent - and its
  coordinate is never rewritten either.

## Non-writing foreign reads

`Tree(root, node_bytes, read_only=False)` mirrors `Log`'s read handle: its loader uses
`read_jsonl(..., seal=False)` (no tail sealing), and `put`/`save` raise. Every path that
reads a chat it does not own uses it - scoped integer navigation, string-name zoom, and a
child reading its master (at capture and at navigation). The owning chat keeps sealing.
`View.save` refuses to write through a read-only tree.

## Recovery coverage

`OptChatEngine._recover` extends the view to cover every id position at or beyond its end
whether or not the record's level-0 node already exists (the tree save may have succeeded
while the view save did not), and a refused id occupies its position, so the next append
lands on the id it expects.

## Missing view state

`View.load` never treats a missing `view.json` as a normal load. A chat with no records
builds the empty view silently. A chat that has records - or saved tree nodes, when the
caller cannot know the record count - warns that the view state is lost (and with it this
chat's cache entries), rebuilds the closest available view from the tree's saved nodes -
never from the log - and records `"recovered": true` in `view.json` so a later load can
see that the recovery happened. `View(tree, min_bytes, max_bytes, records=...)` takes the
owner's record count; the engine passes `log.count()`, and a foreign read passes nothing.
The version-2 object gains that boolean.

---

# Revision 13 - a duplicate name leaves the new report addressable (2026-10-11)

P6b, Plan Order 4. This supersedes revision 9's sentence that a conflicting name "keeps its first
mapping and warns once": the first mapping is still kept, but the new report no longer carries the
taken name.

- A delegate result whose name is already mapped to a different child receives the first free
  suffixed label (`"Name 2"`, `"Name 3"`, ..., incrementing until free), chosen by probing the
  persisted `<root>/agents/index.json`. That label is what the work record carries
  (`[Researcher 2] ...`) and what the index maps (`"Researcher 2" -> child`): the name the model
  reads in a record is the name it can zoom by (revision 3).
- The original name stays mapped to the first child as an alias, so older labels keep working; a
  mapping is never overwritten.
- A label already mapped to the same child is reused rather than replaced, so a child's label is
  stable across turns and restarts: it lives in the index, not in memory. The conflict still warns
  once per name, per engine.
- Registration happens when a result is resolved - at selection for the running turn, or at
  ingestion - before the record text is built, so the record and the index can never disagree about
  the label.

---

# Revision 14 - the durability protocol (2026-10-11)

Plan Order 2 (P1b, P1d). The recovery findings in this section are corrected and
superseded by revision 16 (2026-10-12). The log stays append-only; logged records
are never repaired by deletion or rewriting.

## The in-flight marker (P1b, per D4; corrected by revision 16)

`<chat_root>/ingest.json` records the committed record frontier separately from
an interrupted row's expected payload sequence:

```json
{"session": "...", "rows": 0, "fingerprints": [], "committed_count": 0,
 "inflight": {"fingerprint": "<payload fingerprint>", "count_before": 0,
              "records": [{"kind": "...", "digest": "..."}]},
 "lost": null, "failure": null}
```

The marker is published before the first record. Its `count_before` must equal
`committed_count`; a mismatch is an explicit recovery failure. The surviving
range must match the expected kinds and digests at every position, with no refused
position or impossible delta. The whole recomputed sequence must match even when
no complete record survived. A completed range is validated before clearing its
marker and advancing the committed frontier.

Expected kinds and digests are payload checks, **not ordinal provenance or
occurrence identity**. Equal payloads do not establish ownership. The earlier
claim that a content fingerprint identifies an occurrence, or that the marker by
itself prevents every duplicate or skipped occurrence, is withdrawn. See revision
16 for the remaining identity-dependent recovery limitation.

Append transactions serialize the ingestion hooks, including checkpoint loading,
recovery, append and checkpoint publication, with note imports through the same
`Log` instance. Reentrant imports fail explicitly; another thread waits. A
surviving intent also prevents external appends between hook calls or after a
restart. The process-lifetime file lock continues to exclude other owning logs.

## The lost ingestion boundary (P1b; corrected by revision 16)

A missing source boundary records a durable unresolved interval in `lost`,
preserving the old checkpoint and in-flight marker. It does not choose a newest
content match as an anchor. Later turns and restarts ingest nothing across this
interval, even if an old payload reappears. Recovery is unavailable without a
proven contiguous committed occurrence boundary and ordered host evidence. This
change replaces the earlier advance-and-clear behavior; it does not claim the
engine can manufacture the missing host evidence.

## Media failures are preserved, not swallowed (P1d)

A media decode/read/index failure records the failed part's **reference and failure** in
`<root>/assets/pending.json` (`message_id`, `part_index`, `type`, `reference`, the part itself when
JSON-safe, `error`, `attempts`). A retry pass (`optchat.media.retry_pending_media`, run at session
start) re-reads or re-decodes a local path or embedded payload and archives it on success, removing
the entry. A **remote URL keeps its reference only**: fetching remote bytes is a separate, open item,
so the reference survives and the archived-media promise remains open. The row checkpoint still
advances - the log record is the durable text, the pending reference is the durable media intent.

## Durable publications and dispatched reports

`View.save()` writes through the same durable replacement `atomic_json()` uses (fsync'd temporary,
atomic rename, directory sync), so a view publication is as durable as a checkpoint publication; the
media index is written the same way. A delegate result with `status="dispatched"` that carries a
nonempty `inline_results` list is ingested as the `work` records it carries (one per completed
entry); a dispatched payload with nothing completed inline still produces no records.

---

# Revision 15 - missing view state is an explicit unavailable outcome (2026-10-11)

Plan Order 1's P3b, redone after the challenge reversed D5 (`docs/DECISIONS.md`, revision 2).
This supersedes revision 12's "Missing view state" section: the rebuild from the tree's saved
nodes is removed. It was reconstruction - it did not restore the previous pair selection, the
batch state, or the retained compaction contexts - and AGENTS.md forbids rebuilding the view.

## What recovery is, and what it is not

The view is recovered **only by extension, and only when its state exists**: `_recover` extends
coverage for every id position at or beyond the view's end (revision 12, unchanged). Exact-state
recovery of a **missing** `view.json` is impossible from the surviving artifacts: the pair
selection and the batch state depend on when batches ran against built nodes, which is never
recorded, and the retained compaction contexts exist only in `view.json`. A derived pair list
would be a different state wearing this chat's name, so the code does not attempt one - recorded
here as a decision, not an omission.

## The unavailable-state outcome

A missing `view.json` is handled as follows:

- A chat with **no records and no saved tree nodes** builds the empty view silently; an owning
  load saves it, a foreign (read-only) load does not write.
- Otherwise - records exist, or saved tree nodes exist when the caller cannot know the record
  count - `View` raises `ViewStateUnavailable`, naming the chat root and the consequence.
  **Nothing is written**: no pairs, no marker, no `view.json` is created; the state stays
  exactly as found, so restoring `view.json` from a backup resumes the chat.
- The engine surfaces the condition through its start path: one loud error line for the
  session, after which it **steps aside** - `select_context` returns `None` (the host keeps
  its own request), the observation hooks no-op, the navigation tools answer with an explicit
  "not active" error, and a clone carries the same state - instead of retrying the failed
  start on every call.
- Host behaviour is separate and is not a guarantee: the installed host catches
  `on_session_start` exceptions at debug level (`agent/agent_init.py`) and fails open on
  selection exceptions, sending the unmodified request (`agent/conversation_loop.py`). The
  engine does not claim a safe refusal; its loud line in the profile log is the signal, and
  fail-closed behaviour remains an upstream gap (`docs/FINDINGS.md`).
- A child's session-start capture of the master's view no longer silently skips a missing
  master `view.json` (which let the child serve its own empty view as the master's): it raises
  the same explicit error. Scoped navigation keeps its tool-level error.

## The `recovered` marker is retired

`view.json` no longer carries a `recovered` key, and the loader ignores the key in a legacy
file (only the retired rebuild path ever wrote it). Otherwise the version-2 schema is unchanged.


---

# Revision 16 - engine recovery hardening (2026-10-12)

This dated amendment supersedes revision 14's recovery findings, following D4 in
`docs/DECISIONS.md` revision 3 and sections 4 and 5 of the second challenge.

- **Independent start binding:** `committed_count` is published at the clean
  boundary before the row intent, and after a validated completed row. It is not
  inferred from the surviving marker's `count_before` during recovery. A missing
  or invalid committed frontier is unavailable evidence. Failure recording keeps
  that missing value missing rather than silently inventing a new frontier.
- **Validation:** a surviving marker's entire recomputed sequence is checked at
  delta zero as well as positive deltas, before replacement or append. Every
  completed range is checked before the marker is cleared, including uninterrupted
  ingestion. Payload validation does not prove durable host occurrence identity.
- **Unresolved interval:** `lost` preserves the original row frontier, committed
  record count, observed projection length, and unavailable reason. The old
  in-flight marker remains intact. Subsequent turns do not refresh an anchor or
  clear this state. There is no automatic recovery without ordered host evidence.
- **Checkpoint evidence:** only no checkpoint plus no records means a fresh
  stream. Unreadable JSON, malformed outer fields, missing checkpoints on an
  established log, and session mismatches stop ingestion. A durable
  `ingest-unavailable.json` sidecar records the unavailable reason, requested
  session and observed record count; the original checkpoint bytes remain intact.
  The sidecar is sticky across turns and restarts, including restoration of an
  apparently valid checkpoint. Resolving it requires external evidence and an
  operator action; no repair/reset API is introduced here.
- **Session transitions:** a new session on an existing stream is unavailable
  recovery, not permission to replay from zero or discard the old frontier. The
  transition is recorded on the next ingestion hook. Starting a new stream in a
  distinct root remains possible. No automatic stream migration is implemented.
- **Append exclusion:** a per-`Log` reentrant mutex covers each complete ingestion
  hook, including recovery. `import_notes` uses the same transaction scope for its
  whole import and derived-state publication. Nested transactions fail before
  append; competing threads wait. All `Log.append` calls acquire the mutex, and
  external append paths refuse unresolved durable ingestion state. This excludes
  supported note interleaves; it does not retroactively prove ownership of legacy
  records or supply host occurrence provenance.
- **Day-file publication:** a newly created day file is file-synced, then `main/`
  is directory-synced before append succeeds. Creating `main/` also syncs its
  parent. An append I/O failure poisons that open log for further writes; reopening
  reindexes the surviving bytes before any id can be reused. This is tested by
  fault injection, not a physical power-loss experiment.

**Remaining requirement / test conflict:** the legacy source-boundary search still
uses content fingerprints. It cannot distinguish distinct equal-content
occurrences, and must not be represented as satisfying D4's host identity or
ordered-frontier requirements. Removing that fallback requires changing existing
UID-less restart and changed-tail tests beyond the single lost-boundary test
allowed by this task. That scope decision is pending; this amendment records the
engine-side hardening without claiming full D4 compliance or P2a readiness.

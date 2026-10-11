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

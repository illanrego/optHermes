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
    # one writer, process-lifetime lock; torn lines skipped at load

# tree.py
Line = dataclass(l: int, i: int, text: str, size: int)
class Tree:
    def put(self, l: int, i: int, text: str) -> None    # build once
    def get(self, l: int, i: int) -> Line | None
    def buildable(self) -> list[tuple[int, int]]        # ready nodes, queue order
    def needs_summary(self, l: int, i: int) -> bool     # size > NODE_BYTES

# view.py
def due(l: int, i: int, T: int) -> float                # (T - last) / 2**l
class View:
    def merge_batch(self, T: int) -> None               # down to VIEW_MIN_BYTES
    def lines(self) -> list[Line]
    def render(self) -> str                             # "<chat>...</chat>"
    def save(self) -> None
    def load(self) -> None

# compactor.py
class Compactor:
    def submit(self, l: int, i: int) -> None
    def run_pending(self, limit: int = COMPACT_CONCURRENCY) -> int
    def summarize(self, l: int, i: int) -> str          # ruler, retry<=5, keep shortest
```

### `select_context` (the crux)

Returns, for this request only:

```
[system prompt] [<chat> view] [messages of THIS turn only]
```

"THIS turn" = everything from the last user message onward. Every prior turn is
dropped; the view stands in. Returns `None` when there is nothing to change.
Never mutates `conversation_messages`.

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

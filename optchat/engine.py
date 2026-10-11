"""Hermes ContextEngine adapter for the durable OptChat log and view.

Host messages map to log kinds as follows: system is host-owned and omitted;
user becomes user; text-only assistant becomes unii; assistant with tool_calls
becomes tool (name and JSON arguments, including any accompanying content);
delegate_task results become work, and other tool results become echo.
Thinking fields are never logged.

``select_context`` receives the request's outgoing tools (the host seam) and
captures them exactly as ``cache_plan`` does, so a compaction sends the same
tool schemas a turn sends on every route, mark-based or not; ``None`` means the
host has none to give, and the navigation schemas remain the fallback with a
warning once per engine.

``on_message`` ingests each appended message as the turn unfolds, with the same
classification ``on_turn_complete`` uses; the ingestion checkpoint keeps the
turn-boundary pass a reconciler, never a duplicate. The engine deliberately
does not raise ``ContextSelectionBlocked``: selection settles for free before
rendering, and refusing would make the host send its own original request -
full transcript plus Hermes' own compression - which is strictly worse than a
view that still carries placeholders (CONTRACT.md revision 11). A turn served
with an unsettled view logs one debug line; a settled one is silent.

The retained request tail starts at the host's ``incoming_message`` when that
message is present in the request (identity first, then equality). Otherwise it
starts at the last user row. A later steering user row stays in the same turn.

Navigation is scope-explicit. An integer id without a scope resolves against the
log whose view the caller is reading - the master's, for the master and for a
child rendering the master's view; a child's own chat is reached only with an
explicit scope. ``scope="Name"`` resolves against that child's chat, and an
unknown or unmapped scope is an error, never a silent fallback. A delegate
result's name-to-child mapping is registered as soon as the result is visible to
a selection of the running turn, not only at ``on_turn_complete``. A name already
mapped to a different child keeps its first mapping; the conflicting child
instead receives the first free suffixed label (``"Name 2"``, incrementing until
free), stored in the index and used in its work record, so the name the model
reads in a record is the name it can zoom by. A label already mapped to the same
child is reused, so labels stay stable across turns and restarts.

The host's runtime block (date, session id, model, platform) is moved out of the
system text and re-delivered verbatim on the user channel after the view, so the
system text stays byte-stable across sessions; the host's own markers define the
block, and a host without them is left untouched.

``optchat_zoom(id, 1)`` returns its media in the host's multimodal envelope where
the host recognises one, resolving archived image references to inline data URLs;
a host without the envelope keeps today's JSON string. Media never enters the view
or a compaction prompt either way.

The host ABC is imported here, rather than by the package entry point. When
Hermes is absent, a local base permits standalone unit tests; an engine loaded
inside Hermes inherits the real ABC. This binding is fixed at module import.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

try:
    from agent.context_engine import ContextEngine
except ModuleNotFoundError as exc:
    if exc.name != "agent":
        raise

    class ContextEngine:  # type: ignore[no-redef]
        """Standalone test base when the Hermes package is unavailable."""

from .compactor import CompactionRequest, Compactor, Summarizer
from .cache import plan_view_cache
from .media import MEDIA_TYPES, archive_media, has_media_parts
from .persistence import atomic_json
from .prompts import SYSTEM_PROMPT
from .store import Log
from .text import split_for_kind
from .tree import Tree
from .view import PLACEHOLDER, View
from . import tools as opt_tools


logger = logging.getLogger(__name__)


def default_root() -> Path:
    """Fallback chat directory when no constructor root or host home is supplied."""
    env = os.environ.get("HERMES_HOME")
    return (Path(env) if env else Path.home() / ".hermes") / "optchat"


# How many recent host rows the ingestion checkpoint remembers. More than one,
# because the host may rewrite the head of a transcript (compression) while the
# recent rows stay byte-identical - the boundary is then still findable.
CHECKPOINT_ROWS = 8


def _row_fingerprint(message: dict[str, Any]) -> str:
    """A stable identity for one host row: what it is, not when it arrived."""
    stable = {key: message.get(key) for key in ("role", "content", "tool_calls", "tool_call_id")}
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_config(root: Path) -> dict[str, Any]:
    """The engine's size knobs, from ``<root>/config.json`` when it exists.

    A missing, unreadable or malformed file is ignored rather than fatal: the engine must
    still start with its defaults.
    """
    try:
        data = json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _missing_summarizer(_request: CompactionRequest) -> str:
    raise RuntimeError("No OptChat summarizer has been configured")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict)
                         and (not isinstance(part.get("type"), str)
                              or part["type"] not in MEDIA_TYPES)
                         and isinstance(part.get("text"), str))
    return "" if content is None else str(content)


def _runtime_markers() -> tuple[str, str] | None:
    """The host's own runtime-block delimiters, or ``None`` when it defines none.

    The host renders the block (date, session id, model, platform) between these
    markers at the end of its system prompt. They are read from the host itself
    (``agent.prompt_builder``), never re-derived here; a host without them gets
    today's behaviour, and nothing moves.
    """
    try:
        from agent.prompt_builder import RUNTIME_ENVIRONMENT_END, RUNTIME_ENVIRONMENT_HEADING
    except Exception:
        return None
    if not RUNTIME_ENVIRONMENT_HEADING or not RUNTIME_ENVIRONMENT_END:
        return None
    return RUNTIME_ENVIRONMENT_HEADING, RUNTIME_ENVIRONMENT_END


def _split_runtime_block(content: str) -> tuple[str, str | None]:
    """Lift the host's runtime block out of a system text: ``(stripped, block)``.

    The block is ``{heading}\\n\\n{hints}\\n\\n{end}``, the host renderer's own
    shape; only the marker-delimited block moves and the rest is returned byte
    for byte. ``(content, None)`` when the host defines no markers or the text
    carries no such block.
    """
    markers = _runtime_markers()
    if not content or markers is None:
        return content, None
    heading, end = markers
    start = content.rfind(f"\n\n{heading}\n\n")
    if start < 0:
        return content, None
    stop = content.find(f"\n\n{end}", start)
    if stop < 0:
        return content, None
    stop += 2 + len(end)
    return content[:start] + content[stop:], content[start + 2:stop]


def _multimodal_envelope_supported() -> bool:
    """Whether this host recognises the ``_multimodal`` tool-result envelope.

    The host's own predicate (``agent.tool_dispatch_helpers._is_multimodal_tool_result``)
    is the marker: where it exists, a tool result carrying ``_multimodal=True`` and
    a content list is turned into real multimodal content (persisted; downgraded to
    its text summary for text-only models). Where it does not, tools keep returning
    plain JSON strings.
    """
    try:
        from agent.tool_dispatch_helpers import _is_multimodal_tool_result
    except Exception:
        return False
    return callable(_is_multimodal_tool_result)


def _delegate_name(text: str) -> str:
    """Use a named JSON result when available, otherwise label the child generically.

    Host delegate results have no fixed shape. Only nonempty string values in
    name, agent, or task are useful labels; all other shapes use ``subagent``.
    """
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return "subagent"
    if isinstance(result, dict):
        for key in ("name", "agent", "task"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "subagent"


class OptChatEngine(ContextEngine):
    def __init__(self, root: Path | None = None, *, view_min_bytes: int | None = None,
                 view_max_bytes: int | None = None, node_bytes: int | None = None,
                 summarize: Summarizer | None = None):
        self.root = Path(root) if root is not None else None
        self._size_overrides = (view_min_bytes, view_max_bytes, node_bytes)
        config = _load_config(self.root) if self.root is not None else {}
        # The profile's config.json supplies the sizes; an explicit argument wins over it.
        self.view_min_bytes = (view_min_bytes if view_min_bytes is not None
                               else config.get("view_min_bytes", 64_000))
        self.view_max_bytes = (view_max_bytes if view_max_bytes is not None
                               else config.get("view_max_bytes", 128_000))
        self.node_bytes = node_bytes if node_bytes is not None else config.get("node_bytes", 512)
        self.summarize = summarize
        self.log: Log | None = None
        self.tree: Tree | None = None
        self.view: View | None = None
        self.compactor: Compactor | None = None
        self._master_view: View | None = None
        self._master_log: Log | None = None
        self._master_tree: Tree | None = None
        self._chat_root = self.root
        self._turn_system = ""
        self._turn_tools: list[dict[str, Any]] = []
        self._has_turn_tools = False
        self._warned_missing_tools = False
        self._turn_model: str | None = None
        self._warned_route_override = False
        self._session_id = ""
        self._warned_ingest_boundary = False
        self._warned_child_conflicts: set[str] = set()
        self._start_failure_logged = False
        # Delegate results already resolved, keyed by row fingerprint - by a
        # selection of the running turn or by ingestion itself: ``_arrived_rows``
        # guards against resolving a row twice, ``_arrived_records`` carries a
        # selection's resolution to ingestion (and is popped there).
        self._arrived_rows: set[str] = set()
        self._arrived_records: dict[str, list[tuple[str, str, str | None]]] = {}
        # Cron reads only; a child writes to its own chat after session start.
        self._writes = False
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0

    @property
    def name(self) -> str:
        return "optchat"

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        try:
            self._start_session(session_id, **kwargs)
        except Exception as exc:
            # The host catches this and logs at debug level, so the failure must be
            # loud here: the session then runs without OptChat and its guarantee.
            if not self._start_failure_logged:
                self._start_failure_logged = True
                logger.error(
                    "OptChat failed to start for session %s: %s. OptChat is not active "
                    "for this session: no view is served and nothing is folded, and the "
                    "no-loss guarantee is not in effect.", session_id, exc)
            raise

    def _start_session(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        if "model" in kwargs:
            self._turn_model = kwargs["model"]
        self._check_compaction_route()
        if self.root is None:
            home = kwargs.get("hermes_home")
            self.root = Path(home) / "optchat" if home is not None else default_root()
            self._chat_root = self.root
            config = _load_config(self.root)
            view_min, view_max, node = self._size_overrides
            self.view_min_bytes = view_min if view_min is not None else config.get("view_min_bytes", 64_000)
            self.view_max_bytes = view_max if view_max is not None else config.get("view_max_bytes", 128_000)
            self.node_bytes = node if node is not None else config.get("node_bytes", 512)
        platform = kwargs.get("platform") or "cli"
        self._writes = platform != "cron"
        if platform == "subagent":
            if not session_id or Path(session_id).name != session_id or session_id in (".", ".."):
                raise ValueError("Subagent session id must be a safe path component")
            # A child renders the master's view, so the ids in front of it are the
            # master's: keep the master's view, log and tree before the child gets
            # its own. The split happens once; a later call for the same child must
            # not re-capture the child's own objects as the master's.
            if self._master_view is None and self._chat_root == self.root:
                self._master_view = self.view
                if self._master_view is None and (self.root / "view.json").is_file():
                    self._master_view = View(Tree(self.root, self.node_bytes),
                                             self.view_min_bytes, self.view_max_bytes)
            if self._master_log is None and self._chat_root == self.root and self.log is not None:
                self._master_log = self.log
                self._master_tree = self.tree
            child_root = self.root / "agents" / session_id
            if self._chat_root == child_root and self.log is not None:
                return
            self.log = self.tree = self.view = self.compactor = None
            self._chat_root = child_root
        elif self.log is not None:
            return
        self.log = Log(self._chat_root)
        try:
            self.tree = Tree(self._chat_root, self.node_bytes)
            self.view = View(self.tree, self.view_min_bytes, self.view_max_bytes)
            self.compactor = Compactor(self.tree, self.summarize or _missing_summarizer,
                                       node_bytes=self.node_bytes, view=self.view)
            if not self._has_turn_tools:
                self._turn_tools = self.get_tool_schemas()
            self.compactor.system = self._turn_system
            self.compactor.tools = self._turn_tools
            self._recover()
        except BaseException:
            self.log.close()
            self.log = None
            raise

    def _recover(self) -> None:
        """Reconcile the derived state with the log after a restart or a crash.

        The log is the source of truth. Every record needs a level-0 node - its text
        is in the log, so a long source is never stranded in memory - and the view
        must cover every record. The view is extended, never rebuilt: a rebuilt view
        renders differently and kills every cache entry (spec section 3).
        """
        log, tree, view, compactor = self.log, self.tree, self.view, self.compactor
        end = view._end()
        recovered = 0
        for i in range(log.count()):
            if tree.get(0, i) is not None:
                continue
            record = log.get(i)
            if record is None:
                continue
            compactor.seed(i, record.text, record.kind)
            recovered += 1
            if i == end:
                view.add_root(i)
                end += 1
        if recovered:
            # Free work only: starting a session must never wait on a model call.
            compactor.run_pending(limit=8, model_calls=False)
            tree.save()
            view.save()

    def _checkpoint_path(self) -> Path:
        return self._chat_root / "ingest.json"

    def _load_checkpoint(self) -> dict[str, Any] | None:
        """The stored ingestion boundary for this chat, or ``None``."""
        try:
            raw = json.loads(self._checkpoint_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("session") != self._session_id:
            return None
        rows, fingerprints = raw.get("rows"), raw.get("fingerprints")
        if (not isinstance(rows, int) or rows < 0 or not isinstance(fingerprints, list)
                or not all(isinstance(item, str) for item in fingerprints)):
            return None
        return {"rows": rows, "fingerprints": fingerprints}

    def _resume_row(self, messages: list[dict[str, Any]],
                    checkpoint: dict[str, Any] | None) -> int | None:
        """The row to resume at, or ``None`` when the boundary cannot be found."""
        if checkpoint is None:
            return 0
        known = set(checkpoint["fingerprints"])
        if not known:
            return checkpoint["rows"] if checkpoint["rows"] <= len(messages) else None
        for row in range(min(len(messages), checkpoint["rows"]) - 1, -1, -1):
            if _row_fingerprint(messages[row]) in known:
                return row + 1
        return None

    def _save_checkpoint(self, rows: int, recent: list[str]) -> None:
        atomic_json(self._checkpoint_path(),
                    {"session": self._session_id, "rows": rows,
                     "fingerprints": recent[-CHECKPOINT_ROWS:]})

    def _check_compaction_route(self) -> None:
        """Report an explicit aux route without changing the host's routing."""
        if self._warned_route_override:
            return
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return  # Standalone use and older hosts have no config reader.
        auxiliary = config.get("auxiliary") if isinstance(config, dict) else None
        route = auxiliary.get("optchat") if isinstance(auxiliary, dict) else None
        if not isinstance(route, dict):
            return
        overrides = {key: route[key] for key in ("provider", "model")
                     if route.get(key) not in (None, "", "auto")}
        if overrides:
            self._warned_route_override = True
            warnings.warn(
                f"auxiliary.optchat explicitly overrides {overrides!r}; compactions "
                f"may not share the turns' cache entry (turn model: {self._turn_model!r}). "
                "Use provider/model auto or unset to inherit the turn's route.",
                RuntimeWarning, stacklevel=2)

    def _check_compaction_tools(self) -> None:
        if not self._has_turn_tools and not self._warned_missing_tools:
            self._warned_missing_tools = True
            warnings.warn(
                "OptChat has never captured the turn's outgoing tools: the host's "
                "cache-policy gates cache_plan, the only hook that carries tools. "
                "Compactions retain the navigation fallback schemas and cannot match "
                "the turn's tool prefix. A host seam must pass tools to select_context.",
                RuntimeWarning, stacklevel=2)

    def _open(self) -> tuple[Log, Tree, View, Compactor]:
        if self.log is None:
            self.on_session_start("")
        assert self.log is not None and self.tree is not None
        assert self.view is not None and self.compactor is not None
        return self.log, self.tree, self.view, self.compactor

    def select_context(self, request_messages: list[dict[str, Any]], *,
                       conversation_messages: list[dict[str, Any]] | None = None,
                       incoming_message: dict[str, Any] | None = None,
                       budget_tokens: int = 0,
                       tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]] | None:
        """Select the request's context for this call (see CONTRACT.md).

        ``tools`` is the request's outgoing schemas as the host hands them over
        (a deep copy; ``None`` when unavailable). They are captured exactly as
        ``cache_plan`` captures them, so a compaction sends the same tools a
        turn sends on any route - including routes where no cache-plan hook
        fires. Nothing is invented for ``None``: the navigation fallback and its
        one-time warning stay in force.
        """
        if tools is not None:
            self._capture_turn_tools(tools)
        if not request_messages:
            return None
        turn_start = None
        if incoming_message is not None:
            turn_start = next((i for i, message in enumerate(request_messages)
                               if message is incoming_message), None)
            if turn_start is None:
                turn_start = next((i for i, message in enumerate(request_messages)
                                   if message == incoming_message), None)
        if turn_start is None:
            turn_start = next((i for i in range(len(request_messages) - 1, -1, -1)
                               if request_messages[i].get("role") == "user"), None)
        if turn_start is None:
            return None
        system = [copy.deepcopy(message) for message in request_messages
                  if message.get("role") == "system"]
        runtime_block: str | None = None
        if not system:
            system = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            original = _content_text(system[0].get("content"))
            # The host's runtime block (date, session id, model, platform) is state,
            # and state never belongs in the system prompt: the system text must stay
            # byte-stable across sessions for the stable prefix to be stable. Move
            # the block verbatim to the user channel after the view; everything else
            # in the system text stays byte for byte.
            content, runtime_block = _split_runtime_block(original)
            if SYSTEM_PROMPT not in content:
                content = content + "\n\n" + SYSTEM_PROMPT
            if content != original:
                system[0]["content"] = content
        turn = [copy.deepcopy(message) for message in request_messages[turn_start:]]
        _, tree, view, compactor = self._open()
        # A delegate result is visible in the running turn's request before the
        # turn completes; registering it here makes the report zoomable by name
        # in the same turn.
        self._register_arrived_delegates(request_messages, turn_start)
        self._turn_system = _content_text(system[0].get("content"))
        compactor.system = self._turn_system
        compactor.tools = self._turn_tools
        built = 0
        try:
            max_nodes, timeout_s = 64, 20.0
            config_path = self.root / "config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                max_nodes = config.get("settle_max_nodes", max_nodes)
                timeout_s = config.get("settle_timeout_s", timeout_s)
            # model_calls stays False: this runs synchronously inside the host's request,
            # and a summarizer call cannot be interrupted by the deadline. Anything that
            # needs a model is built by the background run_pending after the turn.
            built = compactor.settle(max_nodes=max_nodes, timeout_s=timeout_s)
            tree.save()
            view.save()
        except Exception:
            logger.exception("OptChat settle failed; serving the current view")
        logger.debug("OptChat settle built %d nodes; view settled: %s", built, view.settled())
        served = self._master_view or view
        if not served.settled():
            # The honest gap: a long message's node still needs a model, and the
            # turn path must never start one. Visible when it happens, silent when not.
            logger.debug(
                "OptChat serves an unsettled view for this turn: %d line(s) still hold "
                "placeholders; a long message awaits its background summary",
                sum(1 for line in served.lines() if line.text == PLACEHOLDER),
            )
        # The runtime block travels where the design puts per-turn state: after the
        # view and before this turn's messages (the host merges it into the turn's
        # first user row, exactly as it delivers its own per-turn notes).
        runtime = ([{"role": "user", "content": runtime_block}]
                   if runtime_block is not None else [])
        return system + [{"role": "assistant", "content": (
            self._master_view or view).render()}] + runtime + turn

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        integer = {"type": "integer"}
        zoom_scope = {"type": "string",
                      "description": "A subagent's name: resolve the id against that "
                                     "child's chat instead of the chat whose view you "
                                     "are reading."}
        date_scope = {"type": "string",
                      "description": "A subagent's name: read the id from that child's "
                                     "chat instead of the chat whose view you are reading."}
        return [
            {"type": "function", "function": {"name": "optchat_zoom",
                "description": "Open a child chat by name, a chat span, or a raw message. "
                               "Without scope, an integer id resolves against the chat "
                               "whose view you are reading; a returned child chat names "
                               "the scope its ids resolve with.",
                "parameters": {"type": "object", "properties": {
                    "id": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "n": integer, "scope": zoom_scope},
                               "required": ["id"]}}},
            {"type": "function", "function": {"name": "optchat_date",
                "description": "Get the stored ISO timestamp of a raw message, from the "
                               "chat whose view you are reading or from the subagent's "
                               "chat named by scope.",
                "parameters": {"type": "object", "properties": {"id": integer, "scope": date_scope},
                               "required": ["id"]}}},
        ]

    def _reading_scope(self) -> tuple[Log, Tree, View | None]:
        """The chat whose view this caller is reading.

        The master reads its own view. A child renders the master's view, so the
        ids in front of it are the master's, not its own chat's; the child's own
        chat is reached only with an explicit scope.
        """
        if self._chat_root == self.root:
            log, tree, view, _ = self._open()
            return log, tree, view
        if self._master_log is None:
            self._master_log = Log(self.root, read_only=True)
        if self._master_tree is None:
            self._master_tree = (self._master_view.tree if self._master_view is not None
                                 else Tree(self.root, self.node_bytes))
        return self._master_log, self._master_tree, self._master_view

    def _child_scope(self, scope: str) -> tuple[Log, Tree] | str:
        """Open a named child's chat for scoped navigation, or a JSON error payload.

        The name is resolved through ``<root>/agents/index.json``; an unmapped or
        unsafe name, and a mapped name with no chat, are errors - navigation never
        falls back to another namespace. The log opens read-only because the
        child's own engine may still hold its lock.
        """
        child_root = opt_tools.child_chat_root(self.root, scope)
        if child_root is None:
            return json.dumps({"error": f"unknown scope: {scope!r}"})
        if not (child_root / "view.json").is_file():
            return json.dumps({"error": f"scope {scope!r} has no chat"})
        try:
            return Log(child_root, read_only=True), Tree(child_root, self.node_bytes)
        except (OSError, ValueError) as exc:
            return json.dumps({"error": f"scope {scope!r} cannot be read: {exc}"})

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if not isinstance(args, dict):
            return json.dumps({"error": "tool arguments must be an object"})
        if name not in ("optchat_zoom", "optchat_date"):
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        self._open()
        scope = args.get("scope")
        if scope is not None and (not isinstance(scope, str) or not scope):
            return json.dumps({"error": "scope must be a nonempty subagent name"})
        if name == "optchat_zoom":
            # The host's envelope is only used where the host itself recognises it;
            # everywhere else the tool keeps returning today's JSON string.
            envelope = _multimodal_envelope_supported()
            if scope is not None:
                if isinstance(args.get("id"), str):
                    return json.dumps({"error": "a string id opens its own chat; "
                                                "do not pass scope with it"})
                child = self._child_scope(scope)
                if isinstance(child, str):
                    return child
                child_log, child_tree = child
                return opt_tools.zoom(child_log, child_tree, None, args.get("id"),
                                      args.get("n", 1), scope=scope, envelope=envelope)
            log, tree, view = self._reading_scope()
            return opt_tools.zoom(log, tree, view, args.get("id"), args.get("n", 1),
                                  master_root=self.root, envelope=envelope)
        if name == "optchat_date":
            if scope is not None:
                child = self._child_scope(scope)
                if isinstance(child, str):
                    return child
                return opt_tools.date(child[0], args.get("id"), scope=scope)
            log, _tree, _view = self._reading_scope()
            return opt_tools.date(log, args.get("id"))

    @staticmethod
    def _records(message: dict[str, Any], node_bytes: int,
                 tool_name: str | None = None) -> list[tuple[str, str]]:
        role = message.get("role")
        if role == "user":
            return [("user", chunk) for chunk in split_for_kind(
                _content_text(message.get("content")), node_bytes, "user")]
        if role == "tool":
            text = _content_text(message.get("content"))
            name = _delegate_name(text) if tool_name == "delegate_task" else None
            if len(text) > 30_000:
                text = text[:15_000] + text[-15_000:]
            if name is not None:
                return [("work", f"[{name}] {text}")]
            return [("echo", text)]
        if role == "assistant" and message.get("tool_calls"):
            records = []
            for call in message["tool_calls"]:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                payload = {"name": function.get("name"),
                           "arguments": arguments}
                content = _content_text(message.get("content"))
                if content:
                    payload["content"] = content
                records.append(("tool", json.dumps(payload, ensure_ascii=False)))
            return records
        if role == "assistant":
            content = _content_text(message.get("content"))
            return ([("unii", chunk) for chunk in split_for_kind(content, node_bytes, "unii")]
                    if content or has_media_parts(message.get("content")) else [])
        return []

    @staticmethod
    def _message_records(messages: list[dict[str, Any]], node_bytes: int,
                         start: int = 0) -> list[tuple[str, str]]:
        """Classify rows using the call ids seen earlier in the transcript."""
        tool_names: dict[str, str] = {}
        records: list[tuple[str, str]] = []
        for row, message in enumerate(messages):
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if (isinstance(call.get("id"), str) and isinstance(function, dict)
                            and isinstance(function.get("name"), str)):
                        tool_names[call["id"]] = function["name"]
            call_id = message.get("tool_call_id") if message.get("role") == "tool" else None
            tool_name = tool_names.pop(call_id, None) if isinstance(call_id, str) else None
            if row >= start:
                records.extend(OptChatEngine._records(message, node_bytes, tool_name))
        return records

    def _child_index(self) -> dict[str, str]:
        path = self.root / "agents" / "index.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in data.items()):
            raise ValueError("Invalid subagent index")
        return data

    def _save_child_index(self, mapping: dict[str, str]) -> None:
        path = self.root / "agents" / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("index.json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(mapping, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _register_child(self, name: str, key: str) -> str:
        """Record ``name -> key`` and return the durable label for ``key``.

        The first mapping wins: a name already mapped to a different child is never
        repointed, so older labels keep working. The conflicting child instead
        receives the first free suffixed label (``"Name 2"``, ``"Name 3"``, ...),
        probed in the persisted index so the label stays stable across turns and
        restarts; a label already mapped to this child is reused, never replaced.
        Re-registering the same pair is idempotent.
        """
        index = self._child_index()
        existing = index.get(name)
        if existing == key:
            return name
        if existing is None:
            index[name] = key
            self._save_child_index(index)
            return name
        # The name is taken by another child. Never repoint it; give this child the
        # first label it already owns, else the first free one.
        suffix = 2
        while True:
            label = f"{name} {suffix}"
            mapped = index.get(label)
            if mapped == key:
                break
            if mapped is None:
                index[label] = key
                self._save_child_index(index)
                break
            suffix += 1
        if name not in self._warned_child_conflicts:
            self._warned_child_conflicts.add(name)
            warnings.warn(
                f"OptChat: subagent name {name!r} already maps to child {existing!r}; "
                f"the conflicting report is labeled {label!r} so its own chat stays "
                f"addressable.",
                RuntimeWarning, stacklevel=2)
        return label

    def _register_arrived_delegates(self, messages: list[dict[str, Any]], start: int) -> None:
        """Register ``label -> child`` for delegate results already visible this turn.

        The host shows the running turn's rows to ``select_context`` as they arrive,
        so a report can be zoomed by name in the turn it returns, not only after
        ``on_turn_complete`` ingests it. Each result row is resolved once (by row
        fingerprint) - ``_delegate_work`` claims each child's durable label in the
        index as it resolves it - and the resolution is kept for ingestion, so the
        logged ``[Label]`` always matches the registered mapping. Rows before the
        turn boundary are left to ingestion.
        """
        if not self._writes or self._chat_root != self.root:
            return
        tool_names: dict[str, str] = {}
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if (isinstance(call.get("id"), str) and isinstance(function, dict)
                        and isinstance(function.get("name"), str)):
                    tool_names[call["id"]] = function["name"]
        try:
            for message in messages[start:]:
                if message.get("role") != "tool":
                    continue
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or tool_names.get(call_id) != "delegate_task":
                    continue
                fingerprint = _row_fingerprint(message)
                if fingerprint in self._arrived_rows:
                    continue
                self._arrived_rows.add(fingerprint)
                known_keys = set(self._child_index().values())
                resolved = self._delegate_work(_content_text(message.get("content")), known_keys)
                if not resolved:
                    continue
                self._arrived_records[fingerprint] = resolved
        except (OSError, ValueError) as exc:
            logger.warning("OptChat could not register an arrived subagent report: %s", exc)

    def _delegate_work(self, text: str, known_keys: set[str]) -> list[tuple[str, str, str | None]]:
        """Return (label, record text, child key) for a completed delegate result.

        Each child is registered under its label here, before the record text is
        built: a name already mapped to a different child leaves that mapping
        untouched and this child claims the first free suffixed label, so the name
        the model reads in the record is the name it can zoom by.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("status") == "dispatched":
            return []
        entries = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            entries = payload.get("inline_results") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            entries = [payload] if isinstance(payload, dict) else [None]
        work = []
        for entry in entries:
            candidate = entry if isinstance(entry, dict) else {}
            key = candidate.get("child_session_id") or candidate.get("session_id")
            if not isinstance(key, str) or Path(key).name != key or key in ("", ".", ".."):
                key = None
            if key is None:
                # Stock Hermes omits the child's session id. A lone unclaimed chat is
                # unambiguous; multiple concurrent children need the host to emit it.
                agents = self.root / "agents"
                available = [path.name for path in agents.iterdir() if path.is_dir()
                             and path.name not in known_keys] if agents.is_dir() else []
                key = available[0] if len(available) == 1 else None
            if key is not None:
                known_keys.add(key)
            name = next((candidate[field].strip() for field in ("name", "agent", "task")
                         if isinstance(candidate.get(field), str) and candidate[field].strip()),
                        key or _delegate_name(text))
            body = json.dumps(entry, ensure_ascii=False) if entry is not None and len(entries) > 1 else text
            label = self._register_child(name, key) if key is not None else name
            work.append((label, f"[{label}] {body}", key))
        return work

    def _ingest_rows(self, messages: list[dict[str, Any]], start: int, end: int) -> None:
        """Append the log records for host rows ``[start, end)``.

        The classification is the one ``on_turn_complete`` has always used (call
        ids are tracked while walking, so a tool result is attributed to the call
        it answers); ``on_message`` shares it, so a row's kind never depends on
        which hook ingested it. The checkpoint advances once per row, after every
        record of that row is durable, so a crash re-ingests at most the row it
        was on - never a skipped or duplicated row.
        """
        log, tree, view, compactor = self._open()
        chat_root = self._chat_root or self.root
        assert chat_root is not None
        recent = [_row_fingerprint(messages[row])
                  for row in range(max(0, start - CHECKPOINT_ROWS), start)]
        child_index = self._child_index() if self._chat_root == self.root else {}
        known_keys = set(child_index.values())
        tool_names: dict[str, str] = {}
        for row, message in enumerate(messages):
            if row >= end:
                break
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") if isinstance(call, dict) else None
                    if (isinstance(call.get("id"), str) and isinstance(function, dict)
                            and isinstance(function.get("name"), str)):
                        tool_names[call["id"]] = function["name"]
            call_id = message.get("tool_call_id") if message.get("role") == "tool" else None
            tool_name = tool_names.pop(call_id, None) if isinstance(call_id, str) else None
            if row < start:
                continue
            if tool_name == "delegate_task" and self._chat_root == self.root:
                fingerprint = _row_fingerprint(message)
                arrived = self._arrived_records.pop(fingerprint, None)
                if arrived is not None:
                    # A selection of this running turn already resolved the row and
                    # claimed its labels; reuse it so the logged [Label] matches the
                    # registered mapping.
                    records = [("work", record, name, key) for name, record, key in arrived]
                    for _name, _record, key in arrived:
                        if key is not None:
                            known_keys.add(key)
                else:
                    records = [("work", record, name, key) for name, record, key in
                               self._delegate_work(_content_text(message.get("content")), known_keys)]
                # Ingestion resolved this row: a later selection must not resolve it
                # again (each child's label was claimed during the resolution above).
                self._arrived_rows.add(fingerprint)
            else:
                records = [(kind, text, None, None) for kind, text in
                           self._records(message, self.node_bytes, tool_name)]
            for record_index, (kind, text, _label, _key) in enumerate(records):
                index = log.append(kind, text)
                if record_index == 0:
                    archive_media(chat_root, index, message.get("content"))
                compactor.seed(index, text, kind)
                view.add_root(index)
            # Advance only once every record of this row is durable, so a crash
            # mid-row re-ingests that row instead of skipping its other records.
            recent.append(_row_fingerprint(message))
            self._save_checkpoint(row + 1, recent)

    def on_message(self, messages: list[dict[str, Any]], message: dict[str, Any],
                   **kwargs: Any) -> None:
        """Ingest the message the host just appended, as the turn unfolds.

        Fires after each host append (assistant replies and tool results), so the
        log is written during the turn instead of only at its boundary. Ingestion
        starts at the checkpoint boundary, so rows the host appended without a
        notification (the turn's user row, synthetic nudges) are ingested too;
        the appended row itself is located by equality against the structural
        clone the host hands over, never assumed to be the last row. Fail-open:
        anything unexpected leaves the turn untouched, and ``on_turn_complete``
        reconciles at the boundary.
        """
        self._open()
        if not self._writes:
            return
        checkpoint = self._load_checkpoint()
        start = self._resume_row(messages, checkpoint)
        if start is None:
            # The recorded rows are gone (a compression rewrote them). Ingest
            # nothing here; the turn-complete reconciler warns once instead of
            # guessing a boundary that could duplicate history.
            return
        row = next((index for index in range(len(messages) - 1, -1, -1)
                    if messages[index] == message), None)
        if row is None or start >= row + 1:
            # Not found, or already ingested (a duplicate notification).
            return
        self._ingest_rows(messages, start, row + 1)

    def on_turn_complete(self, messages: list[dict[str, Any]], usage: dict[str, Any] | None = None,
                         **kwargs: Any) -> None:
        # _open() first: an engine whose session never started learns its context there
        # (on_session_start("") defaults to a primary, writable context).
        log, tree, view, compactor = self._open()
        if not self._writes:
            return
        # The host resends a transcript prefix. The boundary is found by row
        # identity, not by a counter: a counter resets on restart (re-ingesting
        # everything) and skips a transcript that happens to be the same length.
        checkpoint = self._load_checkpoint()
        start = self._resume_row(messages, checkpoint)
        if start is None:
            if not self._warned_ingest_boundary:
                self._warned_ingest_boundary = True
                warnings.warn(
                    "OptChat cannot find its ingestion boundary in this transcript: the "
                    "recorded rows are gone (a compression rewrote them). Ingesting nothing "
                    "this turn rather than guessing a boundary that could duplicate history.",
                    RuntimeWarning, stacklevel=2)
            return
        # The reconciler: rows already ingested through on_message are before the
        # checkpoint boundary, so this walk adds only what it has not seen.
        self._ingest_rows(messages, start, len(messages))
        # One save per turn, not per record: the log append is already fsync'd per
        # record, and the tree and view are derived state.
        tree.save()
        self._check_compaction_tools()
        self._check_compaction_route()
        compactor.run_pending(limit=8)
        tree.save()
        view.merge_batch(log.count())
        view.save()
        if usage is not None:
            self.update_from_response(usage)

    def on_session_reset(self) -> None:
        pass

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        return False

    def compress(self, messages: list[dict[str, Any]], current_tokens: int | None = None,
                 focus_topic: str | None = None, force: bool = False,
                 memory_context: str = "") -> list[dict[str, Any]]:
        """Satisfy the host ABC; the host must not invoke lossy compression."""
        return messages

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0) or 0
        self.last_completion_tokens = usage.get("completion_tokens", 0) or 0
        self.last_total_tokens = usage.get("total_tokens", 0) or 0

    def clone_for_agent(self) -> OptChatEngine:
        self._open()
        clone = type(self)(self.root, view_min_bytes=self.view_min_bytes,
                           view_max_bytes=self.view_max_bytes, node_bytes=self.node_bytes,
                           summarize=self.summarize)
        clone.log, clone.tree, clone.view, clone.compactor = (
            self.log, self.tree, self.view, self.compactor)
        clone._chat_root = self._chat_root
        clone._master_view = self._master_view
        clone._master_log = self._master_log
        clone._master_tree = self._master_tree
        clone._session_id = self._session_id
        clone._writes = self._writes
        clone._turn_system = self._turn_system
        clone._turn_tools = self._turn_tools
        clone._has_turn_tools = self._has_turn_tools
        clone._turn_model = self._turn_model
        for field in ("threshold_tokens", "context_length", "threshold_percent"):
            if hasattr(self, field):
                setattr(clone, field, getattr(self, field))
        return clone

    def _capture_turn_tools(self, tools: list[dict[str, Any]]) -> None:
        """Record the request's outgoing tools for compactions.

        The single capture rule, shared by ``cache_plan`` (the mark-based route)
        and ``select_context`` (every route, including the no-mark one): a deep
        copy of the host's schemas, ``_has_turn_tools`` latched, and the
        compactor updated at once so the next compaction reads the turn's tools.
        """
        self._turn_tools = copy.deepcopy(tools)
        self._has_turn_tools = True
        if self.compactor is not None:
            self.compactor.tools = self._turn_tools

    def _plan_cache(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        """Shared seam/shim entry, called only after the host's explicit plan."""
        self._capture_turn_tools(tools)
        return plan_view_cache(messages, tools)


if hasattr(ContextEngine, "cache_plan"):
    def _cache_plan(self: OptChatEngine, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        sections = self._plan_cache(messages, tools)
        if sections is None:
            return None
        try:
            from agent.prompt_caching import PromptCachePlan
        except ImportError:
            return None
        return PromptCachePlan(*sections)

    OptChatEngine.cache_plan = _cache_plan

"""Hermes ContextEngine adapter for the durable OptChat log and view.

Host messages map to log kinds as follows: system is host-owned and omitted;
user becomes user; text-only assistant becomes unii; assistant with tool_calls
becomes tool (name and JSON arguments, including any accompanying content);
delegate_task results become work, and other tool results become echo.
Thinking fields are never logged.

On a no-mark route, the outgoing tools are the one part of the turn's request
identity that compaction cannot yet match: the host's cache-policy gates the
only hook carrying tools. Until a host seam passes tools to ``select_context``,
the navigation schemas remain the fallback, with a warning once per engine.

The retained request tail starts at the host's ``incoming_message`` when that
message is present in the request (identity first, then equality). Otherwise it
starts at the last user row. A later steering user row stays in the same turn.

Navigation is scope-explicit. An integer id without a scope resolves against the
log whose view the caller is reading - the master's, for the master and for a
child rendering the master's view; a child's own chat is reached only with an
explicit scope. ``scope="Name"`` resolves against that child's chat, and an
unknown or unmapped scope is an error, never a silent fallback. A delegate
result's name-to-child mapping is registered as soon as the result is visible to
a selection of the running turn, not only at ``on_turn_complete``; a name already
mapped to a different child keeps its first mapping and warns once.

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
from .view import View
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
        # Delegate results already seen by a selection of the running turn, keyed
        # by row fingerprint: ``_arrived_rows`` guards against re-resolving a row,
        # ``_arrived_records`` carries its resolution to ingestion.
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
                       budget_tokens: int = 0) -> list[dict[str, Any]] | None:
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
        if not system:
            system = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            content = _content_text(system[0].get("content"))
            if SYSTEM_PROMPT not in content:
                system[0]["content"] = content + "\n\n" + SYSTEM_PROMPT
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
        return system + [{"role": "assistant", "content": (
            self._master_view or view).render()}] + turn

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
            if scope is not None:
                if isinstance(args.get("id"), str):
                    return json.dumps({"error": "a string id opens its own chat; "
                                                "do not pass scope with it"})
                child = self._child_scope(scope)
                if isinstance(child, str):
                    return child
                child_log, child_tree = child
                return opt_tools.zoom(child_log, child_tree, None, args.get("id"),
                                      args.get("n", 1), scope=scope)
            log, tree, view = self._reading_scope()
            return opt_tools.zoom(log, tree, view, args.get("id"), args.get("n", 1),
                                  master_root=self.root)
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

    def _register_child(self, name: str, key: str) -> None:
        """Record ``name -> key``; an existing mapping to another child is kept.

        The first mapping wins: a repeated name that resolves to a different child
        warns once (per name) instead of silently repointing the model at another
        chat. Re-registering the same pair is idempotent.
        """
        index = self._child_index()
        existing = index.get(name)
        if existing == key:
            return
        if existing is not None:
            if name not in self._warned_child_conflicts:
                self._warned_child_conflicts.add(name)
                warnings.warn(
                    f"OptChat: subagent name {name!r} already maps to child {existing!r}; "
                    f"keeping it and ignoring child {key!r}.",
                    RuntimeWarning, stacklevel=2)
            return
        index[name] = key
        self._save_child_index(index)

    def _register_arrived_delegates(self, messages: list[dict[str, Any]], start: int) -> None:
        """Register ``name -> child`` for delegate results already visible this turn.

        The host shows the running turn's rows to ``select_context`` as they arrive,
        so a report can be zoomed by name in the turn it returns, not only after
        ``on_turn_complete`` ingests it. Each result row is resolved once (by row
        fingerprint) and its resolution is kept for ingestion, so the logged
        ``[Name]`` always matches the registered mapping. Rows before the turn
        boundary are left to ingestion.
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
                for name, _record, key in resolved:
                    if key is not None:
                        self._register_child(name, key)
        except (OSError, ValueError) as exc:
            logger.warning("OptChat could not register an arrived subagent report: %s", exc)

    def _delegate_work(self, text: str, known_keys: set[str]) -> list[tuple[str, str, str | None]]:
        """Return (label, record text, child key) for a completed delegate result."""
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
            work.append((name, f"[{name}] {body}", key))
        return work

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
        recent = [_row_fingerprint(messages[row])
                  for row in range(max(0, start - CHECKPOINT_ROWS), start)]
        child_index = self._child_index() if self._chat_root == self.root else {}
        known_keys = set(child_index.values())
        tool_names: dict[str, str] = {}
        for row, message in enumerate(messages):
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
                arrived = self._arrived_records.pop(_row_fingerprint(message), None)
                if arrived is not None:
                    # A selection of this running turn already resolved the row;
                    # reuse it so the logged [Name] matches the registered mapping.
                    records = [("work", record, name, key) for name, record, key in arrived]
                    for _name, _record, key in arrived:
                        if key is not None:
                            known_keys.add(key)
                else:
                    records = [("work", record, name, key) for name, record, key in
                               self._delegate_work(_content_text(message.get("content")), known_keys)]
            else:
                records = [(kind, text, None, None) for kind, text in
                           self._records(message, self.node_bytes, tool_name)]
            for record_index, (kind, text, name, key) in enumerate(records):
                index = log.append(kind, text)
                if record_index == 0:
                    archive_media(self._chat_root, index, message.get("content"))
                compactor.seed(index, text, kind)
                view.add_root(index)
                if name is not None and key is not None:
                    self._register_child(name, key)
            # Advance only once every record of this row is durable, so a crash
            # mid-row re-ingests that row instead of skipping its other records.
            recent.append(_row_fingerprint(message))
            self._save_checkpoint(row + 1, recent)
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

    def _plan_cache(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        """Shared seam/shim entry, called only after the host's explicit plan."""
        self._turn_tools = copy.deepcopy(tools)
        self._has_turn_tools = True
        if self.compactor is not None:
            self.compactor.tools = self._turn_tools
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

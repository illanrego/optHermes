"""Hermes ContextEngine adapter for the durable OptChat log and view.

Host messages map to log kinds as follows: system is host-owned and omitted;
user becomes user; text-only assistant becomes unii; assistant with tool_calls
becomes tool (name and JSON arguments, including any accompanying content);
delegate_task results become work, and other tool results become echo.
Thinking fields are never logged.

The host ABC is imported here, rather than by the package entry point. When
Hermes is absent, a local base permits standalone unit tests; an engine loaded
inside Hermes inherits the real ABC. This binding is fixed at module import.
"""

from __future__ import annotations

import copy
import json
import logging
import os
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
from .prompts import SYSTEM_PROMPT
from .store import Log
from .text import split_text
from .tree import Tree
from .view import View
from . import tools as opt_tools


logger = logging.getLogger(__name__)


def default_root() -> Path:
    """The chat directory for the active Hermes home.

    The context-engine directory loader hands a plugin a minimal collector with no
    plugin manager, so the home cannot be read from ``ctx``. The launcher exports
    HERMES_HOME for the active profile, which is the same value, and reading it here
    keeps the engine profile-scoped.
    """
    env = os.environ.get("HERMES_HOME")
    return (Path(env) if env else Path.home() / ".hermes") / "optchat"


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
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
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
        self.root = Path(root) if root is not None else default_root()
        config = _load_config(self.root)
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
        self._chat_root = self.root
        self._turn_system = ""
        self._turn_tools: list[dict[str, Any]] = []
        self._seen_messages = 0
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
        platform = kwargs.get("platform") or "cli"
        self._writes = platform != "cron"
        if platform == "subagent":
            if not session_id or Path(session_id).name != session_id or session_id in (".", ".."):
                raise ValueError("Subagent session id must be a safe path component")
            if self._master_view is None:
                self._master_view = self.view
                if self._master_view is None and (self.root / "view.json").is_file():
                    self._master_view = View(Tree(self.root, self.node_bytes),
                                             self.view_min_bytes, self.view_max_bytes)
            child_root = self.root / "agents" / session_id
            if self._chat_root == child_root and self.log is not None:
                return
            self.log = self.tree = self.view = self.compactor = None
            self._chat_root = child_root
            self._seen_messages = 0
        elif self.log is not None:
            return
        self.log = Log(self._chat_root)
        try:
            self.tree = Tree(self._chat_root, self.node_bytes)
            self.view = View(self.tree, self.view_min_bytes, self.view_max_bytes)
            self.compactor = Compactor(self.tree, self.summarize or _missing_summarizer,
                                       node_bytes=self.node_bytes, view=self.view)
            self._turn_tools = self.get_tool_schemas()
            self.compactor.system = self._turn_system
            self.compactor.tools = self._turn_tools
        except BaseException:
            self.log.close()
            self.log = None
            raise

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
        last_user = next((i for i in range(len(request_messages) - 1, -1, -1)
                          if request_messages[i].get("role") == "user"), None)
        if last_user is None:
            return None
        system = [copy.deepcopy(message) for message in request_messages
                  if message.get("role") == "system"]
        if not system:
            system = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            content = _content_text(system[0].get("content"))
            if SYSTEM_PROMPT not in content:
                system[0]["content"] = content + "\n\n" + SYSTEM_PROMPT
        turn = [copy.deepcopy(message) for message in request_messages[last_user:]]
        _, tree, view, compactor = self._open()
        self._turn_system = _content_text(system[0].get("content"))
        compactor.system = self._turn_system
        if not hasattr(ContextEngine, "cache_plan"):
            self._turn_tools = self.get_tool_schemas()
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
        except Exception:
            logger.exception("OptChat settle failed; serving the current view")
        logger.debug("OptChat settle built %d nodes; view settled: %s", built, view.settled())
        return system + [{"role": "assistant", "content": (
            self._master_view or view).render()}] + turn

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        integer = {"type": "integer"}
        return [
            {"type": "function", "function": {"name": "optchat_zoom",
                "description": "Open a child chat by name, a chat span, or a raw message.",
                "parameters": {"type": "object", "properties": {
                    "id": {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "n": integer},
                               "required": ["id"]}}},
            {"type": "function", "function": {"name": "optchat_date",
                "description": "Get the stored ISO timestamp of a raw message.",
                "parameters": {"type": "object", "properties": {"id": integer},
                               "required": ["id"]}}},
        ]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if not isinstance(args, dict):
            return json.dumps({"error": "tool arguments must be an object"})
        log, tree, view, _ = self._open()
        if name == "optchat_zoom":
            return opt_tools.zoom(log, tree, view, args.get("id"), args.get("n", 1),
                                  master_root=self.root)
        if name == "optchat_date":
            return opt_tools.date(log, args.get("id"))
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    @staticmethod
    def _records(message: dict[str, Any], node_bytes: int,
                 tool_name: str | None = None) -> list[tuple[str, str]]:
        role = message.get("role")
        if role == "user":
            return [("user", chunk) for chunk in split_text(
                _content_text(message.get("content")), node_bytes)]
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
            return [("unii", chunk) for chunk in split_text(content, node_bytes)] if content else []
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
        # The host resends a transcript prefix. Count host rows, not log records:
        # one assistant row can contain several tool calls or no visible text.
        if len(messages) < self._seen_messages:
            self._seen_messages = 0
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
            if row < self._seen_messages:
                continue
            if tool_name == "delegate_task" and self._chat_root == self.root:
                records = [("work", record, name, key) for name, record, key in
                           self._delegate_work(_content_text(message.get("content")), known_keys)]
            else:
                records = [(kind, text, None, None) for kind, text in
                           self._records(message, self.node_bytes, tool_name)]
            for kind, text, name, key in records:
                index = log.append(kind, text)
                compactor.seed(index, text)
                view.add_root(index)
                if name is not None and key is not None:
                    child_index[name] = key
                    self._save_child_index(child_index)
        self._seen_messages = len(messages)
        # One save per turn, not per record: the log append is already fsync'd per
        # record, and the tree and view are derived state.
        tree.save()
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
        clone._seen_messages = self._seen_messages
        clone._writes = self._writes
        clone._turn_system = self._turn_system
        clone._turn_tools = self._turn_tools
        for field in ("threshold_tokens", "context_length", "threshold_percent"):
            if hasattr(self, field):
                setattr(clone, field, getattr(self, field))
        return clone


if hasattr(ContextEngine, "cache_plan"):
    def _cache_plan(self: OptChatEngine, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        # A mismatched tool list breaks the turn cache prefix compaction reuses.
        self._turn_tools = copy.deepcopy(tools)
        if self.compactor is not None:
            self.compactor.tools = self._turn_tools
        try:
            from agent.prompt_caching import (PromptCachePlan,
                strip_anthropic_cache_control, strip_anthropic_tool_cache_control)
        except ImportError:
            return None
        planned = copy.deepcopy(messages)
        index = next((i for i, message in enumerate(planned)
                      if message.get("role") == "assistant"
                      and _content_text(message.get("content")).startswith("<chat>\n")), None)
        if index is None:
            return None
        strip_anthropic_cache_control(planned)
        planned[index]["content"] = [{"type": "text", "text": _content_text(planned[index].get("content")),
                                       "cache_control": {"type": "ephemeral"}}]
        return PromptCachePlan(planned, strip_anthropic_tool_cache_control(tools))

    OptChatEngine.cache_plan = _cache_plan

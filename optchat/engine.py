"""Hermes ContextEngine adapter for the durable OptChat log and view.

Host messages map to log kinds as follows: system is host-owned and omitted;
user becomes user; text-only assistant becomes unii; assistant with tool_calls
becomes tool (name and JSON arguments, including any accompanying content);
tool results become echo. Thinking fields are never logged.

The host ABC is imported here, rather than by the package entry point. When
Hermes is absent, a local base permits standalone unit tests; an engine loaded
inside Hermes inherits the real ABC. This binding is fixed at module import.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    from agent.context_engine import ContextEngine
except ModuleNotFoundError as exc:
    if exc.name != "agent":
        raise

    class ContextEngine:  # type: ignore[no-redef]
        """Standalone test base when the Hermes package is unavailable."""

from .compactor import Compactor, Summarizer
from .prompts import SYSTEM_PROMPT
from .store import Log
from .tree import Tree
from .view import View
from . import tools as opt_tools


def _missing_summarizer(_prompt: str) -> str:
    raise RuntimeError("No OptChat summarizer has been configured")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
    return "" if content is None else str(content)


class OptChatEngine(ContextEngine):
    def __init__(self, root: Path, *, view_min_bytes: int = 64_000,
                 view_max_bytes: int = 128_000, node_bytes: int = 512,
                 summarize: Summarizer | None = None):
        self.root = Path(root)
        self.view_min_bytes = view_min_bytes
        self.view_max_bytes = view_max_bytes
        self.node_bytes = node_bytes
        self.summarize = summarize
        self.log: Log | None = None
        self.tree: Tree | None = None
        self.view: View | None = None
        self.compactor: Compactor | None = None
        self._seen_messages = 0
        # Only a primary context writes. Subagents and cron runs read the view but must
        # never append to the chat: the host reports platform="subagent" / "cron" on
        # on_session_start. Fail closed until that says otherwise.
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
        # Set this before the early return: a clone shares the parent's Log, so it would
        # otherwise never learn its own context and would inherit the parent's write flag.
        self._writes = (kwargs.get("platform") or "cli") not in ("subagent", "cron")
        if self.log is not None:
            return
        self.log = Log(self.root)
        try:
            self.tree = Tree(self.root, self.node_bytes)
            self.view = View(self.tree, self.view_min_bytes, self.view_max_bytes)
            self.compactor = Compactor(self.tree, self.summarize or _missing_summarizer,
                                       node_bytes=self.node_bytes)
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
        _, _, view, _ = self._open()
        return system + [{"role": "assistant", "content": view.render()}] + turn

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        integer = {"type": "integer"}
        return [
            {"type": "function", "function": {"name": "optchat_zoom",
                "description": "Open a chat view span into its two halves, or a single raw message.",
                "parameters": {"type": "object", "properties": {"id": integer, "n": integer},
                               "required": ["id", "n"]}}},
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
            return opt_tools.zoom(log, tree, view, args.get("id"), args.get("n"))
        if name == "optchat_date":
            return opt_tools.date(log, args.get("id"))
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    @staticmethod
    def _records(message: dict[str, Any]) -> list[tuple[str, str]]:
        role = message.get("role")
        if role == "user":
            return [("user", _content_text(message.get("content")))]
        if role == "tool":
            text = _content_text(message.get("content"))
            if len(text) > 30_000:
                text = text[:15_000] + text[-15_000:]
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
            return [("unii", content)] if content else []
        return []

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
        for message in messages[self._seen_messages:]:
            for kind, text in self._records(message):
                index = log.append(kind, text)
                compactor.seed(index, text)
                view.add_root(index)
            self._seen_messages += 1
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
        clone._seen_messages = self._seen_messages
        clone._writes = self._writes
        for field in ("threshold_tokens", "context_length", "threshold_percent"):
            if hasattr(self, field):
                setattr(clone, field, getattr(self, field))
        return clone


if hasattr(ContextEngine, "cache_plan"):
    def _cache_plan(self: OptChatEngine, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]], **kwargs: Any) -> Any:
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

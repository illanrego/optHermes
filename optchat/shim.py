"""Interim stock-Hermes cache shim; remove when the upstream cache seam lands.

Stock Hermes strips request cache marks before applying its own plan. This shim
wraps that pass and marks the single assistant ``<chat>`` view's content after
the host plan. It leaves the host's request-end breakpoint alone. The assistant
role keeps the view separate from the changing user turn, so the view remains a
stable cache prefix. The upstream ``ContextEngine.cache_plan`` seam makes this
module unnecessary.
"""

from __future__ import annotations

import functools
import logging
from typing import Any


logger = logging.getLogger(__name__)


def _warn_missing_once() -> None:
    if not getattr(_warn_missing_once, "warned", False):
        logger.warning("Hermes cache decorator is unavailable; OptChat cache shim is inactive")
        _warn_missing_once.warned = True


def _view_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
        isinstance(part, dict) and part.get("type", "text") == "text"
        and isinstance(part.get("text"), str) for part in content
    ):
        return "".join(part["text"] for part in content)
    return None


def _mark_view(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        text = _view_text(content)
        if text is None or not text.startswith("<chat>\n"):
            continue
        marker = {"type": "ephemeral"}
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content,
                                   "cache_control": marker}]
        else:
            parts = list(content)
            parts[-1] = {**parts[-1], "cache_control": marker}
            message["content"] = parts
        return


def install(engine: Any) -> None:
    """Wrap the host's request-local decorator for the active OptChat engine."""
    if engine is None:
        return
    try:
        from agent.context_engine import ContextEngine
    except ImportError:
        pass
    else:
        if callable(getattr(ContextEngine, "cache_plan", None)):
            return  # The upstream seam owns this breakpoint now.
    try:
        from agent import conversation_loop
    except ImportError:
        _warn_missing_once()
        return
    original = getattr(conversation_loop, "_redecorate_prompt_cache_for_provider", None)
    if not callable(original):
        _warn_missing_once()
        return
    if getattr(original, "_optchat_cache_shim_original", None) is not None:
        return

    @functools.wraps(original)
    def wrapped(agent: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(agent, *args, **kwargs)
        if getattr(agent, "context_compressor", None) is engine:
            messages = result[0] if isinstance(result, tuple) and result else None
            if isinstance(messages, list):
                _mark_view(messages)
        return result

    wrapped._optchat_cache_shim_original = original  # type: ignore[attr-defined]
    conversation_loop._redecorate_prompt_cache_for_provider = wrapped


def uninstall() -> None:
    """Restore the exact original host function when this shim is installed."""
    try:
        from agent import conversation_loop
    except ImportError:
        return
    current = getattr(conversation_loop, "_redecorate_prompt_cache_for_provider", None)
    original = getattr(current, "_optchat_cache_shim_original", None)
    if original is not None:
        conversation_loop._redecorate_prompt_cache_for_provider = original

"""Interim stock-Hermes cache shim; remove when the upstream cache seam lands.

Stock Hermes strips request cache marks before applying its own plan. This shim
wraps that pass and dispatches through the request's active OptChat engine, using
the same planner as the seam. The upstream ``ContextEngine.cache_plan`` seam
makes this module unnecessary.
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
        # Mirror the host's branch, including the prepared MoA bypass. This is
        # its policy decision, not provider detection based on names or marks.
        prepared = kwargs.get("moa_prepared")
        if (not getattr(agent, "_use_prompt_caching", False)
                or prepared is not None and getattr(agent, "provider", None) == "moa"):
            return result
        from .engine import OptChatEngine
        active = getattr(agent, "context_compressor", None)
        if isinstance(active, OptChatEngine) and isinstance(result, tuple) and len(result) == 3:
            messages, prepared, tools = result
            sections = active._plan_cache(messages, tools)
            if sections is not None:
                return sections[0], prepared, sections[1]
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

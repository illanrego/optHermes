"""Request-local view breakpoints layered onto the host's explicit cache plan.

The caller owns the provider gate. This module never infers a route from a
model name, endpoint, or the presence of markers.
"""

from __future__ import annotations

import copy
from typing import Any


MAX_BREAKPOINTS = 4
LOOKBACK_BLOCKS = 20


def _view_text(message: dict[str, Any]) -> str | None:
    if message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type", "text") == "text"
        and isinstance(part.get("text"), str)
        and set(part) <= {"type", "text", "cache_control"} for part in content
    ):
        content = "".join(part["text"] for part in content)
    if (isinstance(content, str) and content.startswith("<chat>\n")
            and content.endswith("\n</chat>")):
        return content
    return None


def _marked(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("content")
    sites = [message] + (parts if isinstance(parts, list) else [])
    return [site for site in sites if isinstance(site, dict) and "cache_control" in site]


def _block_count(message: dict[str, Any]) -> int:
    # Count content parts and tool calls, not message rows. Counting nested tool
    # result parts separately is conservative for adapters that envelope them.
    content = message.get("content")
    return max(1, len(content) if isinstance(content, list) else int(bool(content))) + len(
        message.get("tool_calls") or [])


def plan_view_cache(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]]] | None
):
    """Preserve prefix marks; budget complete-view and host-legal end marks.

    Each newline-terminated view line is a complete block. The opening tag has
    its own block; the closing tag is a separate, unmarked tail. Appending lines
    leaves every preceding block and boundary byte-identical. Merges naturally
    invalidate the affected prefix. No persisted view or transcript is changed.

    System/tool markers (including TTLs) are protected. Only excess conversation
    markers are evicted, oldest first. The host has at most two prefix markers
    in its supported layouts. If an incompatible plan reserves all four slots,
    retain it instead of sacrificing a prefix boundary.
    """
    index = next((i for i, message in enumerate(messages)
                  if isinstance(message, dict) and _view_text(message) is not None), None)
    if index is None:
        return None
    planned, planned_tools = copy.deepcopy((messages, tools))
    protected = [tool for tool in planned_tools if "cache_control" in tool]
    protected += [site for message in planned if message.get("role") == "system"
                  for site in _marked(message)]
    if len(protected) >= MAX_BREAKPOINTS:
        return None

    view = planned[index]
    old_view_marks = _marked(view)
    conversation_marks = [site for message in planned if message.get("role") != "system"
                          for site in _marked(message)]
    sources = old_view_marks or conversation_marks or protected
    marker = copy.deepcopy(sources[-1]["cache_control"] if sources else {"type": "ephemeral"})
    text = _view_text(view)
    assert text is not None
    # Split on literal LF, matching View.render (Unicode line separators are
    # ordinary summary text). Keep delimiters so concatenation is byte-exact.
    blocks = [{"type": "text", "text": line + "\n"}
              for line in text[:-len("</chat>")].split("\n")[:-1]]
    blocks.append({"type": "text", "text": "</chat>"})
    view.pop("cache_control", None)
    view["content"] = blocks

    # Reuse the host's choice of the last legal cacheable message. In particular,
    # never invent a part marker on a tool result when its adapter forbids one.
    def can_mark_end(message: dict[str, Any]) -> bool:
        if message.get("role") == "system":
            return False
        if _marked(message):
            return True
        content = message.get("content")
        return (message.get("role") in ("user", "assistant")
                and not message.get("tool_calls")
                and (isinstance(content, str) and bool(content)
                     or isinstance(content, list) and bool(content)
                     and isinstance(content[-1], dict) and content[-1].get("type") == "text"
                     and bool(content[-1].get("text"))))

    end_message = next((message for message in reversed(planned[index + 1:])
                        if can_mark_end(message)), None)
    end_site = None
    if end_message is not None:
        existing = _marked(end_message)
        end_site = existing[-1] if existing else None
        content = end_message.get("content")
        if isinstance(content, str) and end_site is None:
            content = end_message["content"] = [{"type": "text", "text": content}]
        if end_site is not end_message and isinstance(content, list):
            # A host scaffold mark can precede the changing tail. Keep that mark
            # if budget permits, but also cache the actual request endpoint.
            end_marker = copy.deepcopy(end_site["cache_control"] if end_site else marker)
            end_site = content[-1]
            end_site["cache_control"] = end_marker
    elif index == len(planned) - 1:
        end_site = blocks[-1]
        end_site["cache_control"] = copy.deepcopy(marker)

    # The closing tag plus the current turn must fit in the lookback window.
    # Long tool turns retain the host's recent marks instead of spending a slot
    # on a view boundary more than 19 block positions behind the request end.
    trailing = 1 + sum(_block_count(message) for message in planned[index + 1:])
    view_site = None
    if len(blocks) > 2 and trailing < LOOKBACK_BLOCKS:
        view_site = blocks[-2]
        view_site["cache_control"] = copy.deepcopy(marker)

    sites = [site for message in planned if message.get("role") != "system"
             for site in _marked(message)]
    excess = len(protected) + len(sites) - MAX_BREAKPOINTS
    # Request end has priority if an unusual host plan leaves just one slot.
    eviction_order = [site for site in sites if site is not end_site and site is not view_site]
    if view_site is not None:
        eviction_order.append(view_site)
    for site in eviction_order[:max(0, excess)]:
        site.pop("cache_control", None)
    return planned, planned_tools

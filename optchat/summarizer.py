"""Route OptChat compactions through Hermes' configured auxiliary task."""

from __future__ import annotations

import logging
import math

from .compactor import CompactionRequest, Summarizer


logger = logging.getLogger(__name__)


def _ordinary_content(response: object) -> object:
    """Read the reply's content without consulting reasoning fields."""
    if isinstance(response, str):
        return response
    choices = (response.get("choices") if isinstance(response, dict)
               else getattr(response, "choices", None))
    if choices is not None:
        if not choices:
            return None
        first = choices[0]
        response = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    return response.get("content") if isinstance(response, dict) else getattr(response, "content", None)


def make_host_summarizer(*, timeout: float = 300.0) -> Summarizer:
    """Return a request-to-line call using the host's ``auxiliary.optchat`` route.

    Host imports stay inside this factory so standalone OptChat imports need no
    Hermes installation. The host owns provider, model, credentials and retries.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite, positive number of seconds")

    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    try:
        from agent.auxiliary_task_config import _get_auxiliary_task_config
    except ImportError:
        # Older hosts ship no task-config reader; call_llm(task=...) resolves the
        # route itself, so the pre-check is simply skipped.
        _get_auxiliary_task_config = None

    def summarize(request: CompactionRequest) -> str:
        if not isinstance(request, CompactionRequest):
            raise TypeError("compaction request must be a CompactionRequest")
        if _get_auxiliary_task_config is not None:
            route = _get_auxiliary_task_config("optchat")
            provider = route.get("provider") or "auto"
            model = route.get("model") or ""
            if provider != "auto" and not model:
                raise RuntimeError("auxiliary.optchat pins a provider but no model")
        user_content = (request.context + "\n\n" + request.task
                        if request.context else request.task)
        messages = ([{"role": "system", "content": request.system}]
                    if request.system else [])
        messages.append({"role": "user", "content": user_content})
        logger.debug("OptChat compaction prompt: %d UTF-8 bytes",
                     len(user_content.encode("utf-8")))
        response = call_llm(
            task="optchat",
            messages=messages,
            tools=request.tools or None,
            timeout=float(timeout),
        )
        # Pass only ordinary content to the host's think-block scrubber. A reply
        # with reasoning fields alone must not be turned into a tree line.
        line = " ".join(extract_content_or_reasoning(
            {"content": _ordinary_content(response)}).splitlines()).strip()
        if not line:
            raise RuntimeError("OptChat auxiliary model returned an empty reply")
        return line

    return summarize

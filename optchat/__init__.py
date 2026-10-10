"""Persistent components for the OptChat context engine."""

import logging
from pathlib import Path

from .engine import OptChatEngine, default_root

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register the engine with the host.

    The context-engine directory loader passes a minimal collector: it exposes
    ``register_context_engine`` but carries no plugin manager, so the Hermes home
    cannot be read from ``ctx``. Fall back to ``default_root()`` (the launcher's
    HERMES_HOME), which keeps the engine profile-scoped.
    """
    register_engine = getattr(ctx, "register_context_engine", None)
    if not callable(register_engine):
        return
    manager = getattr(ctx, "_manager", None)
    home = getattr(manager, "home_path", None) if manager is not None else None
    root = (Path(home) / "optchat") if home is not None else default_root()

    engine = OptChatEngine(root)
    try:
        from .summarizer import make_host_summarizer

        engine.summarize = make_host_summarizer()
    except Exception as exc:  # optional route; a missing one fails closed per call
        logger.warning("OptChat: host summarizer unavailable (%s)", exc)
    try:
        from . import shim

        # A no-op when the upstream ContextEngine.cache_plan seam is present.
        shim.install(engine)
    except Exception as exc:
        logger.debug("OptChat: cache shim not installed (%s)", exc)

    register_engine(engine)


__all__ = ["OptChatEngine", "default_root", "register"]

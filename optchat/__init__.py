"""Persistent components for the OptChat context engine."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from .engine import OptChatEngine, default_root
    from .notes import import_notes
except Exception as exc:
    logger.error(
        "OptChat is not active: the package failed to import (%s). The host will fall "
        "back to its built-in compressor and compress this session lossily; the fold "
        "and the no-loss guarantee are NOT in effect.", exc)
    raise


def _not_active(reason: str) -> None:
    """One loud line for every path where the engine never becomes active."""
    logger.error(
        "OptChat is not active: %s. The host will keep its built-in compressor for "
        "this session and compress it lossily; the fold and the no-loss guarantee "
        "are NOT in effect.", reason)


def register(ctx) -> None:
    """Register the engine with the host.

    The context-engine directory loader passes a minimal collector: it exposes
    ``register_context_engine`` but carries no plugin manager, so the Hermes home
    cannot be read from ``ctx``. When no manager home is available, pass no root:
    the engine then resolves its chat directory from the ``hermes_home`` the host
    supplies at session start, falling back to ``default_root()`` (the launcher's
    HERMES_HOME). Passing ``default_root()`` here would freeze the ambient home and
    the host's own home could never take precedence.

    Every failure path here is loud: the host logs a failed load at debug level and
    silently keeps its built-in compressor, so the lost guarantee must be named by
    the engine itself.
    """
    register_engine = getattr(ctx, "register_context_engine", None)
    if not callable(register_engine):
        _not_active("the host collector has no register_context_engine")
        return
    manager = getattr(ctx, "_manager", None)
    home = getattr(manager, "home_path", None) if manager is not None else None
    root = (Path(home) / "optchat") if home is not None else None

    try:
        engine = OptChatEngine(root)
    except Exception as exc:
        _not_active(f"the engine could not be constructed ({exc})")
        return
    try:
        from .summarizer import make_host_summarizer

        engine.summarize = make_host_summarizer()
    except Exception as exc:  # optional route; a missing one fails closed per call
        logger.error(
            "OptChat is active but cannot summarize: the host summarizer route is "
            "unavailable (%s). Compactions cannot run, so long messages stay "
            "unsummarized and the fold will not build; navigation still works and "
            "no lossy compression runs.", exc)
    try:
        from . import shim

        # A no-op when the upstream ContextEngine.cache_plan seam is present.
        shim.install(engine)
    except Exception as exc:
        logger.error(
            "OptChat cache shim not installed (%s): the host's own cache plan "
            "applies and OptChat's view cache breakpoints are not added; cache "
            "reuse will be weaker than planned.", exc)

    try:
        register_engine(engine)
    except Exception as exc:
        _not_active(f"the host refused the engine ({exc})")


__all__ = ["OptChatEngine", "default_root", "import_notes", "register"]

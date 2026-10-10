"""Persistent components for the OptChat context engine."""

from pathlib import Path

from .engine import OptChatEngine


def register(ctx) -> None:
    """Register under the plugin manager's captured Hermes home."""
    register_engine = getattr(ctx, "register_context_engine", None)
    manager = getattr(ctx, "_manager", None)
    home = getattr(manager, "home_path", None)
    if callable(register_engine) and home is not None:
        register_engine(OptChatEngine(Path(home) / "optchat"))


__all__ = ["OptChatEngine", "register"]

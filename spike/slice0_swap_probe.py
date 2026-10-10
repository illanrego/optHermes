"""Slice 0 spike: prove the host honors ContextEngine.select_context().

Throwaway probe, not module code. It drives the REAL host function
(`agent.conversation_loop._apply_context_engine_selection`) with a stub agent and
asserts the transcript is withheld and the view injected.

Run with the Hermes interpreter and the seam worktree on sys.path:

    HERMES_SRC=~/Documents/coding/opthermes-seam \
      ~/.hermes/tools/python-3.14.7*/bin/python3 spike/slice0_swap_probe.py
"""
import os
import sys

HERMES_SRC = os.environ.get("HERMES_SRC", os.path.expanduser("~/Documents/coding/opthermes-seam"))
sys.path.insert(0, HERMES_SRC)

from agent.context_engine import ContextEngine  # noqa: E402
from agent.conversation_loop import (  # noqa: E402
    _apply_context_engine_selection,
    _engine_overrides_hook,
)


class SpikeEngine(ContextEngine):
    """Minimal engine that replaces the request with [system][view][this turn]."""

    @property
    def name(self):
        return "spike"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages

    def select_context(self, request_messages, *, conversation_messages=None,
                       incoming_message=None, budget_tokens=0):
        system = [m for m in request_messages if m.get("role") == "system"]
        last_user = [m for m in request_messages if m.get("role") == "user"][-1:]
        view = {"role": "user", "content": "<chat>\n0+1|SPIKE-VIEW: the only memory\n</chat>"}
        return system + [view] + last_user


class StubAgent:
    def __init__(self, engine):
        self.context_compressor = engine
        self.session_id = "spike"


class Logger:
    def warning(self, *a, **k):
        print("WARN:", a)


def main():
    engine = SpikeEngine()
    print("engine overrides select_context:", _engine_overrides_hook(engine, "select_context"))

    convo = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "TURN1 secret the model must NOT see"},
        {"role": "assistant", "content": "TURN1 reply"},
        {"role": "user", "content": "TURN2 secret"},
        {"role": "assistant", "content": "TURN2 reply"},
        {"role": "user", "content": "CURRENT question"},
    ]

    out = _apply_context_engine_selection(StubAgent(engine), convo, convo, convo[-1], logger=Logger())

    print("in :", [(m["role"], m["content"][:28]) for m in convo])
    print("out:", [(m["role"], m["content"][:36]) for m in out])

    leaked = any("TURN1 secret" in (m.get("content") or "") for m in out)
    print("transcript withheld:", not leaked)
    assert not leaked, "transcript leaked"
    assert out[0]["role"] == "system"
    assert "<chat>" in out[1]["content"]
    assert out[-1]["content"] == "CURRENT question"
    assert len(out) == 3, f"expected 3 messages, got {len(out)}"
    print("history untouched:", "TURN1 secret" in convo[1]["content"])
    print("SPIKE PASS: host honors select_context; transcript withheld; view injected")


if __name__ == "__main__":
    main()

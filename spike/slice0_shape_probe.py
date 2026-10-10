"""Slice 0b: which message shape survives Hermes' post-selection passes?

The host's `_drop_thinking_only_and_merge_users` runs AFTER select_context and
merges adjacent `user` messages. This probe asks the real function which shapes
survive, so the view can be a stable cache prefix.

Run:
    HERMES_SRC=~/Documents/coding/opthermes-seam \
      ~/.hermes/hermes-agent/venv/bin/python3 spike/slice0_shape_probe.py
"""
import os
import sys

sys.path.insert(0, os.environ.get("HERMES_SRC", os.path.expanduser("~/Documents/coding/opthermes-seam")))

from run_agent import AIAgent  # noqa: E402

S = {"role": "system", "content": "SYSPROMPT"}
VIEW_U = {"role": "user", "content": "<chat>VIEW</chat>"}
VIEW_A = {"role": "assistant", "content": "<chat>VIEW</chat>"}
VIEW_S = {"role": "system", "content": "<chat>VIEW</chat>"}
MSG = {"role": "user", "content": "MSG"}


def show(label, msgs):
    out = AIAgent._drop_thinking_only_and_merge_users([dict(m) for m in msgs])
    print(f"{label}")
    print("   in :", [(m["role"], m["content"][:20]) for m in msgs])
    print("   out:", [(m["role"], m["content"][:20]) for m in out])
    print("   view survived as its own message:", sum(1 for m in out if "VIEW" in m["content"]) == 1
          and not any("VIEW" in m["content"] and "MSG" in m["content"] for m in out))
    print()


def main():
    show("A  system, user:view, user:msg", [S, VIEW_U, MSG])
    show("B  system, user:view, system:'', user:msg", [S, VIEW_U, {"role": "system", "content": ""}, MSG])
    show("C  system, assistant:view, user:msg", [S, VIEW_A, MSG])
    show("D  system, system:view, user:msg", [S, VIEW_S, MSG])


if __name__ == "__main__":
    main()

"""Read-only access to raw messages and built summary children."""

from __future__ import annotations

import json
from pathlib import Path

from .media import media_for
from .store import Log
from .tree import Tree
from .view import View


def _error(message: str) -> str:
    return json.dumps({"error": message})


def zoom(log: Log, tree: Tree, view: View, id: int | str, n: int = 1,
         *, master_root: Path | None = None) -> str:
    """Open a child chat, a named span, or a raw message."""
    if isinstance(id, str):
        root = Path(master_root) if master_root is not None else log.root
        try:
            index = json.loads((root / "agents" / "index.json").read_text(encoding="utf-8"))
            key = index.get(id) if isinstance(index, dict) else None
            if not isinstance(key, str) or not key or Path(key).name != key or key in (".", ".."):
                return _error("unknown subagent name")
            child_root = root / "agents" / key
            if not (child_root / "view.json").is_file():
                return _error("subagent chat does not exist")
            child_view = View(Tree(child_root))
            return json.dumps({"name": id, "chat": child_view.render()}, ensure_ascii=False)
        except (OSError, ValueError, TypeError):
            return _error("unknown subagent name")
    if type(id) is not int or type(n) is not int or id < 0 or n < 1:
        return _error("id must be a string or nonnegative integer; n must be a positive integer")
    if n & (n - 1):
        return _error("n must be a power of two")
    if id % n:
        return _error("id must be a multiple of n")
    if n == 1:
        record = log.get(id)
        if record is None:
            return _error("message does not exist")
        return json.dumps({"id": id, "n": 1, "kind": record.kind,
                           "text": record.text, "media": media_for(log.root, id)},
                          ensure_ascii=False)

    # A named node covers [id, id+n). Both children must have been built.
    if id + n > log.count():
        return _error("span does not exist")
    level = n.bit_length() - 2
    if tree.get(level + 1, id // n) is None:
        return _error("span is not summarized yet")
    children = [tree.get(level, id // (n // 2)),
                tree.get(level, id // (n // 2) + 1)]
    if any(line is None for line in children):
        return _error("span is not summarized yet")
    halves = [{"id": id + offset, "n": n // 2, "text": line.text}
              for offset, line in zip((0, n // 2), children)]
    return json.dumps({"id": id, "n": n, "halves": halves}, ensure_ascii=False)


def date(log: Log, id: int) -> str:
    """Return the stored ISO timestamp for a raw message."""
    if type(id) is not int or id < 0:
        return _error("id must be a nonnegative integer")
    record = log.get(id)
    if record is None:
        return _error("message does not exist")
    return json.dumps({"id": id, "date": record.date})

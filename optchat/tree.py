"""Build-once summary nodes and their incremental ready queue."""

from __future__ import annotations

import bisect
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .persistence import read_jsonl


@dataclass(frozen=True)
class Line:
    l: int
    i: int
    text: str
    size: int


class Tree:
    def __init__(self, root: Path, node_bytes: int = 512):
        if node_bytes <= 0:
            raise ValueError("node_bytes must be positive")
        self.root = Path(root)
        self.node_bytes = node_bytes
        self._nodes: dict[tuple[int, int], Line] = {}
        self._ready: list[tuple[int, int]] = []
        self._ready_set: set[tuple[int, int]] = set()
        self._pending: deque[Line] = deque()
        self.load()

    def _insert(self, line: Line, *, pending: bool) -> None:
        key = (line.l, line.i)
        if key in self._nodes:
            raise ValueError(f"Node already built: {key}")
        if line.l > 0 and not all(child in self._nodes for child in self.children(*key)):
            raise ValueError(f"Node children are missing: {key}")
        self._nodes[key] = line
        if pending:
            self._pending.append(line)
        if key in self._ready_set:
            self._ready_set.remove(key)
            self._ready.pop(bisect.bisect_left(self._ready, key))
        parent = (line.l + 1, line.i // 2)
        if parent not in self._nodes and parent not in self._ready_set:
            if all(child in self._nodes for child in self.children(*parent)):
                bisect.insort(self._ready, parent)
                self._ready_set.add(parent)

    def put(self, l: int, i: int, text: str) -> None:
        if not isinstance(l, int) or not isinstance(i, int) or l < 0 or i < 0:
            raise ValueError("Node coordinates must be nonnegative integers")
        if not isinstance(text, str):
            raise TypeError("Node text must be a string")
        self._insert(Line(l, i, text, len(text.encode("utf-8"))), pending=True)

    def get(self, l: int, i: int) -> Line | None:
        return self._nodes.get((l, i))

    def nodes(self) -> dict[tuple[int, int], Line]:
        """A copy of every built node, keyed by ``(level, index)``.

        A copy, not the live mapping: callers must not be able to write into the
        tree's internals.
        """
        return dict(self._nodes)

    def children(self, l: int, i: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if l < 1 or i < 0:
            raise ValueError("Only positive-level nodes have children")
        return (l - 1, 2 * i), (l - 1, 2 * i + 1)

    def free_text(self, l: int, i: int) -> str | None:
        if l == 0:
            line = self.get(l, i)
            return line.text if line is not None and line.size <= self.node_bytes else None
        left_key, right_key = self.children(l, i)
        left, right = self.get(*left_key), self.get(*right_key)
        if left is None or right is None:
            return None
        joined = left.text + "\n" + right.text
        return joined if len(joined.encode("utf-8")) <= self.node_bytes else None

    def buildable(self, count: int | None = None) -> list[tuple[int, int]]:
        if count is not None and count < 0:
            raise ValueError("count must be nonnegative")
        if count is None:
            return self._ready.copy()
        return [key for key in self._ready if (key[1] + 1) * (1 << key[0]) <= count]

    def needs_summary(self, l: int, i: int) -> bool:
        return self.get(l, i) is None and self.free_text(l, i) is None

    def save(self) -> None:
        if not self._pending:
            return
        date = datetime.now(timezone.utc).date().isoformat()
        path = self.root / "tree" / f"{date}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
            stream.seek(0, os.SEEK_END)
            while self._pending:
                line = self._pending[0]
                payload = (json.dumps(line.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                self._pending.popleft()

    def load(self) -> None:
        if self._pending:
            raise ValueError("Save pending nodes before loading")
        self._nodes.clear()
        self._ready.clear()
        self._ready_set.clear()
        restored: dict[tuple[int, int], Line] = {}
        for path in sorted((self.root / "tree").glob("????-??-??.jsonl")):
            # Same tail rule as the log: a node missing only its newline is kept
            # (read_jsonl seals it), a sealed fragment is skipped, and mid-file
            # corruption raises.
            for _offset, data in read_jsonl(path):
                try:
                    line = Line(**data)
                    valid = (
                        isinstance(line.l, int)
                        and isinstance(line.i, int)
                        and line.l >= 0
                        and line.i >= 0
                        and isinstance(line.text, str)
                        and line.size == len(line.text.encode("utf-8"))
                    )
                except (TypeError, KeyError):
                    continue
                if not valid:
                    continue
                key = (line.l, line.i)
                previous = restored.get(key)
                if previous is not None and previous.text != line.text:
                    # Build-once is the invariant; a second, different text for
                    # the same node means one of them is not the built line.
                    raise ValueError(f"Tree node {key} was written twice with different text")
                restored[key] = line
        for line in sorted(restored.values(), key=lambda item: (item.l, item.i)):
            try:
                self._insert(line, pending=False)
            except ValueError:
                continue

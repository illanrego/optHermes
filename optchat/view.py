"""Persistent, oldest-first cover of the chat by tree nodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .tree import Line, Tree


PLACEHOLDER = "(not summarized yet: zoom it)"


def due(l: int, i: int, T: int) -> float:
    """Return the age of sibling nodes (l, i) and (l, i + 1)."""
    return (T - ((i + 2) * (1 << l) - 1)) / (1 << l)


class View:
    def __init__(self, tree: Tree, min_bytes: int = 64_000, max_bytes: int = 128_000):
        if min_bytes < 0 or max_bytes <= 0 or min_bytes > max_bytes:
            raise ValueError("View budgets must satisfy 0 <= min_bytes <= max_bytes")
        self.tree = tree
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self._pairs: list[tuple[int, int]] = []
        self.load()

    def add_root(self, i: int) -> None:
        if i != self._end():
            raise ValueError("Root id must append the next message")
        self._pairs.append((0, i))

    def _end(self) -> int:
        if not self._pairs:
            return 0
        l, i = self._pairs[-1]
        return (i + 1) << l

    def lines(self) -> list[Line]:
        result = []
        for l, i in self._pairs:
            line = self.tree.get(l, i)
            if line is None:
                result.append(Line(l, i, PLACEHOLDER, len(PLACEHOLDER.encode("utf-8"))))
            else:
                result.append(line)
        return result

    def render(self) -> str:
        body = [f"{line.i << line.l}+{1 << line.l}|{line.text.replace(chr(10), ' ').replace(chr(13), ' ')}" for line in self.lines()]
        return "<chat>\n" + "\n".join(body) + ("\n" if body else "") + "</chat>"

    def _size(self) -> int:
        return len(self.render().encode("utf-8"))

    def merge_batch(self, T: int) -> None:
        if T != self._end():
            raise ValueError("Message count does not match view coverage")
        if self._size() <= self.max_bytes:
            return
        while self._size() > self.min_bytes:
            best: tuple[float, int, int] | None = None
            for pos in range(len(self._pairs) - 1):
                l, i = self._pairs[pos]
                if i % 2 or self._pairs[pos + 1] != (l, i + 1):
                    continue
                if self.tree.get(l + 1, i // 2) is None:
                    continue
                rank = (due(l, i, T), -(i << l), pos)
                if best is None or rank > best:
                    best = rank
            if best is None:
                break
            pos = best[2]
            l, i = self._pairs[pos]
            self._pairs[pos:pos + 2] = [(l + 1, i // 2)]

    def save(self) -> None:
        path = Path(self.tree.root) / "view.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("view.json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump([list(pair) for pair in self._pairs], stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load(self) -> None:
        path = Path(self.tree.root) / "view.json"
        if not path.exists():
            # The tree is the only source of known nodes on first creation.
            nodes = self.tree.nodes()
            end = max(((i + 1) << l for l, i in nodes), default=0)
            pairs = []
            cursor = 0
            while cursor < end:
                choices = ((l, i) for l, i in nodes if (i << l) == cursor and ((i + 1) << l) <= end)
                pair = max(choices, key=lambda item: item[0], default=(0, cursor))
                pairs.append(pair)
                cursor += 1 << pair[0]
            self._pairs = pairs
            self.save()
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Invalid view.json")
        pairs = []
        cursor = 0
        for pair in raw:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int or value < 0 for value in pair)):
                raise ValueError("Invalid view.json pair")
            l, i = pair
            if (i << l) != cursor:
                raise ValueError("view.json does not cover a contiguous prefix")
            pairs.append((l, i))
            cursor += 1 << l
        self._pairs = pairs

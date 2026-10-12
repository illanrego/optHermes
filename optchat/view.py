"""Persistent, oldest-first cover of the chat by tree nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .persistence import atomic_json
from .tree import Line, Tree


PLACEHOLDER = "(not summarized yet: zoom it)"


@dataclass
class _CompactionView:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    batch_pending: bool = False
    # Tree lines are immutable, so retaining pairs also retains exact request bytes.
    contexts: dict[str, tuple[int, list[tuple[int, int]]]] = field(default_factory=dict)


def due(l: int, i: int, T: int) -> float:
    """Return the age of sibling nodes (l, i) and (l, i + 1)."""
    return (T - ((i + 2) * (1 << l) - 1)) / (1 << l)


class ViewStateUnavailable(RuntimeError):
    """The view state cannot be served because it is missing, and it is never rebuilt.

    A rebuilt view is a different state wearing the chat's name: the previous pair
    selection, the batch state and the retained compaction contexts are not recoverable
    from the log or the tree. Nothing is written by this outcome; restoring view.json
    from a backup is what makes the chat serviceable again.
    """


class View:
    def __init__(self, tree: Tree, min_bytes: int = 64_000, max_bytes: int = 128_000,
                 *, records: int | None = None):
        if min_bytes < 0 or max_bytes <= 0 or min_bytes > max_bytes:
            raise ValueError("View budgets must satisfy 0 <= min_bytes <= max_bytes")
        if records is not None and (type(records) is not int or records < 0):
            raise ValueError("records must be a nonnegative integer")
        self.tree = tree
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        # The chat's record count when its owner knows it: a missing view.json on a
        # chat that has records (or saved tree nodes, when the caller cannot know
        # the count) is an explicit unavailable state, never a rebuild. ``None``
        # when the caller cannot know (a foreign read).
        self._records = records
        self._pairs: list[tuple[int, int]] = []
        self._batch_pending = False
        self._compaction_views: dict[tuple[int, int], _CompactionView] = {}
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
        return self._lines(self._pairs)

    def settled(self) -> bool:
        """Return whether every live view line has a built summary."""
        return all(line.text != PLACEHOLDER for line in self.lines())

    def _lines(self, pairs: list[tuple[int, int]]) -> list[Line]:
        result = []
        for l, i in pairs:
            line = self.tree.get(l, i)
            if line is None:
                result.append(Line(l, i, PLACEHOLDER, len(PLACEHOLDER.encode("utf-8"))))
            else:
                result.append(line)
        return result

    def render(self) -> str:
        return self._render(self._pairs)

    def _render(self, pairs: list[tuple[int, int]]) -> str:
        body = [f"{line.i << line.l}+{1 << line.l}|{line.text.replace(chr(10), ' ').replace(chr(13), ' ')}" for line in self._lines(pairs)]
        return "<chat>\n" + "\n".join(body) + ("\n" if body else "") + "</chat>"

    def _size(self, pairs: list[tuple[int, int]]) -> int:
        return len(self._render(pairs).encode("utf-8"))

    def context_view(self, min_bytes: int = 16_000, max_bytes: int = 32_000,
                     *, end: int | None = None, target: tuple[int, int] | None = None) -> str:
        """Return a maintained, placeholder-free prefix ending before ``end``.

        The first request for each target pins its built pairs. Later requests for
        that target reuse those exact lines, including after save/load. The working
        compaction view advances independently of both these snapshots and the live view.
        """
        if min_bytes < 0 or max_bytes <= 0 or min_bytes > max_bytes:
            raise ValueError("View budgets must satisfy 0 <= min_bytes <= max_bytes")
        if target is not None:
            l, i = target
            if any(type(value) is not int or value < 0 for value in target):
                raise ValueError("Compaction target must have nonnegative coordinates")
            boundary = i if l == 0 else (i + 1) << l
            if end is not None and end != boundary:
                raise ValueError("Compaction boundary does not match target")
            end = boundary
        if end is None:
            end = self._end()
        if type(end) is not int or end < 0:
            raise ValueError("Compaction boundary must be a nonnegative integer")
        state = self._compaction_views.setdefault((min_bytes, max_bytes), _CompactionView())
        key = f"node:{target[0]}:{target[1]}" if target is not None else f"end:{end}"
        if key in state.contexts:
            return self._render(state.contexts[key][1])

        cursor = sum(1 << l for l, _ in state.pairs)
        while cursor < end and self.tree.get(0, cursor) is not None:
            state.pairs.append((0, cursor))
            cursor += 1
        state.batch_pending = self._merge_pairs(
            state.pairs, cursor, min_bytes, max_bytes, state.batch_pending)

        # A late request can precede the working frontier. Split only nodes that
        # straddle its boundary: using their parent text would leak later messages.
        pairs: list[tuple[int, int]] = []

        def prefix(l: int, i: int) -> None:
            if (i << l) >= end:
                return
            if ((i + 1) << l) <= end:
                pairs.append((l, i))
            else:
                for child in self.tree.children(l, i):
                    prefix(*child)

        for pair in state.pairs:
            prefix(*pair)
        # Splitting a historical boundary can expose more lines than its budget.
        # This is done once; the resulting snapshot is never recomputed.
        self._merge_pairs(pairs, min(cursor, end), min_bytes, max_bytes)
        state.contexts[key] = end, pairs
        return self._render(pairs)

    def merge_batch(self, T: int) -> None:
        if T != self._end():
            raise ValueError("Message count does not match view coverage")
        self._batch_pending = self._merge_pairs(
            self._pairs, T, self.min_bytes, self.max_bytes, self._batch_pending)
        for (minimum, maximum), state in self._compaction_views.items():
            end = sum(1 << l for l, _ in state.pairs)
            state.batch_pending = self._merge_pairs(
                state.pairs, end, minimum, maximum, state.batch_pending)

    def _merge_pairs(self, pairs: list[tuple[int, int]], T: int,
                     min_bytes: int, max_bytes: int, pending: bool = False) -> bool:
        if not pending and self._size(pairs) <= max_bytes:
            return False
        while self._size(pairs) > min_bytes:
            best: tuple[float, int, int] | None = None
            for pos in range(len(pairs) - 1):
                l, i = pairs[pos]
                if i % 2 or pairs[pos + 1] != (l, i + 1):
                    continue
                if self.tree.get(l + 1, i // 2) is None:
                    continue
                rank = (due(l, i, T), -(i << l), pos)
                if best is None or rank > best:
                    best = rank
            if best is None:
                return True
            pos = best[2]
            l, i = pairs[pos]
            pairs[pos:pos + 2] = [(l + 1, i // 2)]
        return False

    def save(self) -> None:
        if self.tree._read_only:
            # A view read through a foreign tree's read handle must not write
            # that chat's view.json without its lock.
            raise ValueError("View is read-only")
        path = Path(self.tree.root) / "view.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # The same durable replacement a checkpoint publication uses: fsync'd
        # temporary, atomic rename, and the directory sync that makes the rename
        # itself durable. A view publication is as durable as a checkpoint.
        atomic_json(path, {"version": 2, "pairs": self._pairs,
                           "batch_pending": self._batch_pending,
                           "compaction_views": [
                               {"min_bytes": minimum, "max_bytes": maximum,
                                "pairs": state.pairs, "batch_pending": state.batch_pending,
                                "contexts": state.contexts}
                               for (minimum, maximum), state in self._compaction_views.items()
                           ]})

    def load(self) -> None:
        path = Path(self.tree.root) / "view.json"
        if not path.exists():
            nodes = self.tree.nodes()
            if not nodes and not self._records:
                # A chat with no records: nothing was saved, nothing is lost, and
                # the empty view is the correct view. Built silently; a foreign
                # (read-only) read does not write it.
                self._pairs = []
                if not self.tree._read_only:
                    self.save()
                return
            # Missing state in a chat that has records (or saved tree nodes, when
            # the caller cannot know the record count) is an explicit unavailable
            # state. The view is never rebuilt: a rebuilt view is a different
            # state, and the previous pair selection, batch state and retained
            # compaction contexts are not recoverable from the log or the tree.
            # Nothing is written, so a restored view.json resumes the chat.
            if self._records:
                detail = f"and this chat has {self._records} record(s)"
            else:
                detail = "and saved tree nodes exist for this chat"
            raise ViewStateUnavailable(
                f"OptChat view state is unavailable for {self.tree.root}: "
                f"view.json is missing {detail}. The view is never rebuilt - a "
                "rebuilt view is a different state (pair selection, batch state "
                "and retained compaction contexts are not recoverable), and "
                "serving it would silently replace this chat's memory. Nothing "
                "was written; restore view.json from a backup to resume this chat.")
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._compaction_views = {}
        self._batch_pending = False
        if isinstance(raw, list):
            self._pairs = self._read_pairs(raw)
            return
        if not isinstance(raw, dict) or raw.get("version") != 2:
            raise ValueError("Invalid view.json version")
        self._pairs = self._read_pairs(raw["pairs"])
        self._batch_pending = self._read_pending(raw["batch_pending"])
        # A legacy "recovered" marker (written only by the retired rebuild path)
        # is ignored; nothing reads or writes it anymore.
        for item in raw["compaction_views"]:
            minimum, maximum = item["min_bytes"], item["max_bytes"]
            if (type(minimum) is not int or type(maximum) is not int
                    or minimum < 0 or maximum <= 0 or minimum > maximum):
                raise ValueError("Invalid compaction view budgets")
            state = _CompactionView(
                self._read_pairs(item["pairs"], built=True),
                self._read_pending(item["batch_pending"]))
            for key, (end, pairs) in item["contexts"].items():
                context = self._read_pairs(pairs, built=True)
                if type(end) is not int or end < 0 or sum(1 << l for l, _ in context) > end:
                    raise ValueError("Compaction context exceeds its boundary")
                state.contexts[key] = end, context
            self._compaction_views[minimum, maximum] = state

    @staticmethod
    def _read_pending(raw: object) -> bool:
        if type(raw) is not bool:
            raise ValueError("Invalid view batch state")
        return raw

    def _read_pairs(self, raw: object, *, built: bool = False) -> list[tuple[int, int]]:
        if not isinstance(raw, list):
            raise ValueError("Invalid view.json pairs")
        pairs = []
        cursor = 0
        for pair in raw:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int or value < 0 for value in pair)):
                raise ValueError("Invalid view.json pair")
            l, i = pair
            if (i << l) != cursor:
                raise ValueError("view.json does not cover a contiguous prefix")
            if built and self.tree.get(l, i) is None:
                raise ValueError("Compaction view contains an unbuilt line")
            pairs.append((l, i))
            cursor += 1 << l
        return pairs

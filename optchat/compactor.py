"""Queue-driven, build-once compactions."""

from __future__ import annotations

import heapq
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

from .prompts import (COMPACTION_MERGE_TASK, COMPACTION_SYSTEM, COMPACTION_TASK,
                      OVER_LENGTH_NOTICE, RUBRIC)
from .tree import Tree
from .view import View


@dataclass(frozen=True)
class CompactionRequest:
    system: str
    context: str
    task: str
    tools: list


Summarizer = Callable[[CompactionRequest], str]


class Compactor:
    def __init__(self, tree: Tree, summarize: Summarizer, *,
                 node_bytes: int = 512, max_retries: int = 5, view: View | None = None):
        if node_bytes <= 0 or max_retries <= 0:
            raise ValueError("Compaction budgets must be positive")
        self.tree = tree
        self.summarize = summarize
        self.node_bytes = node_bytes
        self.max_retries = max_retries
        self.view = view
        self.system: str = ""
        self.tools: list = []
        self._pending: list[tuple[int, int]] = []
        self._queued: set[tuple[int, int]] = set()
        self._sources: dict[int, str] = {}
        for key in tree.buildable():
            self.submit(*key)

    def seed(self, i: int, text: str) -> None:
        if not isinstance(i, int) or i < 0 or not isinstance(text, str):
            raise ValueError("A seed needs a nonnegative id and text")
        if self.tree.get(0, i) is not None or i in self._sources:
            raise ValueError(f"Message {i} was already seeded")
        if len(text.encode("utf-8")) <= self.node_bytes:
            self.tree.put(0, i, text)
            self._submit_ready_parent(0, i)
        else:
            self._sources[i] = text
            self.submit(0, i)

    def submit(self, l: int, i: int) -> None:
        if not isinstance(l, int) or not isinstance(i, int) or l < 0 or i < 0:
            raise ValueError("Node coordinates must be nonnegative integers")
        key = (l, i)
        if self.tree.get(l, i) is None and key not in self._queued:
            heapq.heappush(self._pending, key)
            self._queued.add(key)

    def _submit_ready_parent(self, l: int, i: int) -> None:
        parent = (l + 1, i // 2)
        if self.tree.get(*parent) is None and all(
            self.tree.get(l, sibling) is not None for sibling in (i & ~1, i | 1)
        ):
            self.submit(*parent)

    def _request(self, l: int, i: int) -> CompactionRequest | None:
        if l == 0:
            source = self._sources.get(i)
            if source is None:
                return None
            template = COMPACTION_TASK
        else:
            left, right = (self.tree.get(*key) for key in self.tree.children(l, i))
            if left is None or right is None:
                return None
            source = left.text + "\n" + right.text
            template = COMPACTION_MERGE_TASK
        task = template.format(id=i << l, count=1 << l, node_bytes=self.node_bytes,
                               ruler="-" * self.node_bytes, input=source)
        return CompactionRequest(
            system=self.system,
            context=self.view.context_view() if self.view is not None else "",
            task=COMPACTION_SYSTEM.format(rubric=RUBRIC) + "\n\n" + task,
            tools=self.tools,
        )

    def _summarize(self, request: CompactionRequest, key: tuple[int, int]) -> str | None:
        shortest: str | None = None
        task = request.task
        for _ in range(self.max_retries):
            try:
                reply = self.summarize(replace(request, task=task))
            except Exception as exc:
                warnings.warn(f"Compaction {key} failed: {exc}", RuntimeWarning, stacklevel=2)
                return None
            if not isinstance(reply, str) or not reply.strip():
                warnings.warn(f"Compaction {key} returned an empty reply", RuntimeWarning, stacklevel=2)
                return None
            line = " ".join(reply.splitlines()).strip()
            length = len(line.encode("utf-8"))
            if shortest is None or length < len(shortest.encode("utf-8")):
                shortest = line
            if length <= self.node_bytes:
                return shortest
            prefix = line.encode("utf-8")[:self.node_bytes].decode("utf-8", errors="ignore")
            task = request.task + "\n\n" + OVER_LENGTH_NOTICE.format(
                actual_bytes=length, node_bytes=self.node_bytes, cut_point=prefix)
        return shortest

    def run_pending(self, limit: int = 8) -> int:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        built = 0
        deferred: list[tuple[int, int]] = []
        while self._pending and built < limit:
            l, i = heapq.heappop(self._pending)
            self._queued.remove((l, i))
            if self.tree.get(l, i) is not None:
                continue
            if l > 0 and any(self.tree.get(*child) is None for child in self.tree.children(l, i)):
                deferred.append((l, i))
                continue
            free = self.tree.free_text(l, i)
            if free is None:
                request = self._request(l, i)
                if request is None:
                    warnings.warn(f"Compaction {(l, i)} has no source", RuntimeWarning, stacklevel=2)
                    deferred.append((l, i))
                    continue
                result = self._summarize(request, (l, i))
                if result is None:
                    deferred.append((l, i))
                    continue
            else:
                result = free
            self.tree.put(l, i, result)
            if l == 0:
                self._sources.pop(i, None)
            built += 1
            self._submit_ready_parent(l, i)
        for key in deferred:
            self.submit(*key)
        return built

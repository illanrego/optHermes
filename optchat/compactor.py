"""Queue-driven, build-once compactions."""

from __future__ import annotations

import heapq
import time
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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
COMPACT_CONCURRENCY = 8


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

    def settle(self, max_nodes: int = 64, timeout_s: float = 20.0, *,
               model_calls: bool = False) -> int:
        """Build pending nodes until the view settles or admission is bounded.

        ``model_calls`` defaults to False: the turn path calls this synchronously, and a
        summarizer call already in flight cannot be interrupted by the deadline, so a turn
        must never start one. Model work is left to the background ``run_pending``.
        """
        built = 0
        try:
            deadline = time.monotonic() + timeout_s
            while built < max_nodes and time.monotonic() < deadline:
                if self.view is not None and self.view.settled():
                    break
                progress = self.run_pending(limit=1, deadline=deadline, model_calls=model_calls)
                if progress == 0:
                    break
                built += progress
        except Exception as exc:
            warnings.warn(f"Settle failed: {exc}", RuntimeWarning, stacklevel=2)
        return built

    def run_pending(self, limit: int = COMPACT_CONCURRENCY, *, deadline: float | None = None,
                    model_calls: bool = True) -> int:
        """Build at most ``limit`` nodes, with up to ``limit`` model calls in flight.

        ``model_calls=False`` builds only nodes that need no model (``free_text``) and
        defers the rest. A synchronous caller must use it: the deadline is checked between
        nodes and cannot interrupt a summarizer call already in flight.
        """
        if not isinstance(limit, int) or limit < 0:
            warnings.warn("Compaction limit must be a nonnegative integer",
                          RuntimeWarning, stacklevel=2)
            return 0
        if limit == 0:
            return 0
        built = 0
        deferred: list[tuple[int, int]] = []
        try:
            with ThreadPoolExecutor(max_workers=limit) as executor:
                while (self._pending and built < limit
                       and (deadline is None or time.monotonic() < deadline)):
                    batch: list[tuple[tuple[int, int], Future[str | None]]] = []
                    while (self._pending and built + len(batch) < limit
                           and (deadline is None or time.monotonic() < deadline)):
                        key = heapq.heappop(self._pending)
                        self._queued.discard(key)
                        l, i = key
                        try:
                            if self.tree.get(l, i) is not None:
                                continue
                            if l > 0 and any(self.tree.get(*child) is None
                                             for child in self.tree.children(l, i)):
                                deferred.append(key)
                                continue
                            free = self.tree.free_text(l, i)
                            if free is not None:
                                self.tree.put(l, i, free)
                                built += 1
                                self._submit_ready_parent(l, i)
                                continue
                            if not model_calls:
                                deferred.append(key)
                                continue
                            request = self._request(l, i)
                            if request is None:
                                warnings.warn(f"Compaction {key} has no source",
                                              RuntimeWarning, stacklevel=2)
                                deferred.append(key)
                                continue
                            batch.append((key, executor.submit(self._summarize, request, key)))
                        except Exception as exc:
                            warnings.warn(f"Compaction {key} failed: {exc}",
                                          RuntimeWarning, stacklevel=2)
                            deferred.append(key)

                    # All requests in this batch are already running. Applying their
                    # results in queue order makes admission independent of finish order.
                    for key, future in batch:
                        try:
                            result = future.result()
                            if result is None:
                                deferred.append(key)
                                continue
                            l, i = key
                            self.tree.put(l, i, result)
                            if l == 0:
                                self._sources.pop(i, None)
                            built += 1
                            self._submit_ready_parent(l, i)
                        except Exception as exc:
                            warnings.warn(f"Compaction {key} failed: {exc}",
                                          RuntimeWarning, stacklevel=2)
                            deferred.append(key)
        except Exception as exc:
            warnings.warn(f"Compaction pass failed: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            for key in deferred:
                try:
                    self.submit(*key)
                except Exception as exc:
                    warnings.warn(f"Compaction {key} could not be requeued: {exc}",
                                  RuntimeWarning, stacklevel=2)
        return built

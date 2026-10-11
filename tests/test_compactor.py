import re
import threading
import time

import pytest

from optchat.compactor import Compactor
from optchat.tree import Tree
from optchat.view import View


def test_seed_short_is_free_and_long_uses_one_call(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return "short"

    compactor = Compactor(tree, summarize, node_bytes=8)
    compactor.seed(0, "é", "user")
    assert tree.get(0, 0).text == "user: é"
    assert tree.get(0, 0).size == len("user: é".encode("utf-8")) == 8
    assert prompts == []
    compactor.seed(1, "éé", "user")
    assert tree.get(0, 1) is None
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(0, 1).text == "short"
    assert len(prompts) == 1
    assert "--------" in prompts[0].task
    assert "<input>\nuser: éé\n</input>" in prompts[0].task


def test_kind_prefix_pushes_source_over_free_node_limit(tmp_path):
    tree = Tree(tmp_path, node_bytes=10)
    requests = []
    compactor = Compactor(tree, lambda request: requests.append(request) or "short",
                          node_bytes=10)

    compactor.seed(0, "hello", "user")

    assert tree.get(0, 0) is None
    assert compactor.run_pending(limit=1, model_calls=False) == 0
    assert compactor.run_pending(limit=1) == 1
    assert "<input>\nuser: hello\n</input>" in requests[0].task


def test_seed_requires_explicit_kind(tmp_path):
    compactor = Compactor(Tree(tmp_path), lambda _request: "summary")
    with pytest.raises(TypeError, match="kind"):
        compactor.seed(0, "hello")


def test_free_parent_and_long_parent(tmp_path):
    tree = Tree(tmp_path, node_bytes=16)
    calls = []
    compactor = Compactor(tree, lambda prompt: calls.append(prompt) or "merged", node_bytes=16)
    compactor.seed(0, "a", "user")
    compactor.seed(1, "b", "user")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0).text == "user: a\nuser: b"
    assert calls == []
    compactor.seed(2, "12345", "user")
    compactor.seed(3, "67890", "user")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 1).text == "merged"
    assert len(calls) == 1
    assert "merge lines 2+1 and 3+1, adjacent" in calls[0].task
    assert "their messages, 2 to 3, in more detail" in calls[0].task
    assert "<input>\nuser: 12345\nuser: 67890\n</input>" in calls[0].task
    assert tree.get(2, 0) is None
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(2, 0) is not None
    assert "merge lines 0+2 and 2+2, adjacent" in calls[1].task
    assert "their messages, 0 to 3, in more detail" in calls[1].task
    assert all(tree.get(*child) for child in tree.children(2, 0))


def test_oversize_retry_keeps_shortest(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    replies = iter(["x" * 20, "short"])
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return next(replies)

    compactor = Compactor(tree, summarize, node_bytes=8)
    compactor.seed(0, "long source", "user")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(0, 0).text == "short"
    assert len(prompts) == 2
    assert "20 bytes" in prompts[1].task
    assert "xxxxxxxx" in prompts[1].task
    assert prompts[1].context == prompts[0].context


@pytest.mark.parametrize("summarize", [lambda _prompt: "", lambda _prompt: (_ for _ in ()).throw(RuntimeError("offline"))])
def test_failed_summary_is_reported_and_unbuilt(tmp_path, summarize):
    tree = Tree(tmp_path, node_bytes=4)
    compactor = Compactor(tree, summarize, node_bytes=4)
    compactor.seed(0, "too long", "user")
    with pytest.warns(RuntimeWarning):
        assert compactor.run_pending(limit=1) == 0
    assert tree.get(0, 0) is None


def test_four_independent_summaries_run_at_once(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    barrier = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def summarize(_request):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait()
            return "short"
        except threading.BrokenBarrierError as exc:
            raise AssertionError("Four summarizer calls did not meet the 5-second barrier") from exc
        finally:
            with lock:
                active -= 1

    compactor = Compactor(tree, summarize, node_bytes=8)
    for i in range(4):
        compactor.seed(i, "long source", "user")

    assert compactor.run_pending(limit=4) == 4, "Four calls must meet the 5-second barrier"
    assert maximum_active == 4
    assert all(tree.get(0, i) is not None for i in range(4))


def test_limit_does_not_change_the_built_nodes(tmp_path):
    snapshots = []
    for limit in (1, 4):
        tree = Tree(tmp_path / str(limit), node_bytes=6)
        compactor = Compactor(tree, lambda _request: "abcd", node_bytes=6)
        for i in range(4):
            compactor.seed(i, "long source", "user")
        assert sum(compactor.run_pending(limit=limit) for _ in range(7)) == 7
        snapshots.append(tree.nodes())

    assert snapshots[0] == snapshots[1]
    assert set(snapshots[0]) == {(0, i) for i in range(4)} | {(1, 0), (1, 1), (2, 0)}


def test_one_pass_builds_the_same_nodes_regardless_of_finish_order(tmp_path):
    snapshots = []
    for delayed_pair in (0, 1):
        tree = Tree(tmp_path / str(delayed_pair), node_bytes=6)

        def summarize(request):
            message = re.search(r"compress message (\d+)", request.task)
            if message and int(message.group(1)) // 2 == delayed_pair:
                time.sleep(0.02)
            return "abcd"

        compactor = Compactor(tree, summarize, node_bytes=6)
        for i in range(4):
            compactor.seed(i, "long source", "user")
        assert compactor.run_pending(limit=5) == 5
        snapshots.append(tree.nodes())

    assert snapshots[0] == snapshots[1]
    assert set(snapshots[0]) == {(0, i) for i in range(4)} | {(1, 0)}


def test_parent_request_follows_both_child_requests(tmp_path):
    tree = Tree(tmp_path, node_bytes=6)
    seen = []
    lock = threading.Lock()

    def summarize(request):
        merge = re.search(r"merge lines (\d+)\+(\d+) and (\d+)\+(\d+), adjacent",
                          request.task)
        if merge:
            start, count, right, right_count = map(int, merge.groups())
            assert (right, right_count) == (start + count, count)
            key = (count.bit_length(), start // (2 * count))
            assert all(tree.get(*child) is not None for child in tree.children(*key))
        else:
            message = re.search(r"compress message (\d+)", request.task)
            assert message is not None
            key = (0, int(message.group(1)))
        with lock:
            seen.append(key)
        return "abcd"

    compactor = Compactor(tree, summarize, node_bytes=6)
    for i in range(4):
        compactor.seed(i, "long source", "user")
    assert sum(compactor.run_pending(limit=4) for _ in range(3)) == 7

    for parent in ((1, 0), (1, 1), (2, 0)):
        assert all(seen.index(child) < seen.index(parent) for child in tree.children(*parent))


def test_concurrent_summary_failure_leaves_only_affected_child_unbuilt(tmp_path):
    tree = Tree(tmp_path, node_bytes=6)

    def summarize(request):
        if "compress message 0" in request.task:
            raise RuntimeError("offline")
        return "abcd"

    compactor = Compactor(tree, summarize, node_bytes=6)
    for i in range(4):
        compactor.seed(i, "long source", "user")
    with pytest.warns(RuntimeWarning, match="offline"):
        compactor.run_pending(limit=4)
    assert tree.get(0, 0) is None
    assert all(tree.get(0, i) is not None for i in range(1, 4))
    assert tree.get(1, 0) is None


def test_unready_parent_waits_for_both_children(tmp_path):
    tree = Tree(tmp_path, node_bytes=10)
    compactor = Compactor(tree, lambda _prompt: "done", node_bytes=10)
    compactor.submit(1, 0)
    assert compactor.run_pending() == 0
    compactor.seed(0, "long source", "user")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0) is None
    compactor.seed(1, "yes", "user")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0) is not None


def test_existing_ready_queue_is_resumed(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    tree.put(0, 0, "a")
    tree.put(0, 1, "b")
    compactor = Compactor(tree, lambda _prompt: pytest.fail("free node used a model"), node_bytes=8)
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0).text == "a\nb"


def test_request_carries_turn_prefix_context_and_task(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    requests = []
    compactor = Compactor(tree, lambda request: requests.append(request) or "short",
                          node_bytes=8, view=view)
    compactor.system = "turn system"
    compactor.tools = [{"type": "function", "function": {"name": "lookup"}}]
    compactor.seed(0, "long source", "user")
    view.add_root(0)

    assert compactor.run_pending(limit=1) == 1
    request = requests[0]
    assert request.system == "turn system"
    assert request.context.startswith("<chat>\n0+1|")
    assert request.context.endswith("</chat>")
    assert "Compaction: compress message 0" in request.task
    assert "<input>\nuser: long source\n</input>" in request.task
    assert request.tools == compactor.tools


def test_retry_only_changes_task_and_failure_leaves_node_unbuilt(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    requests = []

    def summarize(request):
        requests.append(request)
        if len(requests) == 1:
            return "x" * 20
        raise RuntimeError("offline")

    compactor = Compactor(tree, summarize, node_bytes=8, view=view)
    compactor.seed(0, "long source", "user")
    view.add_root(0)
    with pytest.warns(RuntimeWarning, match="offline"):
        assert compactor.run_pending(limit=1) == 0
    assert tree.get(0, 0) is None
    assert requests[1].context == requests[0].context
    assert "Too long:" not in requests[1].context
    assert "Too long:" in requests[1].task


def test_settle_default_builds_only_free_nodes(tmp_path):
    """The turn path calls settle synchronously, so by default it must never start a model
    call: free nodes are built, anything needing a model is deferred to the background.

    Note what this can and cannot do: a view is unsettled only when a *message* has no node,
    and a short message's node is free at seed time - so the only work settle can do is a
    free parent elsewhere in the tree.
    """
    tree = Tree(tmp_path, node_bytes=16)
    view = View(tree)
    calls = []

    def summarize(_request):
        calls.append(1)
        return "short"

    compactor = Compactor(tree, summarize, node_bytes=16, view=view)
    compactor.seed(0, "long source", "user")   # needs a model
    view.add_root(0)
    compactor.seed(1, "long source", "user")   # needs a model
    view.add_root(1)
    compactor.seed(2, "a", "user")             # free, and its pair with 3 is a free parent
    view.add_root(2)
    compactor.seed(3, "b", "user")
    view.add_root(3)

    assert not view.settled()
    assert compactor.settle() == 1
    assert tree.get(1, 1).text == "user: a\nuser: b"
    assert tree.get(0, 0) is None      # the model nodes are deferred
    assert calls == []                 # no model call was started
    assert not view.settled()


def test_settle_with_model_calls_settles_the_whole_view(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    compactor = Compactor(tree, lambda _request: "short", node_bytes=8, view=view)
    compactor.seed(0, "long source", "user")
    view.add_root(0)

    assert compactor.settle(model_calls=True) == 1
    assert view.settled()
    assert tree.get(0, 0).text == "short"


def test_settle_stops_at_max_nodes(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    compactor = Compactor(tree, lambda _request: "short", node_bytes=8, view=view)
    for i in range(3):
        compactor.seed(i, "long source", "user")
        view.add_root(i)

    assert compactor.settle(max_nodes=1, model_calls=True) == 1
    assert not view.settled()
    assert sum(tree.get(0, i) is not None for i in range(3)) == 1


def test_settle_stops_after_deadline(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    calls = []

    def summarize(_request):
        calls.append(1)
        time.sleep(0.03)
        return "short"

    compactor = Compactor(tree, summarize, node_bytes=8, view=view)
    for i in range(3):
        compactor.seed(i, "long source", "user")
        view.add_root(i)

    assert compactor.settle(timeout_s=0.005, model_calls=True) == 1
    assert len(calls) == 1
    assert not view.settled()


def test_settle_does_not_retry_a_deferred_failure_in_one_call(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    calls = []

    def summarize(_request):
        calls.append(1)
        raise RuntimeError("offline")

    compactor = Compactor(tree, summarize, node_bytes=8, view=view)
    compactor.seed(0, "long source", "user")
    view.add_root(0)
    with pytest.warns(RuntimeWarning, match="offline"):
        assert compactor.settle(model_calls=True) == 0
    assert calls == [1]
    assert tree.get(0, 0) is None
    assert not view.settled()


def test_settle_stops_when_view_has_no_buildable_work(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    view.add_root(0)
    compactor = Compactor(tree, lambda _request: pytest.fail("unexpected model call"),
                          view=view)

    assert compactor.settle() == 0
    assert not view.settled()

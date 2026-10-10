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
    compactor.seed(0, "éééé")
    assert tree.get(0, 0).text == "éééé"
    assert prompts == []
    compactor.seed(1, "ééééé")
    assert tree.get(0, 1) is None
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(0, 1).text == "short"
    assert len(prompts) == 1
    assert "--------" in prompts[0].task


def test_free_parent_and_long_parent(tmp_path):
    tree = Tree(tmp_path, node_bytes=10)
    calls = []
    compactor = Compactor(tree, lambda prompt: calls.append(prompt) or "merged", node_bytes=10)
    compactor.seed(0, "a")
    compactor.seed(1, "b")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0).text == "a\nb"
    assert calls == []
    compactor.seed(2, "12345")
    compactor.seed(3, "67890")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 1).text == "merged"
    assert len(calls) == 1
    assert tree.get(2, 0) is None
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(2, 0) is not None
    assert all(tree.get(*child) for child in tree.children(2, 0))


def test_oversize_retry_keeps_shortest(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    replies = iter(["x" * 20, "short"])
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return next(replies)

    compactor = Compactor(tree, summarize, node_bytes=8)
    compactor.seed(0, "long source")
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
    compactor.seed(0, "too long")
    with pytest.warns(RuntimeWarning):
        assert compactor.run_pending(limit=1) == 0
    assert tree.get(0, 0) is None


def test_unready_parent_waits_for_both_children(tmp_path):
    tree = Tree(tmp_path, node_bytes=4)
    compactor = Compactor(tree, lambda _prompt: "done", node_bytes=4)
    compactor.submit(1, 0)
    assert compactor.run_pending() == 0
    compactor.seed(0, "long source")
    assert compactor.run_pending(limit=1) == 1
    assert tree.get(1, 0) is None
    compactor.seed(1, "yes")
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
    compactor.seed(0, "long source")
    view.add_root(0)

    assert compactor.run_pending(limit=1) == 1
    request = requests[0]
    assert request.system == "turn system"
    assert request.context.startswith("<chat>\n0+1|")
    assert request.context.endswith("</chat>")
    assert "Compaction: compress message 0" in request.task
    assert "long source" in request.task
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
    compactor.seed(0, "long source")
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
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    calls = []

    def summarize(_request):
        calls.append(1)
        return "short"

    compactor = Compactor(tree, summarize, node_bytes=8, view=view)
    compactor.seed(0, "long source")   # needs a model
    view.add_root(0)
    compactor.seed(1, "long source")   # needs a model
    view.add_root(1)
    compactor.seed(2, "a")             # free, and its pair with 3 is a free parent
    view.add_root(2)
    compactor.seed(3, "b")
    view.add_root(3)

    assert not view.settled()
    assert compactor.settle() == 1
    assert tree.get(1, 1).text == "a\nb"
    assert tree.get(0, 0) is None      # the model nodes are deferred
    assert calls == []                 # no model call was started
    assert not view.settled()


def test_settle_with_model_calls_settles_the_whole_view(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    compactor = Compactor(tree, lambda _request: "short", node_bytes=8, view=view)
    compactor.seed(0, "long source")
    view.add_root(0)

    assert compactor.settle(model_calls=True) == 1
    assert view.settled()
    assert tree.get(0, 0).text == "short"


def test_settle_stops_at_max_nodes(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    view = View(tree)
    compactor = Compactor(tree, lambda _request: "short", node_bytes=8, view=view)
    for i in range(3):
        compactor.seed(i, "long source")
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
        compactor.seed(i, "long source")
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
    compactor.seed(0, "long source")
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

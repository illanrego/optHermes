import pytest

from optchat.compactor import Compactor
from optchat.tree import Tree


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
    assert "--------" in prompts[0]


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
    assert "20 bytes" in prompts[1]
    assert "xxxxxxxx" in prompts[1]


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

import json

import pytest

from optchat.tree import Tree


def test_short_leaf_and_free_parent(tmp_path):
    tree = Tree(tmp_path, node_bytes=20)
    tree.put(0, 0, "é")
    assert tree.get(0, 0).size == 2
    assert tree.free_text(0, 0) == "é"
    assert not tree.needs_summary(0, 0)
    assert tree.buildable(2) == []

    tree.put(0, 1, "hello")
    assert tree.children(1, 0) == ((0, 0), (0, 1))
    assert tree.free_text(1, 0) == "é\nhello"
    assert not tree.needs_summary(1, 0)
    assert tree.buildable(2) == [(1, 0)]
    tree.put(1, 0, tree.free_text(1, 0))
    assert tree.buildable() == []


def test_oversize_parent_needs_summary_and_nodes_build_once(tmp_path):
    tree = Tree(tmp_path, node_bytes=8)
    tree.put(0, 0, "abcd")
    tree.put(0, 1, "efgh")
    assert tree.free_text(1, 0) is None
    assert tree.needs_summary(1, 0)
    tree.put(1, 0, "short")
    assert not tree.needs_summary(1, 0)
    with pytest.raises(ValueError, match="already built"):
        tree.put(1, 0, "again")
    with pytest.raises(ValueError, match="children are missing"):
        tree.put(2, 0, "orphan")


def test_buildable_order_and_coverage(tmp_path):
    tree = Tree(tmp_path)
    for i in (3, 2, 1, 0):
        tree.put(0, i, str(i))
    assert tree.buildable() == [(1, 0), (1, 1)]
    assert tree.buildable(2) == [(1, 0)]
    tree.put(1, 1, "2\n3")
    assert tree.buildable(4) == [(1, 0)]
    tree.put(1, 0, "0\n1")
    assert tree.buildable(3) == []
    assert tree.buildable(4) == [(2, 0)]
    assert all(all(tree.get(*child) for child in tree.children(*key)) for key in tree.buildable())


def test_save_load_round_trip_and_torn_tail(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "alpha")
    tree.put(0, 1, "beta")
    tree.put(1, 0, "alpha\nbeta")
    tree.save()

    path = next((tmp_path / "tree").glob("*.jsonl"))
    raw = path.read_bytes()
    assert [json.loads(line)["l"] for line in raw.splitlines()] == [0, 0, 1]
    with path.open("ab") as stream:
        stream.write(b'{"l":2')

    restored = Tree(tmp_path)
    restored.load()
    assert restored.get(1, 0) == tree.get(1, 0)
    assert restored.buildable() == []
    with pytest.raises(ValueError, match="already built"):
        restored.put(0, 0, "different")
    restored.put(0, 2, "gamma")
    restored.save()
    assert path.read_bytes().startswith(raw)
    assert Tree(tmp_path).get(0, 2).text == "gamma"


def test_parseable_torn_tail_node_is_recovered(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "alpha")
    tree.put(0, 1, "beta")
    tree.save()
    path = next((tmp_path / "tree").glob("*.jsonl"))

    # A complete node whose newline never landed: skipping it would lose a node
    # that was already built, and build-once means it must not be rebuilt.
    node = {"l": 1, "i": 0, "text": "alpha\nbeta", "size": len("alpha\nbeta".encode("utf-8"))}
    with path.open("ab") as stream:
        stream.write(json.dumps(node).encode("utf-8"))

    restored = Tree(tmp_path)
    assert restored.get(1, 0).text == "alpha\nbeta"


def test_duplicate_node_with_different_text_is_rejected(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "alpha")
    tree.save()
    path = next((tmp_path / "tree").glob("*.jsonl"))
    with path.open("ab") as stream:
        stream.write(json.dumps({"l": 0, "i": 0, "text": "other", "size": 5}).encode() + b"\n")

    with pytest.raises(ValueError, match="twice with different text"):
        Tree(tmp_path)


def test_identical_duplicate_node_is_tolerated(tmp_path):
    # A crash between writing a node and dequeuing it re-writes the same line on
    # the next save; that is not a build-once violation and must load cleanly.
    tree = Tree(tmp_path)
    tree.put(0, 0, "alpha")
    tree.save()
    path = next((tmp_path / "tree").glob("*.jsonl"))
    with path.open("ab") as stream:
        stream.write(json.dumps({"l": 0, "i": 0, "text": "alpha", "size": 5}).encode() + b"\n")

    assert Tree(tmp_path).get(0, 0).text == "alpha"

"""Read-only OptChat tool behavior."""

import json

from optchat.compactor import Compactor
from optchat.store import Log
from optchat.tools import child_chat_root, date, zoom
from optchat.tree import Tree
from optchat.view import View


def test_zoom_whole_message_and_two_halves(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        view = View(tree)
        compactor = Compactor(tree, lambda prompt: "short summary")
        for text in ("first message", "second message"):
            index = log.append("user", text)
            compactor.seed(index, text, "user")
            view.add_root(index)
        compactor.run_pending()

        whole = json.loads(zoom(log, tree, view, 0, 1))
        assert whole["text"] == "first message"
        assert whole["media"] == []
        halves = json.loads(zoom(log, tree, view, 0, 2))["halves"]
        assert [(half["id"], half["n"], half["text"]) for half in halves] == [
            (0, 1, "user: first message"), (1, 1, "user: second message")]


def test_zoom_rejects_bad_coordinates_and_date_is_stored(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        view = View(tree)
        index = log.append("user", "hello")
        assert "error" in json.loads(zoom(log, tree, view, 0, 3))
        assert "error" in json.loads(zoom(log, tree, view, 1, 2))
        assert json.loads(date(log, index))["date"] == log.get(index).date
        assert "error" in json.loads(date(log, 99))


def test_zoom_child_chat_and_invalid_names(tmp_path):
    child_root = tmp_path / "agents" / "child-session"
    with Log(child_root) as child_log:
        child_tree = Tree(child_root)
        child_view = View(child_tree)
        index = child_log.append("user", "child task")
        Compactor(child_tree, lambda prompt: "summary").seed(index, "child task", "user")
        child_view.add_root(index)
        child_tree.save()
        child_view.save()
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"Researcher": "child-session"}), encoding="utf-8")

    with Log(tmp_path) as master_log:
        master_tree = Tree(tmp_path)
        master_view = View(master_tree)
        master_log.append("user", "master task")
        whole = json.loads(zoom(master_log, master_tree, master_view, 0, 1))
        assert whole["text"] == "master task"
        assert whole["media"] == []
        result = json.loads(zoom(master_log, master_tree, master_view, "Researcher"))
        assert result["name"] == "Researcher"
        assert result["chat"] == "<chat>\n0+1|user: child task\n</chat>"
        assert "error" in json.loads(zoom(master_log, master_tree, master_view, "Unknown"))
        assert "error" in json.loads(zoom(master_log, master_tree, master_view, 1.5))


def test_zoom_and_date_payloads_name_the_scope(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        view = View(tree)
        compactor = Compactor(tree, lambda prompt: "summary")
        for text in ("hello", "world"):
            index = log.append("user", text)
            compactor.seed(index, text, "user")
            view.add_root(index)
        compactor.run_pending()

        whole = json.loads(zoom(log, tree, view, 0, 1))
        assert whole["scope"] is None
        scoped = json.loads(zoom(log, tree, view, 0, 1, scope="Researcher"))
        assert scoped["scope"] == "Researcher"
        assert scoped["text"] == "hello"
        halves = json.loads(zoom(log, tree, view, 0, 2, scope="Researcher"))
        assert halves["scope"] == "Researcher"
        assert [half["text"] for half in halves["halves"]] == ["user: hello", "user: world"]
        assert json.loads(date(log, 0, scope="Researcher")) == {
            "id": 0, "date": log.get(0).date, "scope": "Researcher"}


def test_child_chat_root_resolves_names_and_rejects_unsafe_keys(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "index.json").write_text(json.dumps({
        "ok": "child-1", "traversal": "../escape", "absolute": "/etc",
        "dot": ".", "empty": "", "nonstring": 7}), encoding="utf-8")

    assert child_chat_root(tmp_path, "ok") == agents / "child-1"
    for name in ("traversal", "absolute", "dot", "empty", "nonstring", "missing", ""):
        assert child_chat_root(tmp_path, name) is None


def test_zoom_child_chat_names_the_scope_its_ids_need(tmp_path):
    child_root = tmp_path / "agents" / "child-session"
    with Log(child_root) as child_log:
        child_tree = Tree(child_root)
        child_view = View(child_tree)
        index = child_log.append("user", "child task")
        Compactor(child_tree, lambda prompt: "summary").seed(index, "child task", "user")
        child_view.add_root(index)
        child_tree.save()
        child_view.save()
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"Researcher": "child-session"}), encoding="utf-8")

    with Log(tmp_path) as master_log:
        master_tree = Tree(tmp_path)
        master_view = View(master_tree)
        result = json.loads(zoom(master_log, master_tree, master_view, "Researcher"))
        assert result["scope"] == "Researcher"
        # The engine resolves a scope to the child's chat before calling; here the
        # read-only handle stands in for the child's log while its owner is closed.
        with Log(child_root, read_only=True) as child_log:
            line = json.loads(zoom(child_log, Tree(child_root), None, 0, 1, scope=result["scope"]))
        assert line["text"] == "child task"
        assert line["scope"] == "Researcher"


def test_zoom_by_name_does_not_seal_a_child_tree_tail(tmp_path):
    child_root = tmp_path / "agents" / "child-session"
    with Log(child_root) as child_log:
        child_tree = Tree(child_root)
        child_view = View(child_tree)
        index = child_log.append("user", "child task")
        Compactor(child_tree, lambda prompt: "summary").seed(index, "child task", "user")
        child_view.add_root(index)
        child_tree.save()
        child_view.save()
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"Researcher": "child-session"}), encoding="utf-8")
    tree_path = next((child_root / "tree").glob("*.jsonl"))

    # A complete node whose newline never landed: a sealing reader would make the
    # file whole - a write to a chat this reader does not own.
    torn = {"l": 0, "i": 1, "text": "user: second",
            "size": len("user: second".encode("utf-8"))}
    with tree_path.open("ab") as stream:
        stream.write(json.dumps(torn).encode("utf-8"))
    before = tree_path.read_bytes()

    with Log(tmp_path) as master_log:
        master_tree = Tree(tmp_path)
        master_view = View(master_tree)
        result = json.loads(zoom(master_log, master_tree, master_view, "Researcher"))
        assert "child task" in result["chat"]

    assert tree_path.read_bytes() == before

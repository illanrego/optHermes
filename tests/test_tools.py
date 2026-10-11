"""Read-only OptChat tool behavior."""

import json

from optchat.compactor import Compactor
from optchat.store import Log
from optchat.tree import Tree
from optchat.tools import date, zoom
from optchat.view import View


def test_zoom_whole_message_and_two_halves(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        view = View(tree)
        compactor = Compactor(tree, lambda prompt: "short summary")
        for text in ("first message", "second message"):
            index = log.append("user", text)
            compactor.seed(index, text)
            view.add_root(index)
        compactor.run_pending()

        assert json.loads(zoom(log, tree, view, 0, 1))["text"] == "first message"
        halves = json.loads(zoom(log, tree, view, 0, 2))["halves"]
        assert [(half["id"], half["n"], half["text"]) for half in halves] == [
            (0, 1, "first message"), (1, 1, "second message")]


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
        Compactor(child_tree, lambda prompt: "summary").seed(index, "child task")
        child_view.add_root(index)
        child_tree.save()
        child_view.save()
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"Researcher": "child-session"}), encoding="utf-8")

    with Log(tmp_path) as master_log:
        master_tree = Tree(tmp_path)
        master_view = View(master_tree)
        master_log.append("user", "master task")
        assert json.loads(zoom(master_log, master_tree, master_view, 0, 1))["text"] == "master task"
        result = json.loads(zoom(master_log, master_tree, master_view, "Researcher"))
        assert result["name"] == "Researcher"
        assert result["chat"] == "<chat>\n0+1|child task\n</chat>"
        assert "error" in json.loads(zoom(master_log, master_tree, master_view, "Unknown"))
        assert "error" in json.loads(zoom(master_log, master_tree, master_view, 1.5))

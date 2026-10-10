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

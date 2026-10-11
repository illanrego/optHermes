"""Importing pre-chat memories into the log, tree, and persisted view."""

from optchat import import_notes
from optchat.compactor import Compactor
from optchat.store import Log
from optchat.tree import Tree
from optchat.view import View


def test_import_notes_splits_seeds_and_extends_the_view(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path, node_bytes=22)
        view = View(tree)
        compactor = Compactor(tree, lambda _request: "summary", node_bytes=22, view=view)
        first = "alpha beta gamma delta epsilon"
        count = import_notes(log, compactor, view, [first])
        assert count > 1
        first_records = log.range(0, count - 1)
        assert all(record.kind == "note" and record.size <= 16 for record in first_records)
        assert "".join(record.text for record in first_records) == first
        assert all(tree.get(0, record.id).text == record.kind + ": " + record.text
                   for record in first_records)
        first_view = view.render()
        assert len(view.lines()) == count

        assert import_notes(log, compactor, view, ["later memory"]) == 1
        assert log.range(0, count - 1) == first_records
        assert log.get(count).kind == "note"
        assert tree.get(0, count).text == "note: later memory"
        assert view.render().startswith(first_view.removesuffix("</chat>"))
        assert len(view.lines()) == log.count() == count + 1
        assert View(Tree(tmp_path, node_bytes=22)).render() == view.render()

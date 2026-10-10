"""Import memories that predate the current chat into its durable log and view."""

from .compactor import Compactor
from .store import Log
from .text import split_text
from .view import View


def import_notes(log: Log, compactor: Compactor, view: View, texts: list[str]) -> int:
    """Append note chunks and persist their seeded view roots.

    The return value counts records, since one long text may become several.
    """
    appended = 0
    for text in texts:
        for chunk in split_text(text, compactor.node_bytes):
            index = log.append("note", chunk)
            compactor.seed(index, chunk)
            view.add_root(index)
            appended += 1
    if appended:
        compactor.tree.save()
        compactor.run_pending(limit=8)
        compactor.tree.save()
        view.merge_batch(log.count())
        view.save()
    return appended

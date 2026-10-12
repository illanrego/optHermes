import json
import warnings

import pytest

from optchat.tree import Tree
from optchat.view import View, due


def push(new_state, states):
    # Direct translation of the rollback push in the canonical spec.
    if states is None:
        return {"keep": 0, "life": 0, "state": new_state, "older": None}
    keep, life, state, older = (states[key] for key in ("keep", "life", "state", "older"))
    if keep == 0:
        return {"keep": 1, "life": life, "state": state, "older": older}
    if life > 0:
        return {"keep": 0, "life": 0, "state": new_state,
                "older": {"keep": 0, "life": life - 1, "state": state, "older": older}}
    return {"keep": 0, "life": life, "state": new_state, "older": push(state, older)}


def push_spans(states, count):
    starts = []
    while states is not None:
        starts.append(states["state"])
        states = states["older"]
    starts.reverse()
    return [(start, end - start) for start, end in zip(starts, starts[1:] + [count])]


def rendered_size(spans):
    body = "".join(f"{start}+{span}|x\n" for start, span in spans)
    return len(("<chat>\n" + body + "</chat>").encode("utf-8"))


def test_due_uses_the_last_message():
    assert due(0, 0, 5) == 4
    assert due(1, 0, 5) == 1


def test_push_equivalence_for_every_count_through_2000(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree, min_bytes=1, max_bytes=1)
    states = None
    for t in range(2001):
        if t:
            tree.put(0, t - 1, "x")
            level = 0
            index = t - 1
            while index % 2:
                parent = index // 2
                tree.put(level + 1, parent, "x")
                level += 1
                index = parent
            view.add_root(t - 1)
            states = push(t - 1, states)
        expected = push_spans(states, t)
        budget = rendered_size(expected)
        view.min_bytes = budget
        view.max_bytes = budget
        view.merge_batch(t)
        actual = [(line.i << line.l, 1 << line.l) for line in view.lines()]
        assert actual == expected, t


def test_sawtooth_and_stable_prefix(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree, min_bytes=40, max_bytes=80)
    sizes = []
    for i in range(20):
        tree.put(0, i, "x")
        level = 0
        index = i
        while index % 2:
            parent = index // 2
            tree.put(level + 1, parent, "x")
            level += 1
            index = parent
        before = view.render()
        view.add_root(i)
        assert view.render().startswith(before[:-7])
        sizes.append(len(view.lines()))
        exceeded = len(view.render().encode()) > view.max_bytes
        view.merge_batch(i + 1)
        if exceeded:
            assert len(view.render().encode()) <= view.min_bytes
    assert sizes[:3] == [1, 2, 3]
    assert any(sizes[i] > len(view.lines()) for i in range(len(sizes)))


def test_render_and_persistence_never_rebuild_existing_view(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    view.add_root(0)
    assert view.render() == "<chat>\n0+1|(not summarized yet: zoom it)\n</chat>"
    tree.put(0, 0, "first\nsecond")
    view.add_root(1)
    tree.put(0, 1, "later")
    tree.put(1, 0, "merged")
    assert view.render() == "<chat>\n0+1|first second\n1+1|later\n</chat>"
    view.save()
    assert json.loads((tmp_path / "view.json").read_text())["pairs"] == [[0, 0], [0, 1]]
    loaded = View(tree)
    assert loaded.render() == view.render()
    loaded.load()
    assert loaded.render() == view.render()
    assert "date" not in loaded.render()


def test_settled_tracks_placeholder_lines(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    assert view.settled()
    view.add_root(0)
    view.add_root(1)
    assert not view.settled()
    tree.put(0, 0, "first")
    assert not view.settled()
    tree.put(0, 1, "second")
    assert view.settled()


def test_missing_view_on_a_chat_with_nodes_warns_and_records_the_recovery(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "a")
    tree.put(0, 1, "b")
    tree.put(1, 0, "a\nb")
    tree.save()
    # view.json is gone (never written, or lost); the tree kept its nodes.
    with pytest.warns(RuntimeWarning, match="view state is missing") as caught:
        view = View(tree)
    assert "cache entries are gone" in str(caught[0].message)
    assert [(line.l, line.i) for line in view.lines()] == [(1, 0)]
    data = json.loads((tmp_path / "view.json").read_text(encoding="utf-8"))
    assert data["recovered"] is True

    # A later load sees the recorded recovery and does not warn again.
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        loaded = View(Tree(tmp_path))
    assert not [item for item in later if "view state" in str(item.message)]
    assert loaded._recovered is True
    assert [(line.l, line.i) for line in loaded.lines()] == [(1, 0)]


def test_missing_view_on_an_empty_chat_is_silent(tmp_path):
    tree = Tree(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view = View(tree)
    assert not [item for item in caught if "view" in str(item.message)]
    assert view.render() == "<chat>\n</chat>"
    assert json.loads((tmp_path / "view.json").read_text(encoding="utf-8"))["recovered"] is False


def test_context_view_merges_further_without_changing_live_pairs(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    for i in range(64):
        tree.put(0, i, "x" * 20)
        level, index = 0, i
        while index % 2:
            parent = index // 2
            tree.put(level + 1, parent, "merged")
            level += 1
            index = parent
        view.add_root(i)
    original = view.render()
    pairs = view._pairs.copy()

    context = view.context_view(min_bytes=100, max_bytes=200)

    assert context.startswith("<chat>\n") and context.endswith("</chat>")
    assert len(context.encode("utf-8")) <= 100
    assert len(context.splitlines()) < len(original.splitlines())
    assert view._pairs == pairs
    assert view.render() == original


def test_context_view_returns_fitting_view_unchanged(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    tree.put(0, 0, "short")
    view.add_root(0)
    assert view.context_view() == view.render()


def test_context_boundary_survives_live_merges_and_later_requests(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree, min_bytes=40, max_bytes=50)
    for i in range(4):
        tree.put(0, i, f"message-{i}")
        view.add_root(i)
    for i in range(2):
        tree.put(1, i, f"pair-{i}")
    tree.put(2, 0, "all four including the future")
    view.merge_batch(4)
    view.context_view(min_bytes=40, max_bytes=50, end=4)

    # This boundary cuts inside the working compaction view's merged node.
    context = view.context_view(min_bytes=40, max_bytes=50, end=1)
    assert context == "<chat>\n0+1|message-0\n</chat>"
    assert "future" not in context
    assert view.context_view(min_bytes=40, max_bytes=50, end=1) == context


def test_context_stops_at_first_hole_and_pins_bytes_across_restart(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    for i in range(4):
        view.add_root(i)
    tree.put(0, 0, "before")
    tree.put(0, 2, "after hole")
    before = view.context_view(end=3)
    assert before == "<chat>\n0+1|before\n</chat>"
    tree.put(0, 1, "newly built")
    tree.put(0, 3, "later")
    after = view.context_view(end=4)
    assert "newly built" in after
    assert view.context_view(end=3) == before
    tree.save()
    view.save()
    loaded = View(Tree(tmp_path))
    assert loaded.context_view(end=3) == before
    assert loaded.context_view(end=4) == after


def test_pending_live_batch_resumes_below_ceiling_after_restart(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree, min_bytes=40, max_bytes=110)
    for i in range(4):
        tree.put(0, i, "x" * 25)
        view.add_root(i)
    tree.put(1, 0, "a")
    assert len(view.render().encode()) > view.max_bytes
    view.merge_batch(4)
    assert view.min_bytes < len(view.render().encode()) <= view.max_bytes
    tree.save()
    view.save()

    tree = Tree(tmp_path)
    view = View(tree, min_bytes=40, max_bytes=110)
    tree.put(1, 1, "b")
    view.merge_batch(4)
    assert len(view.render().encode()) <= view.min_bytes
    assert not view._batch_pending


def test_compaction_batch_resumes_without_changing_pinned_request(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    for i in range(4):
        tree.put(0, i, "x" * 25)
        view.add_root(i)
    tree.put(1, 0, "a")
    context = view.context_view(min_bytes=40, max_bytes=110, end=4)
    assert 40 < len(context.encode()) <= 110
    tree.save()
    view.save()
    tree = Tree(tmp_path)
    view = View(tree)
    tree.put(1, 1, "b")
    view.merge_batch(4)
    state = view._compaction_views[40, 110]
    assert len(view._render(state.pairs).encode()) <= 40
    assert not state.batch_pending
    assert view.context_view(min_bytes=40, max_bytes=110, end=4) == context


def test_load_legacy_pairs_without_rebuilding(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "first")
    tree.put(0, 1, "second")
    tree.put(1, 0, "merged")
    (tmp_path / "view.json").write_text("[[0,0],[0,1]]")
    view = View(tree)
    assert [(line.l, line.i) for line in view.lines()] == [(0, 0), (0, 1)]


def test_compaction_prefix_stays_stable_between_batches(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    previous = "<chat>\n</chat>"
    for i in range(6):
        tree.put(0, i, "é" * 10)
        view.add_root(i)
        context = view.context_view(end=i + 1)
        assert context.startswith(previous[:-7])
        previous = context


def test_targets_with_same_boundary_keep_independent_persisted_snapshots(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    for i in range(3):
        view.add_root(i)
    root = view.context_view(target=(0, 2))
    assert root == "<chat>\n</chat>"
    tree.put(0, 0, "first")
    tree.put(0, 1, "second")
    merge = view.context_view(target=(1, 0))
    assert merge == "<chat>\n0+1|first\n1+1|second\n</chat>"
    assert view.context_view(target=(0, 2)) == root
    tree.save()
    view.save()
    loaded = View(Tree(tmp_path))
    assert loaded.context_view(target=(0, 2)) == root
    assert loaded.context_view(target=(1, 0)) == merge


def test_compaction_view_has_its_own_sawtooth(tmp_path):
    tree = Tree(tmp_path)
    view = View(tree)
    previous = "<chat>\n</chat>"
    batches = 0
    for i in range(32):
        tree.put(0, i, "é" * 10)
        view.add_root(i)
        l, index = 0, i
        while index % 2:
            l, index = l + 1, index // 2
            tree.put(l, index, "merged")
        appended_size = len(previous.encode()) + len(f"{i}+1|{'é' * 10}\n".encode())
        context = view.context_view(min_bytes=100, max_bytes=200, end=i + 1)
        if appended_size > 200:
            batches += 1
            assert len(context.encode()) <= 100
        else:
            assert context.startswith(previous[:-7])
        assert len(view.lines()) == i + 1
        previous = context
    assert batches > 1


def test_view_save_syncs_the_directory_like_a_checkpoint(tmp_path, monkeypatch):
    """A view publication uses the same durable replacement as a checkpoint
    publication: fsync'd temporary, atomic rename, and the directory sync that
    makes the rename itself durable."""
    from optchat import persistence

    synced = []
    monkeypatch.setattr(persistence, "sync_directory", lambda path: synced.append(path))
    view = View(Tree(tmp_path))
    view.save()

    assert tmp_path in synced
    assert json.loads((tmp_path / "view.json").read_text(encoding="utf-8"))["version"] == 2

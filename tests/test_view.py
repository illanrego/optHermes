import json

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
    assert json.loads((tmp_path / "view.json").read_text()) == [[0, 0], [0, 1]]
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


def test_missing_view_is_created_once_from_tree(tmp_path):
    tree = Tree(tmp_path)
    tree.put(0, 0, "a")
    tree.put(0, 1, "b")
    tree.put(1, 0, "a\nb")
    view = View(tree)
    assert [(line.l, line.i) for line in view.lines()] == [(1, 0)]
    assert (tmp_path / "view.json").exists()


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

"""Engine-assertable selection and durable snapshot lifecycle; no provider simulation."""

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from optchat.engine import OptChatEngine
from optchat.prompts import SYSTEM_PROMPT
from optchat.turn_view import TurnView, TurnViewUnavailable


def row(uid, text, role="user"):
    return {"role": role, "content": text, "message_uid": uid}


def boundary(uid="task", turn="turn", scope="scope", start=1, session="session"):
    return SimpleNamespace(session_id=session, turn_id=turn, reading_scope_id=scope,
                           incoming_uid=uid, request_start=start)


def request(rows):
    return [{"role": "system", "content": "host"}] + [
        {key: value for key, value in item.items() if key != "message_uid"} for item in rows]


@pytest.fixture
def engine(tmp_path):
    active = OptChatEngine(tmp_path, summarize=lambda _: "summary")
    active.on_session_start("session")
    active._capture_turn_tools([])
    yield active
    active.log.close()


def select(engine, rows, bound=None, **kwargs):
    return engine.select_context(request(rows), conversation_messages=rows,
                                 turn_boundary=bound or boundary(), **kwargs)


def test_host_suffix_keeps_steering_and_ignores_reanchor(engine):
    rows = [row("prior", "continue"), row("old-reply", "past", "assistant"),
            row("task", "continue"), row("work", "working", "assistant"),
            row("steering", "continue")]
    api = request(rows)
    api[3]["content"] = [{"type": "text", "text": "host-owned exact payload"}]
    before = copy.deepcopy(api)
    selected = engine.select_context(api, conversation_messages=rows,
                                     incoming_message=rows[-1], turn_boundary=boundary(start=3))
    assert selected[2:] == before[3:]
    assert api == before
    selected[2]["content"][0]["text"] = "mutation"
    assert api == before
    assert engine.selection_status == {"boundary": "host", "conformance": "agrees",
                                       "snapshot": "selected"}


@pytest.mark.parametrize("source", [[], [row("task", "one"), row("task", "two")],
                                      [row("task", "one"), row("extra", "two")]])
def test_conformance_disagreement_warns_once_and_uses_host(engine, caplog, source):
    engine.on_turn_complete([row("prior", "historical context")])
    api = request([row("task", "exact tail")])
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            selected = engine.select_context(api, conversation_messages=source,
                                             turn_boundary=boundary())
            assert selected[2:] == api[1:]
    assert sum("conformance disagreement" in entry.message for entry in caplog.records) == 1
    assert engine.selection_status["conformance"] == "disagrees"


def test_boundary_without_source_is_explicitly_unchecked(engine):
    selected = engine.select_context(request([row("task", "task")]), turn_boundary=boundary())
    assert selected[-1]["content"] == "task"
    assert engine.selection_status["conformance"] == "source unavailable"


@pytest.mark.parametrize("anchor", ["identity", "equality", "absent", "unmatched"])
def test_boundaryless_output_matches_legacy_bytes_and_is_flagged(engine, caplog, anchor):
    rows = [row("old", "past"), row("task", "task"), row("reply", "work", "assistant"),
            row("steer", "steering")]
    api = request(rows)
    incoming = {"identity": api[2], "equality": dict(api[2]), "absent": None,
                "unmatched": {"role": "user", "content": "missing"}}[anchor]
    with caplog.at_level(logging.DEBUG):
        actual = engine.select_context(api, incoming_message=incoming, future_keyword=True)
    tail = api[2:] if anchor in ("identity", "equality") else api[-1:]
    expected = [{"role": "system", "content": "host\n\n" + SYSTEM_PROMPT},
                {"role": "assistant", "content": "<chat>\n</chat>"}] + tail
    assert json.dumps(actual).encode() == json.dumps(expected).encode()
    assert engine.selection_status["boundary"] == "absent"
    assert "turn boundary unavailable" in caplog.text
    assert not (engine.root / "turn-selection.json").exists()


def test_boundaryless_path_still_renders_live_view(engine):
    rows = [row("task", "current task")]
    before = engine.select_context(request(rows))
    engine.on_message(rows, rows[0])
    after = engine.select_context(request(rows))
    assert before[1]["content"] != after[1]["content"]
    assert "current task" in after[1]["content"]


def test_frozen_view_survives_reassembly_steering_and_live_merges(engine):
    prior = [row("prior", "older context")]
    engine.on_turn_complete(prior)
    rows = prior + [row("task", "new task")]
    selected = select(engine, rows, boundary(start=2))
    frozen = selected[1]["content"].encode("utf-8")
    engine.on_message(rows, rows[-1])
    rows += [row("work", "current work", "assistant"), row("steer", "steering")]
    engine.on_message(rows, rows[-1])
    engine.compactor.run_pending(limit=100, model_calls=False)
    engine.view.min_bytes, engine.view.max_bytes = 0, 1
    engine.view.merge_batch(engine.log.count())
    for scope in ("scope", "second-scope", "scope"):
        # Provider/refusal retries have no independent identity in the engine:
        # repeated assembly with the same boundary is the assertable operation.
        actual = select(engine, rows, boundary(start=2, scope=scope), incoming_message=rows[-1])
        assert actual[1]["content"].encode("utf-8") == frozen
        assert actual[2:] == request(rows)[2:]
    assert b"new task" not in frozen and b"current work" not in frozen and b"steering" not in frozen
    assert engine.view.render().encode("utf-8") != frozen
    assert TurnView(engine.root).state()["scopes"] == ["scope", "second-scope"]


def test_placeholder_resolution_cannot_change_selected_bytes(engine):
    prior = [row("prior", "long " * 300, "tool")]
    engine.on_message(prior, prior[0])
    # Complete ingestion without a model call so a historical placeholder remains.
    engine.compactor.run_pending = lambda **kwargs: 0
    engine.on_turn_complete(prior)
    rows = prior + [row("task", "task")]
    frozen = select(engine, rows, boundary(start=2))[1]["content"]
    assert "not summarized" in frozen
    for i in range(engine.log.count()):
        if engine.tree.get(0, i) is None:
            engine.tree.put(0, i, "resolved")
    assert select(engine, rows, boundary(start=2))[1]["content"] == frozen


def test_notification_before_selection_captures_before_first_append(engine, monkeypatch):
    prior = [row("prior", "past")]
    engine.on_turn_complete(prior)
    expected = engine.view.render()
    rows = prior + [row("task", "task"), row("reply", "reply", "assistant")]
    append = engine.log.append

    def checked_append(*args, **kwargs):
        store = TurnView(engine.root)
        state = store.state()
        assert state["uid"] == "task" and state["phase"] == "pending"
        assert store.render(state) == expected
        return append(*args, **kwargs)

    monkeypatch.setattr(engine.log, "append", checked_append)
    engine.on_message(rows, rows[-1])
    actual = select(engine, rows, boundary(start=2))
    assert actual[1]["content"] == expected
    assert "task" not in actual[1]["content"]
    assert TurnView(engine.root).state()["turn"] == "turn"


@pytest.mark.parametrize("select_first", [True, False, "before_append"])
def test_restart_restores_selected_or_pending_render(engine, select_first):
    rows = [row("task", "task"), row("reply", "reply", "assistant")]
    expected = engine.view.render()
    if select_first:
        assert select(engine, rows)[1]["content"] == expected
    if select_first != "before_append":
        engine.on_message(rows, rows[-1])
    engine.log.close()
    resumed = OptChatEngine(engine.root)
    resumed.on_session_start("session")
    try:
        if select_first != "before_append":
            assert "task" in resumed.view.render()
        assert select(resumed, rows)[1]["content"] == expected
    finally:
        resumed.log.close()


@pytest.mark.parametrize("loss", ["render", "association", "both", "corrupt"])
def test_restart_missing_snapshot_is_unavailable_never_rerendered(engine, caplog, loss):
    rows = [row("task", "task")]
    select(engine, rows)
    engine.on_message(rows, rows[0])
    engine.log.close()
    if loss in ("render", "both"):
        (engine.root / "turn-render.json").unlink()
    if loss in ("association", "both"):
        (engine.root / "turn-selection.json").unlink()
    if loss == "corrupt":
        (engine.root / "turn-render.json").write_text('"wrong render"')
    resumed = OptChatEngine(engine.root)
    resumed.on_session_start("session")
    try:
        with caplog.at_level(logging.ERROR):
            assert select(resumed, rows) is None
            assert select(resumed, rows) is None
        assert resumed.selection_status["snapshot"] == "unavailable"
        assert sum("selected view unavailable" in entry.message for entry in caplog.records) == 1
        assert resumed.log.count() == 1
    finally:
        resumed.log.close()


def test_capture_crash_between_association_and_render_is_unavailable(engine, monkeypatch):
    import optchat.turn_view as module
    publish = module.atomic_json

    def fail_payload(path, value):
        if path.name == "turn-render.json":
            raise OSError("injected publication failure")
        publish(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_payload)
    assert select(engine, [row("task", "task")]) is None
    assert TurnView(engine.root).state()["phase"] == "selected"
    with pytest.raises(TurnViewUnavailable, match="missing"):
        TurnView(engine.root).render(TurnView(engine.root).state())
    assert engine.log.count() == 0


def test_snapshot_loss_keeps_raw_logging_and_loss_evidence(engine):
    rows = [row("task", "task")]
    select(engine, rows)
    engine.on_message(rows, rows[0])
    (engine.root / "turn-render.json").unlink()
    (engine.root / "turn-selection.json").unlink()
    rows.append(row("reply", "must still be logged", "assistant"))
    engine.on_message(rows, rows[-1])
    assert engine.log.get(1).text == "must still be logged"
    assert json.loads((engine.root / "ingest.json").read_text())["turn_snapshot_required"]
    assert select(engine, rows) is None


def test_uidless_preselection_capture_cannot_be_bound_by_text(engine):
    rows = [{"role": "user", "content": "same task"}]
    engine.on_message(rows, rows[0])
    rows[0]["message_uid"] = "task"
    assert select(engine, rows) is None
    assert engine.selection_status["snapshot"] == "unavailable"


def test_completed_turn_retires_payload_and_next_turn_has_new_snapshot(engine):
    rows = []
    for number in range(12):
        uid = f"task-{number}"
        rows.append(row(uid, f"task {number}"))
        bound = boundary(uid=uid, turn=f"turn-{number}", start=len(rows))
        assert select(engine, rows, bound) is not None
        engine.on_turn_complete(rows)
        state = TurnView(engine.root).state()
        assert state["phase"] == "complete" and state["scopes"] == []
        assert json.loads((engine.root / "turn-render.json").read_text()) is None
    assert len(list(engine.root.glob("turn-*.json"))) == 2
    assert select(engine, rows, bound) is None  # retired is never recaptured


def test_new_turn_without_completion_replaces_only_after_uid_frontier(engine):
    rows = [row("task", "first")]
    select(engine, rows)
    engine.on_message(rows, rows[0])
    rows.append(row("next", "second"))
    actual = select(engine, rows, boundary(uid="next", turn="next-turn", start=2))
    assert "first" in actual[1]["content"] and "second" not in actual[1]["content"]
    assert TurnView(engine.root).state()["turn"] == "next-turn"


def test_late_new_boundary_cannot_recapture_after_its_rows_were_ingested(engine):
    rows = [row("task", "first")]
    select(engine, rows)
    engine.on_message(rows, rows[0])
    rows.append(row("next", "second"))
    engine.on_message(rows, rows[-1])  # no completion or new boundary was supplied
    assert select(engine, rows, boundary(uid="next", turn="next-turn", start=2)) is None
    assert "predates" in engine.selection_status["reason"]


def test_child_owns_frozen_master_render_and_restores_its_scope(engine):
    prior = [row("prior", "master context")]
    engine.on_turn_complete(prior)
    expected = engine.view.render()
    child = engine.clone_for_agent()
    child.on_session_start("child", platform="subagent")
    assert child._master_view is not engine.view
    engine.on_turn_complete(prior + [row("later", "master mutation")])
    rows = [row("task", "child task")]
    bound = boundary(session="child", scope="child-scope")
    try:
        assert select(child, rows, bound)[1]["content"] == expected
        child.on_message(rows, rows[0])
        engine.on_turn_complete(prior + [row("later", "master mutation"), row("last", "more")])
        assert select(child, rows, bound)[1]["content"] == expected
        assert TurnView(child._chat_root).state()["scopes"] == ["child-scope"]
    finally:
        child.log.close()
    resumed = OptChatEngine(engine.root)
    resumed.on_session_start("child", platform="subagent")
    try:
        assert select(resumed, rows, bound)[1]["content"] == expected
    finally:
        resumed.log.close()


@pytest.mark.parametrize("change", [{"request_start": -1}, {"request_start": True},
                                   {"incoming_uid": ""}, {"session_id": "foreign"}])
def test_invalid_boundary_steps_aside_without_text_fallback(engine, change):
    bound = boundary()
    for key, value in change.items():
        setattr(bound, key, value)
    assert select(engine, [row("task", "task")], bound) is None
    assert engine.selection_status["snapshot"] == "unavailable"


def test_same_turn_cannot_change_admitted_uid(engine):
    rows = [row("task", "task")]
    select(engine, rows)
    rows.append(row("steer", "steering"))
    assert select(engine, rows, boundary(uid="steer", start=2)) is None
    assert "UID changed" in engine.selection_status["reason"]


def test_boundary_selection_is_silent_when_conformant_and_settled(engine, caplog):
    with caplog.at_level(logging.WARNING):
        assert select(engine, [row("task", "task")]) is not None
    assert not caplog.records


def test_reviewed_seam_version_has_not_changed():
    """Pin the reviewed selection-entry contract; no invented integration rows."""
    host = Path(__file__).resolve().parents[2] / "opthermes-seam" / "agent"
    if not host.is_dir():
        pytest.skip("version-specific conformance requires sibling opthermes-seam at 202e30e78f")
    expected = {
        "context_engine.py": "d83eb54ffd3fd5d1a55e2017803b59e30784467e2ed7079640f3bb64c6d00dcd",
        "turn_request_assembly.py": "b427e71c80c81b32baa7e9b7e05e85cc99a75e4e04cd218dc0e16d9228b156f4",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((host / name).read_bytes()).hexdigest() == digest, (
            f"Re-review {name}: P2a conformance was checked against seam 202e30e78f")

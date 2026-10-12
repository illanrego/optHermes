"""Host-independent tests for the OptChat adapter."""

import copy
import importlib.util
import json
import logging
import re
import sys
import types
import warnings
from pathlib import Path

import pytest

from optchat import OptChatEngine
from optchat.compactor import Compactor
from optchat.engine import _record_digest, _row_fingerprint
from optchat.prompts import SYSTEM_PROMPT
from optchat.store import Log
from optchat.tree import Tree
from optchat.view import View, ViewStateUnavailable


class FakeSummarizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "summary"


@pytest.fixture
def cache_engine_class(monkeypatch):
    """Load the adapter against the host seam without changing other tests' ABC."""
    agent = types.ModuleType("agent")
    agent.__path__ = []
    base = types.ModuleType("agent.context_engine")
    base.ContextEngine = type("ContextEngine", (), {"cache_plan": lambda self: None})
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.context_engine", base)
    caching = types.ModuleType("agent.prompt_caching")
    caching.PromptCachePlan = lambda messages, tools: types.SimpleNamespace(
        messages=messages, tools=tools)
    monkeypatch.setitem(sys.modules, "agent.prompt_caching", caching)
    spec = importlib.util.spec_from_file_location(
        "optchat._engine_with_cache_seam", Path(__file__).parents[1] / "optchat" / "engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OptChatEngine


def test_seam_without_view_returns_none(cache_engine_class):
    engine = cache_engine_class()
    assert engine.cache_plan([{"role": "user", "content": "now"}], []) is None


@pytest.mark.parametrize("native", [False, True])
def test_shim_and_seam_produce_identical_plan(monkeypatch, cache_engine_class, native):
    from optchat import shim

    marker = {"type": "ephemeral", "ttl": "1h"}
    tools = [{"function": {"name": "host_tool"}}]
    system = [{"type": "text", "text": "static", "cache_control": marker},
              {"type": "text", "text": "runtime"}]
    if native:
        tools[0]["cache_control"] = marker
    else:
        system[-1]["cache_control"] = marker
    messages = [{"role": "system", "content": system},
                {"role": "assistant", "content": "<chat>\n0+1|history\n1+1|more\n</chat>"},
                {"role": "user", "content": [{"type": "text", "text": "task",
                                                 "cache_control": marker}]},
                {"role": "assistant", "content": [{"type": "text", "text": "reply",
                                                      "cache_control": marker}]}]
    seam_engine = cache_engine_class()
    seam = seam_engine.cache_plan(messages, tools)
    assert seam.messages[0] == messages[0]
    assert seam.tools == tools

    # Remove the host ABC hook to exercise the stock installation path.
    del sys.modules["agent.context_engine"].ContextEngine.cache_plan
    loop = types.ModuleType("agent.conversation_loop")
    prepared = {"host": "state"}
    loop._redecorate_prompt_cache_for_provider = lambda _agent, _messages: (
        messages, prepared, tools)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", loop)
    sys.modules["agent"].conversation_loop = loop
    shim.install(OptChatEngine())
    active = OptChatEngine()
    agent = types.SimpleNamespace(context_compressor=active, _use_prompt_caching=True)
    shimmed = loop._redecorate_prompt_cache_for_provider(agent, messages)
    assert shimmed == (seam.messages, prepared, seam.tools)
    assert shimmed[1] is prepared
    assert active._turn_tools == seam_engine._turn_tools
    shim.uninstall()


@pytest.mark.parametrize("capture_before_start", [False, True])
@pytest.mark.parametrize("tools", [[], [
    {"type": "function", "function": {"name": "host_search",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
]])
def test_cache_plan_tools_reach_next_compaction(
        tmp_path, cache_engine_class, capture_before_start, tools):
    summarizer = FakeSummarizer()
    engine = cache_engine_class(tmp_path, summarize=summarizer, node_bytes=16)
    tools = copy.deepcopy(tools)
    expected = copy.deepcopy(tools)
    if not capture_before_start:
        engine.on_session_start("session")
    engine.cache_plan([], tools)
    if not capture_before_start:
        assert engine.compactor.tools == expected  # Updated before any selection.
    tools.append({"function": {"name": "later_host_mutation"}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.select_context([{"role": "user", "content": "task"}])
        engine.on_turn_complete([{"role": "tool", "content": "long result " * 5}])
    assert not caught
    assert summarizer.prompts
    assert all(request.tools == expected for request in summarizer.prompts)
    engine.log.close()


def test_missing_tools_warn_once_per_engine_and_keep_fallback(tmp_path):
    for name in ("first", "second"):
        summarizer = FakeSummarizer()
        engine = OptChatEngine(tmp_path / name, summarize=summarizer, node_bytes=16)
        messages = [{"role": "tool", "content": "long result " * 5,
                     "message_uid": "first-result"}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.on_turn_complete(messages)
            engine.on_turn_complete(messages + [dict(messages[0], message_uid="second-result")])
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert "cache-policy gates cache_plan, the only hook that carries tools" in str(caught[0].message)
        assert summarizer.prompts
        assert all(request.tools == engine.get_tool_schemas() for request in summarizer.prompts)
        engine.log.close()


@pytest.mark.parametrize("route, should_warn", [
    ({}, False),
    ({"provider": None, "model": ""}, False),
    ({"provider": "auto", "model": "auto"}, False),
    ({"provider": "auto"}, False),
    ({"provider": "other"}, True),
    ({"model": "other-model"}, True),
    ({"provider": "auto", "model": "other-model"}, True),
    ({"provider": "other", "model": "other-model"}, True),
])
def test_route_override_warns_once_per_engine(tmp_path, monkeypatch, route, should_warn):
    config = types.ModuleType("hermes_cli.config")
    config.load_config_readonly = lambda: {"auxiliary": {"optchat": dict(route)}}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)
    before = copy.deepcopy(route)
    for name in ("first", "second"):
        engine = OptChatEngine(tmp_path / name, summarize=FakeSummarizer())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.on_session_start("session", model="turn-model")
            engine.on_session_start("session", model="turn-model")
        assert engine._turn_model == "turn-model"
        assert len(caught) == int(should_warn)
        if should_warn:
            assert caught[0].category is RuntimeWarning
            assert "may not share the turns' cache entry" in str(caught[0].message)
            assert "turn-model" in str(caught[0].message)
        engine.log.close()
    assert route == before


def test_selection_keeps_only_current_turn_and_does_not_mutate_history(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "old question"},
                             {"role": "assistant", "content": "old answer"}])
    history = [{"role": "user", "content": {"nested": ["history"]}}]
    before = copy.deepcopy(history)
    request = [{"role": "system", "content": "Host instructions"},
               {"role": "user", "content": "old question"},
               {"role": "assistant", "content": "old answer"},
               {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "current tool step"}]

    selected = engine.select_context(request, conversation_messages=history)
    assert [message["role"] for message in selected] == [
        "system", "assistant", "user", "assistant"]
    assert selected[0]["content"] == "Host instructions\n\n" + SYSTEM_PROMPT
    assert selected[1]["content"] == engine.view.render()
    assert selected[-1] == request[-1]
    assert history == before
    assert engine.select_context(selected)[0]["content"] == selected[0]["content"]


def test_selection_keeps_task_and_tool_work_across_steering(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    task = {"role": "user", "content": {"text": "original task"}}
    request = [{"role": "system", "content": "host"},
               {"role": "user", "content": "older turn"}, task,
               {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
               {"role": "tool", "content": "current tool work", "tool_call_id": "call-1"},
               {"role": "user", "content": "steering"}]
    history = copy.deepcopy(request)
    request_before = copy.deepcopy(request)
    history_before = copy.deepcopy(history)

    for incoming in (task, copy.deepcopy(task)):
        selected = engine.select_context(request, conversation_messages=history,
                                         incoming_message=incoming)
        assert selected[2:] == request[2:]
    assert engine.select_context(request, incoming_message={"role": "user", "content": "missing"})[2:] == request[-1:]
    assert engine.select_context(request)[2:] == request[-1:]
    assert request == request_before
    assert history == history_before


def test_selection_prefers_incoming_identity_over_an_equal_earlier_row(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    task = {"role": "user", "content": "same words"}
    request = [copy.deepcopy(task), task, {"role": "user", "content": "steering"}]

    assert engine.select_context(request, incoming_message=task)[2:] == request[1:]


def test_selection_no_change_and_stable_prefix(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    assert engine.select_context([]) is None
    assert engine.select_context([{"role": "system", "content": "host"}]) is None
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    request = [{"role": "system", "content": "host"},
               {"role": "user", "content": "second"}]
    first = engine.select_context(request)
    engine.on_turn_complete([{"role": "user", "content": "first"},
                             {"role": "assistant", "content": "answer"}])
    second = engine.select_context(request[:-1] + [{"role": "user", "content": "third"}])
    assert first[0] == second[0]
    assert second[1]["content"].startswith(first[1]["content"].removesuffix("</chat>"))


def test_ingestion_is_idempotent_and_clone_shares_store(tmp_path):
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, summarize=summarizer)
    engine.on_session_start("session")
    messages = [{"role": "system", "content": "host"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"}]
    engine.on_turn_complete(messages)
    engine.on_turn_complete(messages + [{"role": "user", "content": "three"}])
    assert [(record.kind, record.text) for record in engine.log.range(0, 2)] == [
        ("user", "one"), ("unii", "two"), ("user", "three")]
    clone = engine.clone_for_agent()
    assert clone is not engine and clone.log is engine.log
    # The same session continuing: the boundary is found by row identity, so only
    # the new row lands even though the clone re-sends the whole transcript.
    clone.on_turn_complete(messages + [{"role": "user", "content": "three"},
                                       {"role": "user", "content": "from clone"}])
    assert engine.log.get(3).text == "from clone"
    assert engine.should_compress() is False
    assert {schema["function"]["name"] for schema in engine.get_tool_schemas()} == {
        "optchat_zoom", "optchat_date"}
    zoom_schema = engine.get_tool_schemas()[0]["function"]["parameters"]
    assert zoom_schema["properties"]["id"]["anyOf"] == [
        {"type": "integer"}, {"type": "string"}]
    assert zoom_schema["required"] == ["id"]


def test_tool_mapping_and_json_errors(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "lookup", "arguments": '{"id": 1}'}}]},
        {"role": "tool", "content": "result"}]
    engine.on_turn_complete(messages)
    assert engine.log.get(0).kind == "tool"
    assert json.loads(engine.log.get(0).text)["name"] == "lookup"
    assert engine.log.get(1).kind == "echo"
    assert "error" in json.loads(engine.handle_tool_call("optchat_date", {"id": -1}))


def test_delegate_result_is_work_and_other_tool_result_is_echo(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    calls = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "child-1", "function": {"name": "delegate_task", "arguments": "{}"}},
        {"id": "lookup-1", "function": {"name": "lookup", "arguments": "{}"}},
    ]}
    messages = [calls,
                {"role": "tool", "tool_call_id": "lookup-1", "content": "lookup result"},
                {"role": "tool", "tool_call_id": "child-1",
                 "content": '{"agent":"Researcher","report":"found it"}'}]
    assert OptChatEngine._message_records(messages, 512)[-2:] == [
        ("echo", "lookup result"),
        ("work", '[Researcher] {"agent":"Researcher","report":"found it"}')]

    engine.on_turn_complete(messages[:1])
    engine.on_turn_complete(messages)
    assert [(record.kind, record.text) for record in engine.log.range(2, 3)] == [
        ("echo", "lookup result"),
        ("work", '[Researcher] {"agent":"Researcher","report":"found it"}')]


def test_delegate_result_name_fallback_and_field_priority():
    messages = [{"role": "assistant", "tool_calls": [
        {"id": "one", "function": {"name": "delegate_task"}},
        {"id": "two", "function": {"name": "delegate_task"}},
    ]},
        {"role": "tool", "tool_call_id": "one", "content": '{"task":"Audit"}'},
        {"role": "tool", "tool_call_id": "two", "content": "plain report"}]
    assert OptChatEngine._message_records(messages, 512)[-2:] == [
        ("work", '[Audit] {"task":"Audit"}'),
        ("work", "[subagent] plain report")]


def test_long_user_message_keeps_raw_log_and_tags_tree_sources(tmp_path):
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    text = "one two three four five six seven eight nine ten eleven twelve"
    engine.on_turn_complete([{"role": "user", "content": text}])
    records = engine.log.range(0, engine.log.count() - 1)
    chunks = [record.text for record in records]
    assert len(records) > 1
    assert all(record.kind == "user" and record.size <= 32 for record in records)
    assert "".join(chunks).split() == text.split()
    for record in records:
        source = f"{record.kind}: {record.text}"
        node = engine.tree.get(0, record.id)
        if len(source.encode("utf-8")) <= 32:
            assert node.text == source
            assert node.size == len(source.encode("utf-8"))
        else:
            assert any(f"<input>\n{source}\n</input>" in prompt.task
                       for prompt in summarizer.prompts)


def test_long_assistant_text_splits_but_tool_result_keeps_its_clip(tmp_path):
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=FakeSummarizer())
    assistant = "alpha beta gamma delta epsilon zeta eta theta"
    assistant_records = engine._records({"role": "assistant", "content": assistant}, 32)
    assert len(assistant_records) > 1
    assert all(kind == "unii" and len(chunk.encode("utf-8")) <= 32
               for kind, chunk in assistant_records)
    assert "".join(chunk for _, chunk in assistant_records) == assistant
    tool_records = engine._records({"role": "tool", "content": "x" * 30_002}, 32)
    assert tool_records == [("echo", "x" * 30_000)]


def _short_chat(n):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_selection_never_starts_a_model_call(tmp_path):
    """select_context runs synchronously inside the host's request, and a summarizer call
    in flight cannot be interrupted by the deadline. So it must not start one: anything
    needing a model is left to the background, and the view may hold a placeholder."""
    summarizer = FakeSummarizer()

    def offline(request):
        summarizer(request)
        raise RuntimeError("offline")

    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=offline)
    with pytest.warns(RuntimeWarning):
        engine.on_turn_complete([{"role": "tool", "content": "x" * 80} for _ in range(9)])
    before = len(summarizer.prompts)
    assert not engine.view.settled()

    selected = engine.select_context([{"role": "user", "content": "next"}])

    assert len(summarizer.prompts) == before
    assert "(not summarized yet: zoom it)" in selected[1]["content"]


def test_end_of_turn_compaction_settles_the_view(tmp_path):
    """v1's admission is delivered by the end-of-turn background pass, not by the inline
    settle: after on_turn_complete the view has no placeholder, so the next turn starts
    settled, including when more than eight model nodes were pending."""
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    engine.on_turn_complete([{"role": "tool", "content": "x" * 80} for _ in range(12)])

    assert engine.tree.get(0, 0).text == "summary"
    assert engine.view.settled()
    assert len(summarizer.prompts) >= 12
    assert all(engine.tree.get(0, i) is not None for i in range(12))


def test_selection_serves_current_view_when_settle_raises(tmp_path, monkeypatch, caplog):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    current_view = engine.view.render()

    def fail_settle(**_kwargs):
        raise RuntimeError("admission failed")

    monkeypatch.setattr(engine.compactor, "settle", fail_settle)
    with caplog.at_level(logging.ERROR, logger="optchat.engine"):
        selected = engine.select_context([{"role": "user", "content": "next"}])

    assert selected[1]["content"] == current_view
    assert "admission failed" in caplog.text


def test_withheld_transcript_can_be_recovered_from_view_and_zoom(tmp_path):
    class CueSummarizer(FakeSummarizer):
        def __call__(self, prompt):
            self.prompts.append(prompt)
            return "user: the project codename is in message 0; zoom for its exact spelling"

    engine = OptChatEngine(tmp_path, node_bytes=80, summarize=CueSummarizer())
    secret = "The project codename is ORCHID-17. " + "x" * 100
    engine.on_turn_complete([{"role": "tool", "content": secret},
                             {"role": "assistant", "content": "I will remember it."}])
    selected = engine.select_context([
        {"role": "system", "content": "host"},
        {"role": "tool", "content": secret},
        {"role": "assistant", "content": "I will remember it."},
        {"role": "user", "content": "What was the codename?"}])
    assert len(selected) == 3
    assert "ORCHID-17" not in selected[1]["content"]
    assert "zoom" in selected[1]["content"]
    recovered = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert "ORCHID-17" in recovered["text"]


def _zoom_from_view(engine, view_text, target):
    """The engine's side of what a model does with the view.

    Read the view's lines, pick the one covering ``target``, call ``optchat_zoom``
    for it, and follow the returned halves until the raw message is reached. Returns
    ``(payload, calls, picked line)``; the picked line is the summary the model saw.
    """
    picked = None
    for line in view_text.splitlines():
        match = re.match(r"^(\d+)\+(\d+)\|", line)
        if match and int(match.group(1)) <= target < int(match.group(1)) + int(match.group(2)):
            picked = (int(match.group(1)), int(match.group(2)), line)
            break
    assert picked is not None, "the view has no line covering the folded message"
    id_, n, line = picked
    for calls in range(1, 64):
        answer = json.loads(engine.handle_tool_call("optchat_zoom", {"id": id_, "n": n}))
        if n == 1:
            return answer, calls, line
        half = next(h for h in answer["halves"] if h["id"] <= target < h["id"] + h["n"])
        id_, n = half["id"], half["n"]
    raise AssertionError("navigation never reached the raw message")


def test_transcript_swap_acceptance_view_and_zoom_recover_folded_detail(tmp_path):
    """The engine's side of a real loop, end to end through its own surfaces.

    The transcript is withheld and the view stands in; a scripted model reads the
    view, picks the summary line covering the folded message, navigates with
    ``optchat_zoom`` until the raw text comes back, and the next request carries
    that tool result with still no transcript. A real model choosing zoom is the
    live run, not this test.
    """
    class CueSummarizer(FakeSummarizer):
        def __call__(self, prompt):
            self.prompts.append(prompt)
            return "compacted span; its exact words are behind optchat_zoom"

    engine = OptChatEngine(tmp_path, node_bytes=96, view_min_bytes=0,
                           view_max_bytes=120, summarize=CueSummarizer())
    secret = "the access code is FERN-42; " + "filler " * 40
    system_row = {"role": "system", "content": "host"}
    old = [
        {"role": "user", "content": "kickoff"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": secret},
        {"role": "assistant", "content": "noted"},
    ]
    engine.on_turn_complete([system_row] + old)
    assert secret not in engine.view.render()  # the detail is folded away

    question = {"role": "user", "content": "What was the access code?"}
    request = [system_row] + old + [question]
    selected = engine.select_context(request, incoming_message=question)

    # The transcript is gone; the view stands in, and the current turn remains.
    assert len(selected) == 3
    assert selected[1]["role"] == "assistant"
    assert selected[1]["content"] == engine.view.render()
    assert selected[2:] == [question]
    for row in old:
        assert row not in selected
    assert all(secret not in str(message.get("content")) for message in selected)

    # The scripted model reads the view, picks the line covering the folded message,
    # and navigates with the tool until the raw message comes back.
    answer, calls, picked_line = _zoom_from_view(engine, selected[1]["content"], target=2)
    assert secret not in picked_line  # the line the model picked is a summary
    assert calls >= 1
    assert (answer["id"], answer["n"], answer["kind"]) == (2, 1, "echo")
    assert secret in answer["text"]  # the folded-away detail came back

    # The next request the model would see: the tool call and its result ride the
    # current turn, and the transcript is still withheld.
    zoom_call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "zoom-1", "function": {"name": "optchat_zoom",
                                      "arguments": json.dumps({"id": 2, "n": 1})}}]}
    zoom_result = {"role": "tool", "tool_call_id": "zoom-1",
                   "content": json.dumps(answer, ensure_ascii=False)}
    next_request = request + [zoom_call, zoom_result]
    engine.on_message(next_request, copy.deepcopy(zoom_call))
    engine.on_message(next_request, copy.deepcopy(zoom_result))
    next_selected = engine.select_context(next_request, incoming_message=question)

    assert next_selected[1]["content"].startswith("<chat>")
    assert next_selected[-1] == zoom_result
    assert secret in next_selected[-1]["content"]
    for row in old:
        assert row not in next_selected
    # The detail appears exactly once - in the tool result - never via the view.
    assert sum(secret in str(message.get("content")) for message in next_selected) == 1
    engine.log.close()


def test_subagent_has_own_chat_and_master_records_one_work(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    master.on_turn_complete([{"role": "user", "content": "master context"}])
    master_view = master.view.render()

    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    selected = child.select_context([{"role": "system", "content": "host"},
                                     {"role": "user", "content": "task"}])
    assert selected[1]["content"] == master_view
    child.on_turn_complete([{"role": "user", "content": "subagent step"},
                            {"role": "assistant", "content": "subagent reply"}])
    assert child.log.root == tmp_path / "agents" / "child-session"
    assert [record.text for record in child.log.range(0, 1)] == [
        "subagent step", "subagent reply"]
    assert master.log.count() == 1
    child.log.close()
    with Log(tmp_path / "agents" / "child-session") as reopened:
        assert reopened.count() == 2

    master.on_turn_complete([{"role": "user", "content": "master context"},
                             {"role": "assistant", "tool_calls": [
                                 {"id": "delegate-1", "function": {"name": "delegate_task"}}]},
                             {"role": "tool", "tool_call_id": "delegate-1", "content": json.dumps({
                                 "results": [{"task_index": 0, "status": "completed",
                                              "summary": "finished"}]})}])
    records = master.log.range(0, master.log.count() - 1)
    assert [record.kind for record in records].count("work") == 1
    assert all("subagent step" not in record.text for record in records)
    assert records[-1].text.startswith("[child-session] ")
    assert json.loads((tmp_path / "agents" / "index.json").read_text())["child-session"] == "child-session"


def test_delegate_name_maps_to_explicit_child_session_id(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "find the answer"}])
    result = {"results": [{"task_index": 0, "name": "Researcher",
                            "child_session_id": "child-session", "summary": "found it"}]}
    master.on_turn_complete([
        {"role": "assistant", "tool_calls": [
            {"id": "delegate-1", "function": {"name": "delegate_task"}}]},
        {"role": "tool", "tool_call_id": "delegate-1", "content": json.dumps(result)},
    ])
    assert master.log.get(1).text.startswith("[Researcher] ")
    assert json.loads((tmp_path / "agents" / "index.json").read_text()) == {
        "Researcher": "child-session"}
    assert "find the answer" in json.loads(master.handle_tool_call(
        "optchat_zoom", {"id": "Researcher"}))["chat"]


def test_child_navigation_resolves_the_view_it_reads_and_its_own_scope(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    master.on_turn_complete([{"role": "user", "content": "master context"}])

    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "child step"}])
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"child-session": "child-session"}), encoding="utf-8")

    # The child renders the master's view, so id 0 is the master's record, not
    # the child's own record 0.
    whole = json.loads(child.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert whole["text"] == "master context"
    assert whole["scope"] is None
    assert json.loads(child.handle_tool_call(
        "optchat_date", {"id": 0}))["date"] == master.log.get(0).date

    # The child's own chat needs its explicit scope, even while its own log holds
    # the lock (the scoped handle opens read-only).
    own = json.loads(child.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 1, "scope": "child-session"}))
    assert own["text"] == "child step"
    assert own["scope"] == "child-session"
    assert json.loads(child.handle_tool_call(
        "optchat_date", {"id": 0, "scope": "child-session"}))["date"] == child.log.get(0).date


def test_named_child_chat_names_its_scope_and_opens_lines_inside(tmp_path):
    child_root = tmp_path / "agents" / "child-session"
    with Log(child_root) as child_log:
        child_tree = Tree(child_root)
        child_view = View(child_tree)
        compactor = Compactor(child_tree, lambda prompt: "summary")
        for text in ("first step", "second step"):
            index = child_log.append("user", text)
            compactor.seed(index, text, "user")
            child_view.add_root(index)
        compactor.run_pending()
        child_tree.save()
        child_view.save()
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"Researcher": "child-session"}), encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("master", platform="cli")

    result = json.loads(engine.handle_tool_call("optchat_zoom", {"id": "Researcher"}))
    assert result["name"] == "Researcher"
    assert result["scope"] == "Researcher"
    assert result["chat"] == "<chat>\n0+1|user: first step\n1+1|user: second step\n</chat>"

    # The scope the payload names opens a line inside that chat.
    halves = json.loads(engine.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 2, "scope": "Researcher"}))
    assert halves["scope"] == "Researcher"
    assert [(half["id"], half["n"], half["text"]) for half in halves["halves"]] == [
        (0, 1, "user: first step"), (1, 1, "user: second step")]
    whole = json.loads(engine.handle_tool_call(
        "optchat_zoom", {"id": 1, "n": 1, "scope": "Researcher"}))
    assert whole["text"] == "second step"


def test_unknown_or_invalid_scope_is_an_error_never_a_fallback(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("master", platform="cli")
    engine.on_turn_complete([{"role": "user", "content": "master record"}])

    # Record 0 exists in the reading log, but the scope names no child: an error,
    # never the reading log's record.
    for args in ({"id": 0, "scope": "Missing"}, {"id": 0, "scope": ""},
                 {"id": 0, "scope": 7}, {"id": "Missing", "scope": "Missing"}):
        result = json.loads(engine.handle_tool_call("optchat_zoom", args))
        assert "error" in result
    assert "error" in json.loads(engine.handle_tool_call(
        "optchat_date", {"id": 0, "scope": "Missing"}))

    # A mapped name whose chat does not exist is also an error.
    (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents" / "index.json").write_text(json.dumps({"Ghost": "ghost-session"}))
    assert "error" in json.loads(engine.handle_tool_call(
        "optchat_zoom", {"id": 0, "scope": "Ghost"}))
    assert "error" in json.loads(engine.handle_tool_call("optchat_other", {"id": 0}))


def test_delegate_result_is_zoomable_in_the_turn_it_arrives(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "find the answer"}])

    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-1", "function": {"name": "delegate_task", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "delegate-1",
              "content": json.dumps({"results": [{"task_index": 0, "status": "completed",
                                                  "summary": "found it"}]})}
    request = [{"role": "user", "content": "task"}, call, result]

    # The running turn's request carries the result: selection registers it at
    # once, so the report can be zoomed before the turn completes.
    master.select_context(request)
    assert json.loads((tmp_path / "agents" / "index.json").read_text()) == {
        "child-session": "child-session"}
    zoomed = json.loads(master.handle_tool_call("optchat_zoom", {"id": "child-session"}))
    assert "find the answer" in zoomed["chat"]
    scoped = json.loads(master.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 1, "scope": "child-session"}))
    assert scoped["text"] == "find the answer"

    # Turn completion reuses that resolution: one work record, named for the
    # registered mapping (not a second, differently named one).
    master.on_turn_complete(request)
    assert [(record.kind, record.text) for record in master.log.range(0, master.log.count() - 1)] == [
        ("user", "task"),
        ("tool", '{"name": "delegate_task", "arguments": {}}'),
        ("work", "[child-session] " + result["content"])]


def test_repeated_name_keeps_the_first_child_and_warns_once(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    for key in ("child-a", "child-b"):
        child = master.clone_for_agent()
        child.on_session_start(key, platform="subagent")
        child.on_turn_complete([{"role": "user", "content": f"work in {key}"}])

    def result(call_id, child_key):
        return {"role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"results": [{"name": "Researcher",
                                                    "child_session_id": child_key,
                                                    "summary": child_key}]})}

    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-a", "function": {"name": "delegate_task"}},
        {"id": "delegate-b", "function": {"name": "delegate_task"}}]}
    messages = [{"role": "user", "content": "task"}, call,
                result("delegate-a", "child-a"), result("delegate-b", "child-b")]
    later = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-c", "function": {"name": "delegate_task"}}]},
        result("delegate-c", "child-b")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        master.select_context(messages)
        master.on_turn_complete(messages)
        master.select_context(messages + later)
        master.on_turn_complete(messages + later)

    conflicts = [item for item in caught if "already maps" in str(item.message)]
    assert len(conflicts) == 1
    assert conflicts[0].category is RuntimeWarning
    index = json.loads((tmp_path / "agents" / "index.json").read_text(encoding="utf-8"))
    assert index["Researcher"] == "child-a"
    # The first child stays the one the name opens, by name and by scope.
    opened = json.loads(master.handle_tool_call("optchat_zoom", {"id": "Researcher"}))
    assert "work in child-a" in opened["chat"]
    scoped = json.loads(master.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 1, "scope": "Researcher"}))
    assert scoped["text"] == "work in child-a"


def test_duplicate_name_leaves_the_new_report_addressable(tmp_path):
    """P6b: a delegate result whose name is already mapped to another child must
    stay reachable. The first mapping is kept, the conflicting child receives the
    first free suffixed label, and that label is what its work record carries and
    what the model can zoom by - durable in the index, so it survives later turns
    and a restart."""
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    for key in ("child-a", "child-b"):
        child = master.clone_for_agent()
        child.on_session_start(key, platform="subagent")
        child.on_turn_complete([{"role": "user", "content": f"work in {key}"}])

    def result(call_id, child_key):
        return {"role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"results": [{"name": "Researcher",
                                                    "child_session_id": child_key,
                                                    "summary": child_key}]})}

    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-a", "function": {"name": "delegate_task"}},
        {"id": "delegate-b", "function": {"name": "delegate_task"}}]}
    messages = [{"role": "user", "content": "task"}, call,
                result("delegate-a", "child-a"), result("delegate-b", "child-b")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        master.on_turn_complete(messages)

    conflicts = [item for item in caught if "already maps" in str(item.message)]
    assert len(conflicts) == 1
    assert "'Researcher 2'" in str(conflicts[0].message)

    index = json.loads((tmp_path / "agents" / "index.json").read_text(encoding="utf-8"))
    assert index == {"Researcher": "child-a", "Researcher 2": "child-b"}

    # Each work record carries the label its child is registered under; the first
    # child keeps the original name, the second is reachable as "Researcher 2".
    records = [record.text for record in master.log.range(0, master.log.count() - 1)]
    assert records[-2].startswith("[Researcher] ")
    assert records[-1].startswith("[Researcher 2] ")

    # The original name stays an alias to the first child; the suffixed label opens
    # the new one, by name and by scope.
    assert "work in child-a" in json.loads(
        master.handle_tool_call("optchat_zoom", {"id": "Researcher"}))["chat"]
    opened = json.loads(master.handle_tool_call("optchat_zoom", {"id": "Researcher 2"}))
    assert opened["scope"] == "Researcher 2"
    assert "work in child-b" in opened["chat"]
    scoped = json.loads(master.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 1, "scope": "Researcher 2"}))
    assert scoped["text"] == "work in child-b"

    # A later report for the same child reuses its label: no new mapping, no new
    # warning, and the record still carries the label that opens it.
    later = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-c", "function": {"name": "delegate_task"}}]},
        result("delegate-c", "child-b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        master.on_turn_complete(messages + later)
    assert not [item for item in caught if "already maps" in str(item.message)]
    assert master.log.get(master.log.count() - 1).text.startswith("[Researcher 2] ")
    assert json.loads((tmp_path / "agents" / "index.json").read_text(
        encoding="utf-8")) == index
    master.log.close()

    # A restart resolves the same conflict to the same label, because the label
    # lives in the index rather than in memory.
    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("master", platform="cli")
    restarted.on_turn_complete(messages + later)  # the same transcript, resent
    extra = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "delegate-d", "function": {"name": "delegate_task"}}]},
        result("delegate-d", "child-b")]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        restarted.on_turn_complete(messages + later + extra)
    assert restarted.log.get(restarted.log.count() - 1).text.startswith("[Researcher 2] ")
    assert json.loads((tmp_path / "agents" / "index.json").read_text(
        encoding="utf-8")) == index
    assert "work in child-b" in json.loads(
        restarted.handle_tool_call("optchat_zoom", {"id": "Researcher 2"}))["chat"]
    assert "work in child-a" in json.loads(
        restarted.handle_tool_call("optchat_zoom", {"id": "Researcher"}))["chat"]
    restarted.log.close()


def test_cron_context_never_writes(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session", platform="cron")
    engine.on_turn_complete([{"role": "user", "content": "scheduled run"}])
    assert engine.log.count() == 0


def test_primary_context_writes(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session", platform="cli")
    engine.on_turn_complete([{"role": "user", "content": "hello"}])
    assert engine.log.count() == 1
    assert engine.log.get(0).kind == "user"


def test_session_start_uses_host_home_and_its_config(tmp_path, monkeypatch):
    ambient = tmp_path / "ambient"
    active = tmp_path / "active"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    (active / "optchat").mkdir(parents=True)
    (active / "optchat" / "config.json").write_text('{"node_bytes": 64}')
    engine = OptChatEngine(summarize=FakeSummarizer())

    engine.on_session_start("session", hermes_home=active)

    assert engine.root == active / "optchat"
    assert engine.log.root == active / "optchat"
    assert engine.node_bytes == 64
    assert not (ambient / "optchat").exists()


def test_explicit_root_wins_over_host_home(tmp_path):
    root = tmp_path / "explicit"
    engine = OptChatEngine(root, summarize=FakeSummarizer())

    engine.on_session_start("session", hermes_home=tmp_path / "active")

    assert engine.root == root
    assert engine.log.root == root


def test_session_start_falls_back_to_environment_home(tmp_path, monkeypatch):
    home = tmp_path / "ambient"
    monkeypatch.setenv("HERMES_HOME", str(home))
    engine = OptChatEngine(summarize=FakeSummarizer())

    engine.on_session_start("session")

    assert engine.root == home / "optchat"
    assert engine.log.root == home / "optchat"


def test_session_start_falls_back_to_default_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    engine = OptChatEngine(summarize=FakeSummarizer())

    engine.on_session_start("session")

    assert engine.root == tmp_path / ".hermes" / "optchat"
    assert engine.log.root == engine.root


def test_recovery_after_a_log_write_that_never_reached_the_derived_state(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    # A record the derived state never saw, as after a crash between the two saves.
    index = engine.log.append("note", "second")
    engine.tree.save()
    engine.view.save()
    engine.log.close()

    recovered = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    recovered.on_session_start("session")

    assert index == 1
    assert recovered.tree.get(0, index) is not None
    assert recovered.view._end() == 2
    assert "note: second" in recovered.view.render()
    recovered.log.close()


def test_recovery_requeues_a_root_whose_source_was_only_in_memory(tmp_path):
    engine = OptChatEngine(tmp_path, node_bytes=16, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    # A long record the derived state never saw: the log has it, the tree does not,
    # and its source text existed only in the crashed process's memory.
    index = engine.log.append("echo", "a tool result far longer than the node budget")
    engine.tree.save()
    engine.view.save()
    engine.log.close()

    recovered = OptChatEngine(tmp_path, node_bytes=16, summarize=FakeSummarizer())
    recovered.on_session_start("session")

    assert recovered.tree.get(0, index) is None       # queued; too long to be free
    recovered.compactor.run_pending(limit=1)
    assert recovered.tree.get(0, index) is not None   # built from the recovered source
    recovered.log.close()


def test_ingestion_resumes_across_a_restart_without_duplicating(tmp_path):
    messages = [{"role": "user", "content": "one", "message_uid": "user-1"},
                {"role": "assistant", "content": "two", "message_uid": "reply-1"}]
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete(messages)
    assert engine.log.count() == 2
    assert json.loads((tmp_path / "ingest.json").read_text())["identities"] == [
        {"kind": "uid", "value": "user-1"}, {"kind": "uid", "value": "reply-1"}]
    engine.log.close()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    restarted.on_turn_complete(messages)  # the same transcript, resent after a restart
    assert restarted.log.count() == 2     # no duplicates

    restarted.on_turn_complete(messages + [{"role": "user", "content": "three"}])
    assert [record.text for record in restarted.log.range(0, 2)] == ["one", "two", "three"]
    restarted.log.close()


def test_ingestion_resumes_at_the_right_row_after_a_changed_tail(tmp_path):
    messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete(messages)
    engine.log.close()

    # Same length, last provisional row rewritten: an older match cannot prove
    # what happened to that occurrence. Preserve the unresolved interval.
    changed = [messages[0], {"role": "assistant", "content": "TWO"}]
    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    with pytest.warns(RuntimeWarning, match="cannot find its ingestion boundary"):
        restarted.on_turn_complete(changed)
    assert [record.text for record in restarted.log.range(0, 1)] == ["one", "two"]
    assert json.loads((tmp_path / "ingest.json").read_text())["lost"] is not None
    restarted.log.close()


def test_unfindable_boundary_ingests_nothing_and_warns(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "one"}])
    engine.log.close()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    with pytest.warns(RuntimeWarning, match="cannot find its ingestion boundary"):
        restarted.on_turn_complete([{"role": "user", "content": "something else entirely"}])
    assert restarted.log.count() == 1  # nothing guessed, nothing duplicated
    restarted.log.close()


RUNTIME_HEADING = "# Hermes runtime environment"
RUNTIME_END = "<!-- End Hermes runtime environment -->"


def _host_runtime_markers(monkeypatch):
    """Install the host's own marker constants so the engine resolves them."""
    agent = types.ModuleType("agent")
    agent.__path__ = []
    builder = types.ModuleType("agent.prompt_builder")
    builder.RUNTIME_ENVIRONMENT_HEADING = RUNTIME_HEADING
    builder.RUNTIME_ENVIRONMENT_END = RUNTIME_END
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.prompt_builder", builder)


def test_selection_moves_the_host_runtime_block_after_the_view(tmp_path, monkeypatch):
    _host_runtime_markers(monkeypatch)
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    block = (f"{RUNTIME_HEADING}\n\n"
             "Session: 2026-10-11; model: m; platform: cli\n\n"
             f"{RUNTIME_END}")
    identity = "Host instructions"
    request = [{"role": "system", "content": identity + "\n\n" + block},
               {"role": "user", "content": "old question"},
               {"role": "assistant", "content": "old answer"},
               {"role": "user", "content": "new question"}]

    selected = engine.select_context(request)

    # Everything but the block stays byte for byte, and the block itself moves
    # verbatim to the user channel after the view, before this turn's messages.
    assert selected[0]["content"] == identity + "\n\n" + SYSTEM_PROMPT
    assert [message["role"] for message in selected] == [
        "system", "assistant", "user", "user"]
    assert selected[1]["content"] == engine.view.render()
    assert selected[2] == {"role": "user", "content": block}
    assert selected[3:] == request[3:]
    # The same information, exactly once: relocated, never dropped.
    assert sum(block in str(message.get("content")) for message in selected) == 1
    assert request[0]["content"] == identity + "\n\n" + block  # request not mutated


def test_selection_moves_the_runtime_block_from_the_middle_too(tmp_path, monkeypatch):
    _host_runtime_markers(monkeypatch)
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    block = f"{RUNTIME_HEADING}\n\nHost: linux\n\n{RUNTIME_END}"
    content = "Host instructions\n\n" + block + "\n\nEphemeral note"

    selected = engine.select_context([{"role": "system", "content": content},
                                      {"role": "user", "content": "task"}])

    assert selected[0]["content"] == "Host instructions\n\nEphemeral note\n\n" + SYSTEM_PROMPT
    assert selected[2] == {"role": "user", "content": block}


def test_selection_leaves_an_unmarked_block_alone(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    block = f"{RUNTIME_HEADING}\n\nHost: linux\n\n{RUNTIME_END}"
    content = "Host instructions\n\n" + block

    selected = engine.select_context([{"role": "system", "content": content},
                                      {"role": "user", "content": "task"}])

    # No host markers: change nothing; the system text keeps the block where it was.
    assert selected[0]["content"] == content + "\n\n" + SYSTEM_PROMPT
    assert [message["role"] for message in selected] == ["system", "assistant", "user"]


def test_selection_requires_the_host_end_marker(tmp_path, monkeypatch):
    _host_runtime_markers(monkeypatch)
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    content = "Host instructions\n\n" + RUNTIME_HEADING + "\n\nno end marker here"

    selected = engine.select_context([{"role": "system", "content": content},
                                      {"role": "user", "content": "task"}])

    assert selected[0]["content"] == content + "\n\n" + SYSTEM_PROMPT
    assert len(selected) == 3  # nothing was lifted without the terminator


@pytest.mark.parametrize("capture_before_start", [False, True])
@pytest.mark.parametrize("tools", [[], [
    {"type": "function", "function": {"name": "host_search",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
]])
def test_selection_captures_turn_tools_for_compactions(tmp_path, capture_before_start, tools):
    """On a route where cache_plan never fires, the tools select_context receives
    are the only carrier; a compaction must send exactly the turn's schemas."""
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, summarize=summarizer, node_bytes=16)
    tools = copy.deepcopy(tools)
    expected = copy.deepcopy(tools)
    if not capture_before_start:
        engine.on_session_start("session")
    engine.select_context([{"role": "user", "content": "task"}], tools=tools)
    assert engine.compactor.tools == expected  # Updated before any compaction.
    tools.append({"function": {"name": "later_host_mutation"}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_turn_complete([{"role": "tool", "content": "long result " * 5}])
    assert not caught  # The capture silences the missing-tools warning.
    assert summarizer.prompts
    assert all(request.tools == expected for request in summarizer.prompts)
    engine.log.close()


def test_selection_without_tools_keeps_fallback_and_warns_once(tmp_path):
    """``tools=None`` means the host has none to give: the navigation fallback
    and its one-time warning stay, and nothing is invented."""
    for name in ("first", "second"):
        summarizer = FakeSummarizer()
        engine = OptChatEngine(tmp_path / name, summarize=summarizer, node_bytes=16)
        messages = [{"role": "tool", "content": "long result " * 5}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.select_context(messages, tools=None)
            engine.on_turn_complete(messages)
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert "the only hook that carries tools" in str(caught[0].message)
        assert summarizer.prompts
        assert all(request.tools == engine.get_tool_schemas() for request in summarizer.prompts)
        engine.log.close()


def test_unsettled_view_logs_at_debug_and_still_serves_the_view(tmp_path, caplog):
    """A turn served with an unsettled view says so once, at debug, and still
    returns the view; a settled turn is silent."""
    summarizer = FakeSummarizer()

    def offline(request):
        summarizer(request)
        raise RuntimeError("offline")

    engine = OptChatEngine(tmp_path / "unsettled", node_bytes=32, summarize=offline)
    with pytest.warns(RuntimeWarning):
        engine.on_turn_complete([{"role": "tool", "content": "x" * 80} for _ in range(9)])
    assert not engine.view.settled()
    before = len(summarizer.prompts)

    with caplog.at_level(logging.DEBUG, logger="optchat.engine"):
        selected = engine.select_context([{"role": "user", "content": "next"}])

    assert len(summarizer.prompts) == before  # the turn path never starts a model call
    assert "(not summarized yet: zoom it)" in selected[1]["content"]
    unsettled = [record for record in caplog.records if "unsettled view" in record.getMessage()]
    assert unsettled and all(record.levelno == logging.DEBUG for record in unsettled)

    settled = OptChatEngine(tmp_path / "settled", summarize=FakeSummarizer())
    settled.on_turn_complete([{"role": "user", "content": "short"}])
    assert settled.view.settled()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="optchat.engine"):
        settled.select_context([{"role": "user", "content": "next"}])
    assert not [record for record in caplog.records if "unsettled view" in record.getMessage()]
    engine.log.close()
    settled.log.close()


def test_on_message_ingests_each_append_exactly_once(tmp_path):
    """The log is written as the turn unfolds, with on_turn_complete's own
    classification; the turn-boundary reconciler then adds nothing, and a
    duplicate notification is a no-op too."""
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    messages = [{"role": "user", "content": "task"}]
    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}]}
    messages.append(call)
    engine.on_message(messages, copy.deepcopy(call))
    result = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    messages.append(result)
    engine.on_message(messages, copy.deepcopy(result))
    reply = {"role": "assistant", "content": "done"}
    messages.append(reply)
    engine.on_message(messages, copy.deepcopy(reply))

    # The user row was appended without a notification; the first on_message
    # ingested it from the checkpoint boundary, not just the appended row.
    assert [(record.kind, record.text) for record in engine.log.range(0, engine.log.count() - 1)] == [
        ("user", "task"),
        ("tool", '{"name": "lookup", "arguments": {}}'),
        ("echo", "result"),
        ("unii", "done")]

    count = engine.log.count()
    engine.on_message(messages, copy.deepcopy(reply))  # duplicate notification
    assert engine.log.count() == count
    engine.on_turn_complete(messages)  # the reconciler adds nothing
    assert engine.log.count() == count
    engine.log.close()


def test_on_message_ingests_a_row_appended_without_notification(tmp_path):
    """A synthetic row the host appends between notifications is ingested from
    the checkpoint boundary when the next notification arrives."""
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": "partial"}]
    engine.on_message(messages, copy.deepcopy(messages[-1]))
    messages.append({"role": "user", "content": "[System: continue]"})  # no notification
    messages.append({"role": "assistant", "content": "final"})
    engine.on_message(messages, copy.deepcopy(messages[-1]))

    assert [record.text for record in engine.log.range(0, engine.log.count() - 1)] == [
        "task", "partial", "[System: continue]", "final"]
    engine.log.close()


def test_on_message_classifies_a_delegate_result_like_turn_complete(tmp_path):
    """The same classification, mid-turn: a delegate result becomes one work
    record and its name is registered before the turn ends."""
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "find the answer"}])

    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "delegate-1", "function": {"name": "delegate_task", "arguments": "{}"}}]}]
    master.on_message(messages, copy.deepcopy(messages[-1]))
    result = {"role": "tool", "tool_call_id": "delegate-1", "content": json.dumps(
        {"results": [{"task_index": 0, "name": "Researcher",
                      "child_session_id": "child-session", "summary": "found it"}]})}
    messages.append(result)
    master.on_message(messages, copy.deepcopy(result))

    assert [(record.kind, record.text) for record in master.log.range(0, master.log.count() - 1)] == [
        ("user", "task"),
        ("tool", '{"name": "delegate_task", "arguments": {}}'),
        ("work", "[Researcher] " + result["content"])]
    assert json.loads((tmp_path / "agents" / "index.json").read_text()) == {
        "Researcher": "child-session"}
    assert "find the answer" in json.loads(master.handle_tool_call(
        "optchat_zoom", {"id": "Researcher"}))["chat"]
    child.log.close()
    master.log.close()


def test_on_message_finds_the_appended_row_wherever_it_sits(tmp_path):
    """The notified row is located in the list it is handed - never assumed to be
    the last row - so a row appended after it waits for its own notification or
    for the turn-boundary reconciler."""
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": "answered"},
                {"role": "user", "content": "appended after, not notified"}]
    engine.on_message(messages, copy.deepcopy(messages[1]))  # the assistant row, not the tail

    assert [record.text for record in engine.log.range(0, engine.log.count() - 1)] == [
        "task", "answered"]
    engine.on_turn_complete(messages)
    assert [record.text for record in engine.log.range(0, engine.log.count() - 1)] == [
        "task", "answered", "appended after, not notified"]
    engine.log.close()


def test_restart_resumes_after_on_message_ingestion_without_gaps(tmp_path):
    """A restart after on_message-only ingestion resumes at the boundary; a row
    appended but never notified is not skipped, and a re-send duplicates nothing."""
    messages = [{"role": "user", "content": "one", "message_uid": "user-1"},
                {"role": "assistant", "content": "two", "message_uid": "reply-1"}]
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_message(messages, copy.deepcopy(messages[-1]))
    assert [record.text for record in engine.log.range(0, engine.log.count() - 1)] == ["one", "two"]
    engine.log.close()  # a crash: the turn never reached on_turn_complete

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    tail = {"role": "tool", "content": "appended but never notified"}
    restarted.on_turn_complete(messages + [tail])
    assert [record.text for record in restarted.log.range(0, restarted.log.count() - 1)] == [
        "one", "two", "appended but never notified"]  # the crash tail is not skipped
    restarted.on_turn_complete(messages + [tail])     # a re-send adds nothing
    assert restarted.log.count() == 3
    restarted.log.close()


def test_recovery_extends_the_view_when_the_tree_is_ahead_of_it(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    # The tree save succeeded; the view save did not: record 1's node is in the
    # tree, and the view still ends at id 1.
    index = engine.log.append("note", "second")
    engine.compactor.seed(index, "second", "note")
    engine.tree.save()
    engine.log.close()

    recovered = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    recovered.on_session_start("session")

    assert recovered.view._end() == 2
    assert "note: second" in recovered.view.render()
    # A later append extends the view on the id it expects instead of failing.
    index = recovered.log.append("note", "third")
    recovered.view.add_root(index)
    assert recovered.view._end() == 3
    recovered.log.close()


def test_missing_view_state_on_a_chat_with_records_is_explicit_and_writes_nothing(
        tmp_path, caplog):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    engine.log.close()
    (tmp_path / "view.json").unlink()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with caplog.at_level(logging.ERROR, logger="optchat.engine"):
        with pytest.raises(ViewStateUnavailable, match="view state is unavailable"):
            restarted.on_session_start("session")

    assert restarted.view is None
    assert restarted.log is None
    assert not (tmp_path / "view.json").exists()       # nothing is written
    loud = [record for record in caplog.records
            if "OptChat failed to start" in record.getMessage()
            and "view state is unavailable" in record.getMessage()]
    assert len(loud) == 1


def test_an_unavailable_view_state_steps_the_engine_aside_for_the_session(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    engine.log.close()
    (tmp_path / "view.json").unlink()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with pytest.raises(ViewStateUnavailable):
        restarted.on_session_start("session")

    # The selection serves nothing: the host keeps its own request.
    request = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert restarted.select_context(request) is None
    # The observation hooks no-op instead of retrying the failed start.
    restarted.on_message([{"role": "user", "content": "hi"}],
                         {"role": "user", "content": "hi"})
    restarted.on_turn_complete([{"role": "user", "content": "hi"}])
    # The navigation tools answer with an explicit error.
    payload = json.loads(restarted.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert "not active" in payload["error"]
    # A clone carries the same state instead of failing.
    clone = restarted.clone_for_agent()
    assert clone.select_context(request) is None
    # Nothing was opened, ingested, or written.
    assert restarted.log is None
    assert not (tmp_path / "view.json").exists()


def test_missing_view_state_on_an_empty_chat_is_silent(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.log.close()
    (tmp_path / "view.json").unlink()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restarted.on_session_start("session")

    assert not [item for item in caught if "view" in str(item.message)]
    assert restarted.view.render() == "<chat>\n</chat>"
    data = json.loads((tmp_path / "view.json").read_text(encoding="utf-8"))
    assert data["pairs"] == []
    assert "recovered" not in data
    restarted.log.close()


def test_missing_view_state_on_a_chat_with_records_but_no_saved_nodes_raises(tmp_path):
    # Records exist, but no tree node was ever saved and view.json is missing:
    # still an explicit outcome, never a silent empty view.
    with Log(tmp_path) as log:
        log.append("user", "first")

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with pytest.raises(ViewStateUnavailable, match=r"has 1 record\(s\)"):
        restarted.on_session_start("session")
    assert not (tmp_path / "view.json").exists()


def test_a_child_refuses_to_start_when_the_master_view_state_is_missing(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("master", platform="cli")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    engine.log.close()
    (tmp_path / "view.json").unlink()

    # A restarted master steps aside; its clone reaches the capture with no view
    # object in hand, and the missing master state is explicit, not a silent
    # fallback to the child's own empty view.
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with pytest.raises(ViewStateUnavailable):
        master.on_session_start("master", platform="cli")

    child = master.clone_for_agent()
    with pytest.raises(ViewStateUnavailable, match="view state is unavailable"):
        child.on_session_start("child-session", platform="subagent")

    assert child._session_unavailable
    assert child.select_context([{"role": "user", "content": "hi"}]) is None
    assert not (tmp_path / "agents" / "child-session").exists()
    assert not (tmp_path / "view.json").exists()


def test_scoped_and_named_navigation_do_not_seal_a_child_tree_tail(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "child step"}])
    (tmp_path / "agents" / "index.json").write_text(
        json.dumps({"child-session": "child-session"}), encoding="utf-8")
    tree_path = next((tmp_path / "agents" / "child-session" / "tree").glob("*.jsonl"))
    torn = {"l": 0, "i": 1, "text": "user: extra",
            "size": len("user: extra".encode("utf-8"))}
    with tree_path.open("ab") as stream:
        stream.write(json.dumps(torn).encode("utf-8"))
    before = tree_path.read_bytes()

    # Scoped integer navigation opens the child's chat read-only.
    scoped = json.loads(child.handle_tool_call(
        "optchat_zoom", {"id": 0, "n": 1, "scope": "child-session"}))
    assert scoped["text"] == "child step"
    # String-name zoom opens the same child's chat.
    named = json.loads(child.handle_tool_call("optchat_zoom", {"id": "child-session"}))
    assert "child step" in named["chat"]

    assert tree_path.read_bytes() == before
    child.log.close()
    master.log.close()


def test_a_child_opening_its_master_does_not_seal_the_master_tree(tmp_path):
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    master.on_turn_complete([{"role": "user", "content": "master context"}])
    master.log.close()
    tree_path = next((tmp_path / "tree").glob("*.jsonl"))
    torn = {"l": 0, "i": 1, "text": "user: extra",
            "size": len("user: extra".encode("utf-8"))}
    with tree_path.open("ab") as stream:
        stream.write(json.dumps(torn).encode("utf-8"))
    before = tree_path.read_bytes()

    # A child whose own engine never held the master's objects reads the master's
    # view (and its tree) at session start and when it navigates.
    child = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    child.on_session_start("child-session", platform="subagent")
    whole = json.loads(child.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert whole["text"] == "master context"

    assert tree_path.read_bytes() == before
    child.log.close()


def test_recovery_covers_a_refused_tail_id_and_a_later_append_lands(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "first"}])
    date = engine.log.get(0).date
    engine.log.close()
    # A corrupt line claims the next id; the log refuses it and keeps the id.
    path = tmp_path / "main" / (date[:10] + ".jsonl")
    bad = json.dumps({"id": 1, "kind": "thought", "text": "refused",
                      "size": 7, "date": date}) + "\n"
    with path.open("ab") as stream:
        stream.write(bad.encode("utf-8"))

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with pytest.warns(RuntimeWarning, match="is refused"):
        restarted.on_session_start("session")

    assert restarted.log.count() == 2     # the refused id keeps its position
    assert restarted.log.get(1) is None
    assert restarted.view._end() == 2     # the view covers the refused position
    # The next row lands on the id after the refused one, and the view grows.
    restarted.on_turn_complete([{"role": "user", "content": "first"},
                                {"role": "user", "content": "second"}])
    assert restarted.log.get(2).text == "second"
    assert restarted.view._end() == 3
    restarted.log.close()


def test_a_refused_level0_node_does_not_block_session_start_or_later_appends(tmp_path):
    with Log(tmp_path) as log:
        log.append("user", "one")
        log.append("user", "two")
    # The view state exists (empty legacy pairs); recovery extends its coverage.
    (tmp_path / "view.json").write_text("[]", encoding="utf-8")
    path = tmp_path / "tree" / "2026-10-11.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"l": 0, "i": 0, "text": "bad", "size": 999}) + "\n",
                    encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with pytest.warns(RuntimeWarning, match="is refused"):
        engine.on_session_start("session")

    assert engine.tree.refused(0, 0)
    assert engine.tree.get(0, 0) is None     # the refused node is never rebuilt
    assert engine.view._end() == 2           # the view still covers both records
    index = engine.log.append("note", "later")
    engine.view.add_root(index)              # a later append still lands
    assert index == 2
    engine.log.close()


# --- Order 2 (P1b): the durable in-flight marker and its recovery arithmetic --


def _three_call_row():
    """One host row whose classification is exactly three records."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "function": {"name": "first", "arguments": "{}"}},
        {"id": "call-2", "function": {"name": "second", "arguments": "{}"}},
        {"id": "call-3", "function": {"name": "third", "arguments": "{}"}},
    ]}


def _marker_checkpoint(row, expected, *, kind="tool"):
    """The durable checkpoint a crash mid-row leaves behind."""
    return {"session": "session", "rows": 0, "fingerprints": [],
            "inflight": {"fingerprint": _row_fingerprint(row), "count_before": 0,
                         "records": [{"kind": kind, "digest": _record_digest(kind, text)}
                                     for text in expected]},
            "lost": None, "failure": None}


def test_crash_between_two_of_three_records_resumes_at_the_third(tmp_path):
    """A crash after the second of a row's three records: the marker is durable,
    and a re-run appends exactly the third - no duplicate, no skip."""
    row = _three_call_row()
    expected = [text for _kind, text in OptChatEngine._records(row, 512)]
    assert len(expected) == 3

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    calls = {"count": 0}
    real_append = engine.log.append

    def crash_on_third(kind, text):
        if calls["count"] == 2:
            raise RuntimeError("crash between the second record and the checkpoint")
        calls["count"] += 1
        return real_append(kind, text)

    engine.log.append = crash_on_third
    with pytest.raises(RuntimeError):
        engine.on_turn_complete([row])

    # Two records are durable, and the marker written *before* the first append
    # is durable too, holding the row's identity, count_before and full sequence.
    assert engine.log.count() == 2
    assert [engine.log.get(i).text for i in range(2)] == expected[:2]
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 0 and checkpoint["inflight"]["count_before"] == 0
    assert checkpoint["inflight"]["identity"] == {
        "kind": "provisional", "value": _row_fingerprint(row)}
    assert [item["digest"] for item in checkpoint["inflight"]["records"]] == [
        _record_digest("tool", text) for text in expected]
    engine.log.close()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    restarted.on_message([row], copy.deepcopy(row))  # the host re-sends the row
    assert [restarted.log.get(i).text for i in range(3)] == expected
    assert restarted.log.count() == 3
    restarted.on_turn_complete([row])                # the reconciler adds nothing
    assert restarted.log.count() == 3
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 1 and checkpoint["inflight"] is None
    restarted.log.close()


def test_crash_mid_row_for_a_split_text_row_reassembles_exactly(tmp_path):
    """The two-of-three case on a split text row: the marker's recorded sequence
    makes the remainder exact - the reassembled message equals the original."""
    text = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango")
    row = {"role": "user", "content": text}
    first = OptChatEngine(tmp_path, node_bytes=24, summarize=FakeSummarizer())
    first.on_session_start("session")
    calls = {"count": 0}
    real_append = first.log.append

    def crash_after_two(kind, value):
        if calls["count"] == 2:
            raise RuntimeError("crash after two records")
        calls["count"] += 1
        return real_append(kind, value)

    first.log.append = crash_after_two
    with pytest.raises(RuntimeError):
        first.on_turn_complete([row])
    assert first.log.count() == 2
    first.log.close()

    restarted = OptChatEngine(tmp_path, node_bytes=24, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    restarted.on_turn_complete([row])
    records = restarted.log.range(0, restarted.log.count() - 1)
    assert all(record.kind == "user" for record in records)
    assert len({record.id for record in records}) == len(records)
    assert "".join(record.text for record in records) == text  # nothing lost
    assert restarted.log.count() == len(records)
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 1 and checkpoint["inflight"] is None
    restarted.on_turn_complete([row])
    assert restarted.log.count() == len(records)
    restarted.log.close()


def test_crash_after_all_three_records_skips_the_row_whole(tmp_path):
    """A crash between the last record and the checkpoint advance: the row wrote
    all three, so recovery skips it whole - the records are never duplicated."""
    row = _three_call_row()
    expected = [text for _kind, text in OptChatEngine._records(row, 512)]

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    real_save = engine._save_checkpoint

    def crash_before_advance(rows, recent, **kwargs):
        if rows == 1 and kwargs.get("inflight") is None:
            raise RuntimeError("crash between the last record and the advance")
        return real_save(rows, recent, **kwargs)

    engine._save_checkpoint = crash_before_advance
    with pytest.raises(RuntimeError):
        engine.on_turn_complete([row])

    assert engine.log.count() == 3
    assert [engine.log.get(i).text for i in range(3)] == expected
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 0 and checkpoint["inflight"]["count_before"] == 0
    engine.log.close()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    restarted.on_turn_complete([row])
    assert restarted.log.count() == 3
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 1 and checkpoint["inflight"] is None
    restarted.on_turn_complete([row])  # a re-send adds nothing
    assert restarted.log.count() == 3
    restarted.log.close()


def _recovery_warnings(caught):
    return [item for item in caught if "recovery failed" in str(item.message)]


def test_recovery_refuses_a_refused_position_in_the_inflight_range(tmp_path):
    """A refused position in the marker's range is an explicit recovery failure:
    its payload cannot be validated, so nothing is skipped and nothing completes."""
    row = _three_call_row()
    expected = [text for _kind, text in OptChatEngine._records(row, 512)]
    with Log(tmp_path) as log:
        log.append("tool", expected[0])
        date = log.get(0).date
        bad = json.dumps({"id": 1, "kind": "thought", "text": "refused",
                          "size": 7, "date": date}) + "\n"
        with (tmp_path / "main" / (date[:10] + ".jsonl")).open("ab") as stream:
            stream.write(bad.encode("utf-8"))
    # The view state exists (empty legacy pairs); recovery extends its coverage.
    (tmp_path / "view.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ingest.json").write_text(json.dumps(_marker_checkpoint(row, expected)),
                                          encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_session_start("session")
        engine.on_turn_complete([row])

    recovery = _recovery_warnings(caught)
    assert len(recovery) == 1 and recovery[0].category is RuntimeWarning
    assert "is refused" in str(recovery[0].message)
    # Explicit failure, never completion: no skip, no append, no rewrite.
    assert engine.log.count() == 2
    assert engine.log.get(0).text == expected[0]
    assert engine.log.get(1) is None
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == 0
    assert "is refused" in checkpoint["failure"]["reason"]
    engine.log.close()


def test_recovery_refuses_a_content_mismatch_in_the_inflight_range(tmp_path):
    """A foreign record interleaved inside the range (A, note, then a crash) is a
    content mismatch: skipping would silently lose the row's own remaining record,
    so recovery fails explicitly instead."""
    row = _three_call_row()
    expected = [text for _kind, text in OptChatEngine._records(row, 512)]
    with Log(tmp_path) as log:
        log.append("tool", expected[0])
        log.append("note", "a foreign append")  # interleaved inside the range
    # The view state exists (empty legacy pairs); recovery extends its coverage.
    (tmp_path / "view.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ingest.json").write_text(json.dumps(_marker_checkpoint(row, expected)),
                                          encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_session_start("session")
        engine.on_turn_complete([row])

    recovery = _recovery_warnings(caught)
    assert len(recovery) == 1
    assert "does not match the in-flight row's record sequence" in str(recovery[0].message)
    assert engine.log.count() == 2  # the foreign record was never skipped or deleted
    assert engine.log.get(1).kind == "note"
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert "does not match" in checkpoint["failure"]["reason"]
    engine.log.close()


def test_recovery_refuses_an_impossible_delta(tmp_path):
    """More records after the marker than the row's recorded sequence is an
    impossible delta: an append path interleaved, so the range is not owned."""
    row = _three_call_row()
    expected = [text for _kind, text in OptChatEngine._records(row, 512)]
    with Log(tmp_path) as log:
        for text in expected:
            log.append("tool", text)
        log.append("note", "one append too many")
    # The view state exists (empty legacy pairs); recovery extends its coverage.
    (tmp_path / "view.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ingest.json").write_text(json.dumps(_marker_checkpoint(row, expected)),
                                          encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_session_start("session")
        engine.on_turn_complete([row])

    recovery = _recovery_warnings(caught)
    assert len(recovery) == 1
    assert "the row records only 3" in str(recovery[0].message)
    assert engine.log.count() == 4  # untouched: never skipped, never rewritten
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert "records only 3" in checkpoint["failure"]["reason"]
    engine.log.close()


def test_recovery_refuses_a_malformed_marker_and_keeps_refusing(tmp_path):
    """A present-but-unreadable marker cannot validate anything: it is an explicit
    failure that stays recorded, never dropped as if the row were complete."""
    with Log(tmp_path) as log:
        log.append("tool", "partial record")
    # The view state exists (empty legacy pairs); recovery extends its coverage.
    (tmp_path / "view.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ingest.json").write_text(json.dumps({
        "session": "session", "rows": 0, "fingerprints": [],
        "inflight": {"fingerprint": "not-a-real-row", "count_before": 0},
        "lost": None, "failure": None}), encoding="utf-8")

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_session_start("session")
        engine.on_turn_complete([{"role": "user", "content": "next"}])
    recovery = _recovery_warnings(caught)
    assert len(recovery) == 1
    assert "marker is malformed" in str(recovery[0].message)
    assert engine.log.count() == 1  # nothing appended, nothing skipped

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_turn_complete([{"role": "user", "content": "next"}])
    assert not _recovery_warnings(caught)  # warned once per engine
    assert engine.log.count() == 1         # still refusing, never duplicating
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert "marker is malformed" in checkpoint["failure"]["reason"]
    assert checkpoint["inflight"] is not None  # kept, still unreadable
    engine.log.close()


def test_recovery_refuses_a_row_that_no_longer_recomputes_identically(tmp_path):
    """Replay must generate the same record sequence as the interrupted attempt:
    the marker recorded it, and a changed node budget makes the recomputation
    diverge, which is an explicit failure rather than a different remainder."""
    row = {"role": "user", "content": "one two three four five six seven eight nine ten"}
    first = OptChatEngine(tmp_path, node_bytes=16, summarize=FakeSummarizer())
    first.on_session_start("session")
    calls = {"count": 0}
    real_append = first.log.append

    def crash_on_second(kind, text):
        if calls["count"] == 1:
            raise RuntimeError("crash after the first record")
        calls["count"] += 1
        return real_append(kind, text)

    first.log.append = crash_on_second
    with pytest.raises(RuntimeError):
        first.on_turn_complete([row])
    written = first.log.count()
    assert written == 1
    first.log.close()

    restarted = OptChatEngine(tmp_path, node_bytes=8, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restarted.on_session_start("session")
        restarted.on_turn_complete([row])

    recovery = _recovery_warnings(caught)
    assert len(recovery) == 1
    assert "no longer recomputes" in str(recovery[0].message)
    assert restarted.log.count() == written  # the divergent remainder was not written
    checkpoint = json.loads((tmp_path / "ingest.json").read_text(encoding="utf-8"))
    assert "recomputes" in checkpoint["failure"]["reason"]
    restarted.log.close()


def test_lost_boundary_preserves_interval_across_turns_and_restart(tmp_path):
    """Replace the old test that blessed progress from the newest content anchor."""
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete([{"role": "user", "content": "one"}])
    rewritten = [{"role": "user", "content": "X"}, {"role": "user", "content": "Y"}]
    with pytest.warns(RuntimeWarning, match="cannot find its ingestion boundary"):
        engine.on_turn_complete(rewritten)
    checkpoint = json.loads((tmp_path / "ingest.json").read_text())
    assert checkpoint["lost"]["committed_count"] == 1
    assert checkpoint["lost"]["rows"] == 1
    assert "unavailable" in checkpoint["lost"]["reason"]
    later = rewritten + [{"role": "assistant", "content": "new work"}]
    engine.on_turn_complete(later)
    assert json.loads((tmp_path / "ingest.json").read_text()) == checkpoint
    assert engine.log.count() == 1
    engine.log.close()

    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    with pytest.warns(RuntimeWarning, match="unavailable"):
        restarted.on_turn_complete(later)
    assert restarted.log.count() == 1
    assert json.loads((tmp_path / "ingest.json").read_text()) == checkpoint
    restarted.log.close()


def test_dispatched_payload_with_inline_results_produces_work_records(tmp_path):
    """A delegate payload with status='dispatched' carrying completed
    ``inline_results`` is ingested as the work records it carries; a dispatched
    payload with nothing completed inline still logs nothing."""
    master = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    master.on_session_start("master", platform="cli")
    child = master.clone_for_agent()
    child.on_session_start("child-session", platform="subagent")
    child.on_turn_complete([{"role": "user", "content": "find the answer"}])

    payload = {"status": "dispatched", "mode": "background", "count": 2,
               "delegation_id": "deleg-1", "goals": ["a", "b"],
               "inline_results": [
                   {"task_index": 0, "name": "Researcher",
                    "child_session_id": "child-session", "summary": "found it"},
                   {"task_index": 1, "name": "Auditor", "summary": "checked"}]}
    messages = [{"role": "assistant", "content": None, "tool_calls": [
                    {"id": "delegate-1", "function": {"name": "delegate_task",
                                                      "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "delegate-1",
                 "content": json.dumps(payload)}]
    master.on_turn_complete(messages)

    records = master.log.range(0, master.log.count() - 1)
    work = [record for record in records if record.kind == "work"]
    assert len(work) == 2
    assert work[0].text.startswith("[Researcher] ") and "found it" in work[0].text
    assert work[1].text.startswith("[Auditor] ")
    assert json.loads((tmp_path / "agents" / "index.json").read_text(encoding="utf-8")) == {
        "Researcher": "child-session"}

    empty = {"status": "dispatched", "mode": "background", "count": 1,
             "delegation_id": "deleg-2", "goals": ["later"]}
    later = [{"role": "assistant", "content": None, "tool_calls": [
                  {"id": "delegate-2", "function": {"name": "delegate_task",
                                                    "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "delegate-2",
              "content": json.dumps(empty)}]
    master.on_turn_complete(messages + later)
    assert [record.kind for record in
            master.log.range(0, master.log.count() - 1)].count("work") == 2
    child.log.close()
    master.log.close()


def _interrupt_before_first_append(engine, messages, monkeypatch):
    original = engine.log.append

    def crash(kind, text):
        raise RuntimeError("before first append")

    monkeypatch.setattr(engine.log, "append", crash)
    with pytest.raises(RuntimeError, match="before first append"):
        engine.on_turn_complete(messages)
    monkeypatch.setattr(engine.log, "append", original)
    return json.loads((engine._chat_root / "ingest.json").read_text())


@pytest.mark.parametrize("change", ["budget", "classification", "none"])
def test_zero_delta_validates_complete_sequence(tmp_path, monkeypatch, change):
    engine = OptChatEngine(tmp_path, node_bytes=16, summarize=FakeSummarizer())
    engine.on_session_start("session")
    row = {"role": "user", "content": "one two three four five six seven eight"}
    original = _interrupt_before_first_append(engine, [row], monkeypatch)
    assert engine.log.count() == 0
    if change == "budget":
        engine.node_bytes = 8
    elif change == "classification":
        monkeypatch.setattr(engine, "_records", lambda *args: [("unii", "changed")])
    if change == "none":
        engine.on_turn_complete([row])
        checkpoint = json.loads((tmp_path / "ingest.json").read_text())
        assert checkpoint["inflight"] is None
        assert engine.log.count() == len(original["inflight"]["records"])
    else:
        with pytest.warns(RuntimeWarning, match="no longer recomputes"):
            engine.on_turn_complete([row])
        checkpoint = json.loads((tmp_path / "ingest.json").read_text())
        assert checkpoint["inflight"] == original["inflight"]
        assert engine.log.count() == 0
    engine.log.close()


@pytest.mark.parametrize("wrong_start", [0, 2])
def test_wrong_count_before_fails_against_committed_frontier(tmp_path, monkeypatch, wrong_start):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    first = {"role": "user", "content": "X", "message_uid": "first"}
    second = {"role": "user", "content": "X", "message_uid": "second"}
    engine.on_turn_complete([first])
    state = _interrupt_before_first_append(engine, [first, second], monkeypatch)
    assert state["committed_count"] == 1
    if wrong_start == 2:
        # Simulate the first record surviving before an erroneously advanced start.
        with (tmp_path / "main" / (engine.log.get(0).date[:10] + ".jsonl")).open("ab") as out:
            record = dict(engine.log.get(0).__dict__, id=1)
            out.write((json.dumps(record) + "\n").encode())
        engine.log._load_index()
    state["inflight"]["count_before"] = wrong_start
    (tmp_path / "ingest.json").write_text(json.dumps(state))
    with pytest.warns(RuntimeWarning, match="count_before.*committed"):
        engine.on_turn_complete([first, second])
    assert engine.log.count() == (2 if wrong_start == 2 else 1)
    assert json.loads((tmp_path / "ingest.json").read_text())["inflight"] == state["inflight"]
    engine.log.close()


@pytest.mark.parametrize("damage", ["missing", "json", "outer", "session", "permission"])
def test_established_checkpoint_unavailable_never_reingests(tmp_path, monkeypatch, damage):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    messages = [{"role": "user", "content": "old"}]
    engine.on_turn_complete(messages)
    path = tmp_path / "ingest.json"
    if damage == "missing":
        path.unlink()
    elif damage == "json":
        path.write_text("{broken")
    elif damage == "outer":
        path.write_text(json.dumps({"session": "session", "rows": True, "fingerprints": []}))
    elif damage == "session":
        engine.on_session_start("new-session")
    original = path.read_bytes() if path.exists() else None
    if damage == "permission":
        read_text = Path.read_text

        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("injected unreadable checkpoint")
            return read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    with pytest.warns(RuntimeWarning, match="recovery unavailable"):
        engine.on_turn_complete(messages + [{"role": "user", "content": "new"}])
    assert engine.log.count() == 1
    assert (path.read_bytes() if path.exists() else None) == original
    unavailable = (tmp_path / "ingest-unavailable.json").read_bytes()
    engine.on_turn_complete(messages)
    assert (tmp_path / "ingest-unavailable.json").read_bytes() == unavailable
    engine.log.close()
    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    with pytest.warns(RuntimeWarning, match="unavailable"):
        restarted.on_turn_complete(messages)
    assert restarted.log.count() == 1
    restarted.log.close()


def test_fresh_checkpoint_starts_without_recovery_warning(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.on_turn_complete([{"role": "user", "content": "fresh"}])
    assert not _recovery_warnings(caught)
    assert engine.log.get(0).text == "fresh"
    assert not (tmp_path / "ingest-unavailable.json").exists()
    engine.log.close()


def test_note_import_cannot_reenter_ingestion(tmp_path, monkeypatch):
    from optchat.notes import import_notes

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    append = engine.log.append
    rejected = []

    def attempt_note(kind, text):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            import_notes(engine.log, engine.compactor, engine.view, ["foreign note"])
        rejected.append(True)
        return append(kind, text)

    monkeypatch.setattr(engine.log, "append", attempt_note)
    engine.on_turn_complete([_three_call_row()])
    assert len(rejected) == 3
    assert [engine.log.get(i).kind for i in range(3)] == ["tool"] * 3
    engine.log.close()


def test_note_import_refuses_surviving_ingestion_intent(tmp_path, monkeypatch):
    from optchat.notes import import_notes

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    state = _interrupt_before_first_append(engine, [_three_call_row()], monkeypatch)
    with pytest.raises(RuntimeError, match="unresolved ingestion transaction"):
        import_notes(engine.log, engine.compactor, engine.view, ["foreign note"])
    assert engine.log.count() == 0
    assert json.loads((tmp_path / "ingest.json").read_text()) == state
    engine.log.close()


def test_completed_range_is_validated_before_marker_clear(tmp_path, monkeypatch):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    append = engine.log.append

    def substitute(kind, text):
        return append(kind, "incorrect payload")

    monkeypatch.setattr(engine.log, "append", substitute)
    with pytest.warns(RuntimeWarning, match="does not match"):
        engine.on_turn_complete([{"role": "user", "content": "expected payload"}])
    state = json.loads((tmp_path / "ingest.json").read_text())
    assert state["rows"] == 0 and state["inflight"] is not None
    assert state["committed_count"] == 0
    assert "does not match" in state["failure"]["reason"]
    assert engine.log.get(0).text == "incorrect payload"
    engine.log.close()


def test_lost_boundary_keeps_interrupted_marker_even_when_source_returns(tmp_path, monkeypatch):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    row = _three_call_row()
    original = _interrupt_before_first_append(engine, [row], monkeypatch)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        engine.on_turn_complete([{"role": "user", "content": "X"},
                                 {"role": "user", "content": "Y"}])
    state = json.loads((tmp_path / "ingest.json").read_text())
    assert state["inflight"] == original["inflight"]
    assert state["lost"] is not None
    engine.on_turn_complete([row, {"role": "user", "content": "later"}])
    assert json.loads((tmp_path / "ingest.json").read_text()) == state
    assert engine.log.count() == 0
    engine.log.close()


def test_note_thread_waits_for_completed_ingestion_transaction(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from optchat.notes import import_notes

    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    started = Event()
    seed = engine.compactor.seed
    futures = []

    def notes():
        started.set()
        return import_notes(engine.log, engine.compactor, engine.view, ["after row"])

    with ThreadPoolExecutor(max_workers=1) as executor:
        def after_append(*args):
            if not futures:
                futures.append(executor.submit(notes))
                assert started.wait(2)
                assert not futures[0].done()
            return seed(*args)

        monkeypatch.setattr(engine.compactor, "seed", after_append)
        engine.on_turn_complete([_three_call_row()])
        assert futures[0].result(timeout=2) == 1
    assert [engine.log.get(i).kind for i in range(4)] == ["tool", "tool", "tool", "note"]
    assert json.loads((tmp_path / "ingest.json").read_text())["inflight"] is None
    engine.log.close()


def test_zero_delta_validation_after_incomplete_json_tail(tmp_path, monkeypatch):
    engine = OptChatEngine(tmp_path, node_bytes=16, summarize=FakeSummarizer())
    engine.on_session_start("session")
    row = {"role": "user", "content": "one two three four five six seven eight"}
    original = _interrupt_before_first_append(engine, [row], monkeypatch)
    main = tmp_path / "main"
    main.mkdir()
    (main / "2026-10-12.jsonl").write_bytes(b'{"id":0,"kind":"user"')
    engine.log.close()
    restarted = OptChatEngine(tmp_path, node_bytes=8, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    with pytest.warns(RuntimeWarning, match="no longer recomputes"):
        restarted.on_turn_complete([row])
    assert restarted.log.count() == 0
    assert json.loads((tmp_path / "ingest.json").read_text())["inflight"] == original["inflight"]
    restarted.log.close()


# Order 2c: occurrence identity, with explicit limits for provisional rows.
def _identity_engine(root):
    engine = OptChatEngine(root, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine._capture_turn_tools([])
    return engine


@pytest.mark.parametrize("reconcile_hook", ["on_message", "on_turn_complete"])
def test_provisional_identity_upgrades_after_restart_without_new_rows(tmp_path, reconcile_hook):
    rows = [{"role": "user", "content": "task", "message_uid": "task"},
            {"role": "assistant", "content": "answer"}]
    engine = _identity_engine(tmp_path)
    engine.on_message(rows, copy.deepcopy(rows[-1]))
    state = json.loads((tmp_path / "ingest.json").read_text())
    assert state["identities"][-1] == {
        "kind": "provisional", "value": _row_fingerprint(rows[-1])}
    assert "message_uid" not in rows[-1]  # the engine never stamps the host
    engine.log.close()

    rows[-1]["message_uid"] = "answer"
    restarted = _identity_engine(tmp_path)
    if reconcile_hook == "on_message":
        restarted.on_message(copy.deepcopy(rows), copy.deepcopy(rows[-1]))
    else:
        restarted.on_turn_complete(copy.deepcopy(rows))
    assert restarted.log.count() == 2
    assert json.loads((tmp_path / "ingest.json").read_text())["identities"] == [
        {"kind": "uid", "value": "task"}, {"kind": "uid", "value": "answer"}]
    restarted.log.close()

    # Prove the upgrade was durable: matching now uses the UID, even when the
    # committed payload changed, and the new equal-content occurrence is logged.
    rows[-1]["content"] = "host revised answer"
    final = _identity_engine(tmp_path)
    final.on_turn_complete(rows + [{"role": "assistant", "content": "answer",
                                   "message_uid": "another-answer"}])
    assert [r.text for r in final.log.range(0, 2)] == ["task", "answer", "answer"]
    final.log.close()


@pytest.mark.parametrize("durable", [False, True])
def test_equal_content_occurrences_resume_by_uid_or_refuse(tmp_path, durable):
    first = {"role": "user", "content": "continue"}
    second = dict(first)
    if durable:
        first["message_uid"], second["message_uid"] = "first", "second"
    engine = _identity_engine(tmp_path)
    engine.on_turn_complete([first])
    engine.log.close()
    restarted = _identity_engine(tmp_path)
    if durable:
        restarted.on_turn_complete([first, second])
        assert [r.text for r in restarted.log.range(0, 1)] == ["continue", "continue"]
        restarted.on_turn_complete([first, second])
        assert restarted.log.count() == 2
    else:
        with pytest.warns(RuntimeWarning, match="ambiguous provisional"):
            restarted.on_turn_complete([first, second])
        assert restarted.log.count() == 1
        state = json.loads((tmp_path / "ingest.json").read_text())
        assert "ambiguous provisional" in state["failure"]["reason"]
        restarted.log.close()
        restarted = _identity_engine(tmp_path)
        # Even a subsequently unique match cannot clear the unresolved interval.
        with pytest.warns(RuntimeWarning, match="ambiguous provisional"):
            restarted.on_turn_complete([first, {"role": "assistant", "content": "later"}])
        assert restarted.log.count() == 1
        assert json.loads((tmp_path / "ingest.json").read_text()) == state
    restarted.log.close()


def test_equal_payload_replacement_with_different_uid_is_not_a_boundary(tmp_path):
    first = {"role": "user", "content": "continue", "message_uid": "first"}
    engine = _identity_engine(tmp_path)
    engine.on_turn_complete([first])
    engine.log.close()
    restarted = _identity_engine(tmp_path)
    with pytest.warns(RuntimeWarning, match="cannot find its ingestion boundary"):
        restarted.on_turn_complete([dict(first, message_uid="replacement")])
    assert restarted.log.count() == 1
    assert json.loads((tmp_path / "ingest.json").read_text())["lost"] is not None
    restarted.log.close()


@pytest.mark.parametrize("written", [0, 1, 3])
@pytest.mark.parametrize("identity", ["uid", "upgrade", "ambiguous"])
def test_interrupted_row_identity_across_restart(tmp_path, monkeypatch, written, identity):
    row = _three_call_row()
    if identity == "uid":
        row["message_uid"] = "calls"
    engine = _identity_engine(tmp_path)
    original_save = engine._save_checkpoint
    original_append = engine.log.append

    def append(kind, text):
        if engine.log.count() == written:
            raise RuntimeError("identity crash")
        return original_append(kind, text)

    def save(rows, recent, **kwargs):
        if rows == 1:
            raise RuntimeError("identity crash")
        original_save(rows, recent, **kwargs)

    monkeypatch.setattr(engine.log, "append", append)
    monkeypatch.setattr(engine, "_save_checkpoint", save)
    with pytest.raises(RuntimeError, match="identity crash"):
        engine.on_turn_complete([row])
    before = json.loads((tmp_path / "ingest.json").read_text())
    assert before["inflight"]["identity"]["kind"] == (
        "uid" if identity == "uid" else "provisional")
    engine.log.close()
    row["message_uid"] = "calls"
    rows = [row]
    if identity == "ambiguous":
        rows.append(dict(row, message_uid="other-calls"))
    restarted = _identity_engine(tmp_path)
    if identity == "ambiguous":
        with pytest.warns(RuntimeWarning, match="ambiguous provisional"):
            restarted.on_turn_complete(rows)
        after = json.loads((tmp_path / "ingest.json").read_text())
        assert after["inflight"] == before["inflight"]
        assert restarted.log.count() == written
    else:
        restarted.on_turn_complete(rows)
        after = json.loads((tmp_path / "ingest.json").read_text())
        assert after["inflight"] is None
        assert after["identities"] == [{"kind": "uid", "value": "calls"}]
        assert restarted.log.count() == 3
    restarted.log.close()


def test_uid_notification_locates_occurrence_without_content_equality(tmp_path):
    engine = _identity_engine(tmp_path)
    rows = [{"role": "assistant", "content": "same", "message_uid": "first"},
            {"role": "assistant", "content": "same", "message_uid": "second"}]
    engine.on_message(rows, dict(rows[0], content="different clone payload"))
    assert engine.log.count() == 1
    assert json.loads((tmp_path / "ingest.json").read_text())["identities"] == [
        {"kind": "uid", "value": "first"}]
    engine.on_turn_complete(rows)
    assert engine.log.count() == 2
    engine.log.close()


def test_provisional_notification_ambiguity_refuses_before_append(tmp_path):
    engine = _identity_engine(tmp_path)
    row = {"role": "assistant", "content": "same"}
    with pytest.warns(RuntimeWarning, match="ambiguous provisional"):
        engine.on_message([row, dict(row)], copy.deepcopy(row))
    assert engine.log.count() == 0
    assert (tmp_path / "ingest-unavailable.json").exists()
    engine.log.close()


def test_uid_boundary_never_calls_payload_fingerprint(tmp_path, monkeypatch):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "old", "message_uid": "first"}
    engine.on_turn_complete([row])
    engine.log.close()
    restarted = _identity_engine(tmp_path)

    def forbidden(*args):
        pytest.fail("UID boundary matching must not fingerprint payloads")

    monkeypatch.setattr("optchat.engine._row_fingerprint", forbidden)
    restarted.on_turn_complete([dict(row, content="revised")])
    assert restarted.log.count() == 1
    restarted.log.close()


def test_legacy_checkpoint_reconciles_only_as_provisional(tmp_path):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "legacy"}
    engine.on_turn_complete([row])
    path = tmp_path / "ingest.json"
    state = json.loads(path.read_text())
    state.pop("identity_version")
    state.pop("identities")
    state["fingerprints"] = [_row_fingerprint(row)]
    path.write_text(json.dumps(state))
    engine.log.close()
    restarted = _identity_engine(tmp_path)
    restarted.on_turn_complete([dict(row, message_uid="persisted")])
    assert restarted.log.count() == 1
    assert json.loads(path.read_text())["identities"] == [{"kind": "uid", "value": "persisted"}]
    restarted.log.close()


def test_duplicate_uid_candidates_refuse(tmp_path):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "old", "message_uid": "duplicate"}
    engine.on_turn_complete([row])
    with pytest.warns(RuntimeWarning, match="ambiguous uid"):
        engine.on_turn_complete([row, dict(row, content="different")])
    assert engine.log.count() == 1
    assert "ambiguous uid" in json.loads((tmp_path / "ingest.json").read_text())["failure"]["reason"]
    engine.log.close()


@pytest.mark.parametrize("change", ["reorder", "insert", "remove"])
def test_retained_occurrence_order_cannot_be_crossed(tmp_path, change):
    engine = _identity_engine(tmp_path)
    rows = [{"role": "user", "content": "one", "message_uid": "one"},
            {"role": "assistant", "content": "two", "message_uid": "two"}]
    engine.on_turn_complete(rows)
    if change == "reorder":
        changed = list(reversed(rows))
    elif change == "insert":
        changed = [rows[0], {"role": "user", "content": "gap", "message_uid": "gap"}, rows[1]]
    else:
        changed = rows[1:]
    with pytest.warns(RuntimeWarning, match="cannot find its ingestion boundary"):
        engine.on_turn_complete(changed)
    assert engine.log.count() == 2
    state = json.loads((tmp_path / "ingest.json").read_text())
    engine.on_turn_complete(rows + [{"role": "user", "content": "later", "message_uid": "later"}])
    assert json.loads((tmp_path / "ingest.json").read_text()) == state
    engine.log.close()


def test_uid_identity_does_not_bypass_inflight_payload_validation(tmp_path, monkeypatch):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "before", "message_uid": "same-occurrence"}
    before = _interrupt_before_first_append(engine, [row], monkeypatch)
    with pytest.warns(RuntimeWarning, match="no longer recomputes"):
        engine.on_turn_complete([dict(row, content="after")])
    assert engine.log.count() == 0
    assert json.loads((tmp_path / "ingest.json").read_text())["inflight"] == before["inflight"]
    engine.log.close()


def test_provisional_marker_upgrade_is_durable_before_remainder_append(tmp_path, monkeypatch):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "before"}
    _interrupt_before_first_append(engine, [row], monkeypatch)
    upgraded = _interrupt_before_first_append(engine, [dict(row, message_uid="assigned")], monkeypatch)
    assert upgraded["inflight"]["identity"] == {"kind": "uid", "value": "assigned"}
    assert "message_uid" not in row
    engine.log.close()
    restarted = _identity_engine(tmp_path)
    restarted.on_turn_complete([dict(row, message_uid="assigned")])
    assert restarted.log.count() == 1
    restarted.log.close()


@pytest.mark.parametrize("bad", [None, {"kind": "uid", "value": ""},
                                 {"kind": "content", "value": "hash"}])
def test_malformed_identity_never_falls_back_to_content(tmp_path, bad):
    engine = _identity_engine(tmp_path)
    row = {"role": "user", "content": "one", "message_uid": "one"}
    engine.on_turn_complete([row])
    path = tmp_path / "ingest.json"
    state = json.loads(path.read_text())
    state["identities"] = [bad]
    state["fingerprints"] = [_row_fingerprint(row)]
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.warns(RuntimeWarning, match="malformed checkpoint"):
        engine.on_turn_complete([row])
    assert engine.log.count() == 1 and path.read_bytes() == before
    assert (tmp_path / "ingest-unavailable.json").exists()
    engine.log.close()

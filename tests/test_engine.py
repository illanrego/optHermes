"""Host-independent tests for the OptChat adapter."""

import copy
import importlib.util
import json
import logging
import sys
import types
import warnings
from pathlib import Path

import pytest

from optchat import OptChatEngine
from optchat.compactor import Compactor
from optchat.prompts import SYSTEM_PROMPT
from optchat.store import Log
from optchat.tree import Tree
from optchat.view import View


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
        messages = [{"role": "tool", "content": "long result " * 5}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.on_turn_complete(messages)
            engine.on_turn_complete(messages + messages)
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
    messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session")
    engine.on_turn_complete(messages)
    assert engine.log.count() == 2
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

    # Same length, last row rewritten: the boundary is found one row earlier, so the
    # new text is ingested instead of the whole transcript being skipped.
    changed = [messages[0], {"role": "assistant", "content": "TWO"}]
    restarted = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    restarted.on_session_start("session")
    restarted.on_turn_complete(changed)
    assert [record.text for record in restarted.log.range(0, 2)] == ["one", "two", "TWO"]
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

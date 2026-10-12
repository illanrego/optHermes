"""The OptChat summarizer's host boundary, without a host or model call."""

import importlib
import sys
import types
from unittest.mock import patch

import pytest

from optchat.compactor import CompactionRequest, Compactor
from optchat.tree import Tree


def request(task="input", *, system="", context="", tools=None):
    return CompactionRequest(system=system, context=context, task=task, tools=tools or [])


def _fake_host(monkeypatch, call, *, provider="example", model="small"):
    agent = types.ModuleType("agent")
    agent.__path__ = []
    client = types.ModuleType("agent.auxiliary_client")
    client.call_llm = call
    client.extract_content_or_reasoning = lambda response: (
        response.get("content") or response.get("reasoning") or "")
    config = types.ModuleType("agent.auxiliary_task_config")
    config._get_auxiliary_task_config = lambda task: {"provider": provider, "model": model}
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", client)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_task_config", config)


def test_import_does_not_import_host(monkeypatch):
    sys.modules.pop("optchat.summarizer", None)
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "agent" or name.startswith("agent."):
            raise AssertionError("host imported while importing summarizer")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        importlib.import_module("optchat.summarizer")


def test_host_receives_turn_system_context_task_tools_and_configured_route(monkeypatch):
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return "first\nsecond"

    _fake_host(monkeypatch, call)
    from optchat.summarizer import make_host_summarizer

    prompt = "Compaction: café\n<input>\nwhole text\n</input>"
    tools = [{"type": "function", "function": {"name": "optchat_zoom"}}]
    assert make_host_summarizer(timeout=17)(request(prompt, system="turn system",
        context="<chat>\n0+1|prior\n</chat>", tools=tools)) == "first second"
    assert calls == [{"task": "optchat", "messages": [
        {"role": "system", "content": "turn system"},
        {"role": "assistant", "content": "<chat>\n0+1|prior\n</chat>"},
        {"role": "user", "content": prompt}],
        "tools": tools, "timeout": 17.0}]
    assert calls[0]["messages"][1]["content"].startswith("<chat>")
    assert calls[0]["messages"][1]["content"].endswith("</chat>")


@pytest.mark.parametrize("system", ["", "turn system"])
def test_no_context_omits_assistant_message(monkeypatch, system):
    calls = []
    _fake_host(monkeypatch, lambda **kwargs: calls.append(kwargs) or "line")
    from optchat.summarizer import make_host_summarizer

    make_host_summarizer()(request("task", system=system))
    expected = [{"role": "system", "content": system}] if system else []
    assert calls[0]["messages"] == expected + [{"role": "user", "content": "task"}]


def test_no_system_omits_system_but_keeps_context_separate(monkeypatch):
    calls = []
    _fake_host(monkeypatch, lambda **kwargs: calls.append(kwargs) or "line")
    from optchat.summarizer import make_host_summarizer

    context = "<chat>\n0+1|prior\n</chat>"
    make_host_summarizer()(request("task", context=context))
    assert calls[0]["messages"] == [
        {"role": "assistant", "content": context}, {"role": "user", "content": "task"}]


def test_empty_context_and_system_send_only_task(monkeypatch):
    calls = []
    _fake_host(monkeypatch, lambda **kwargs: calls.append(kwargs) or "line")
    from optchat.summarizer import make_host_summarizer

    assert make_host_summarizer()(request("task")) == "line"
    assert calls[0]["messages"] == [{"role": "user", "content": "task"}]
    assert calls[0]["tools"] is None


def test_host_failure_propagates(monkeypatch):
    def call(**_kwargs):
        raise TimeoutError("model timed out")

    _fake_host(monkeypatch, call)
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(TimeoutError, match="model timed out"):
        make_host_summarizer()(request())


def test_empty_reply_and_missing_route_fail_closed(monkeypatch):
    _fake_host(monkeypatch, lambda **_kwargs: " \n ")
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(RuntimeError, match="empty reply"):
        make_host_summarizer()(request())
    _fake_host(monkeypatch, lambda **_kwargs: "unused", model="")
    with pytest.raises(RuntimeError, match="pins a provider but no model"):
        make_host_summarizer()(request())


def test_reasoning_only_reply_raises_and_compactor_leaves_node_unbuilt(monkeypatch, tmp_path):
    _fake_host(monkeypatch, lambda **_kwargs: {
        "choices": [{"message": {"content": None, "reasoning": "private thoughts"}}]})
    from optchat.summarizer import make_host_summarizer

    summarize = make_host_summarizer()
    with pytest.raises(RuntimeError, match="empty reply"):
        summarize(request())

    tree = Tree(tmp_path, node_bytes=8)
    compactor = Compactor(tree, summarize, node_bytes=8)
    compactor.seed(0, "a source that needs a summary", "user")
    with pytest.warns(RuntimeWarning, match="empty reply"):
        assert compactor.run_pending() == 0
    assert tree.get(0, 0) is None


def test_content_wins_over_reasoning(monkeypatch):
    _fake_host(monkeypatch, lambda **_kwargs: {
        "choices": [{"message": {"content": "ordinary\nline", "reasoning": "private thoughts"}}]})
    from optchat.summarizer import make_host_summarizer

    assert make_host_summarizer()(request()) == "ordinary line"


def test_auto_provider_inherits_the_main_model(monkeypatch):
    """provider="auto" is the host's "inherit the main model": no model required."""
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return "a line"

    _fake_host(monkeypatch, call, provider="auto", model="")
    from optchat.summarizer import make_host_summarizer

    assert make_host_summarizer()(request()) == "a line"
    assert calls and calls[0]["task"] == "optchat"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_bounded(timeout):
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(ValueError):
        make_host_summarizer(timeout=timeout)

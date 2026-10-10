"""The OptChat summarizer's host boundary, without a host or model call."""

import importlib
import sys
import types
from unittest.mock import patch

import pytest


def _fake_host(monkeypatch, call, *, provider="example", model="small"):
    agent = types.ModuleType("agent")
    agent.__path__ = []
    client = types.ModuleType("agent.auxiliary_client")
    client.call_llm = call
    client.extract_content_or_reasoning = lambda response: response
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


def test_host_receives_exact_prompt_and_configured_task(monkeypatch):
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return "first\nsecond"

    _fake_host(monkeypatch, call)
    from optchat.summarizer import make_host_summarizer

    prompt = "Compaction: café\n<input>\nwhole text\n</input>"
    assert make_host_summarizer(timeout=17)(prompt) == "first second"
    assert calls == [{"task": "optchat", "messages": [{"role": "user", "content": prompt}],
                      "timeout": 17.0}]


def test_host_failure_propagates(monkeypatch):
    def call(**_kwargs):
        raise TimeoutError("model timed out")

    _fake_host(monkeypatch, call)
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(TimeoutError, match="model timed out"):
        make_host_summarizer()("input")


def test_empty_reply_and_missing_route_fail_closed(monkeypatch):
    _fake_host(monkeypatch, lambda **_kwargs: " \n ")
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(RuntimeError, match="empty reply"):
        make_host_summarizer()("input")
    _fake_host(monkeypatch, lambda **_kwargs: "unused", model="")
    with pytest.raises(RuntimeError, match="pins a provider but no model"):
        make_host_summarizer()("input")


def test_auto_provider_inherits_the_main_model(monkeypatch):
    """provider="auto" is the host's "inherit the main model": no model required."""
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return "a line"

    _fake_host(monkeypatch, call, provider="auto", model="")
    from optchat.summarizer import make_host_summarizer

    assert make_host_summarizer()("input") == "a line"
    assert calls and calls[0]["task"] == "optchat"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_bounded(timeout):
    from optchat.summarizer import make_host_summarizer

    with pytest.raises(ValueError):
        make_host_summarizer(timeout=timeout)

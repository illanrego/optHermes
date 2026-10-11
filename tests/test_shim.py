"""The temporary cache shim's installation and message-shape contract."""

import logging
import sys
import types

import pytest

from optchat.engine import OptChatEngine
from optchat import shim


def _host(monkeypatch, original=None):
    agent_module = types.ModuleType("agent")
    agent_module.__path__ = []
    loop = types.ModuleType("agent.conversation_loop")
    if original is not None:
        loop._redecorate_prompt_cache_for_provider = original
    agent_module.conversation_loop = loop
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", loop)
    return loop


def test_install_idempotent_and_uninstall_restores_identity(monkeypatch):
    calls = []
    view = {"role": "assistant", "content": "<chat>\n0+1|x\n</chat>"}
    end = {"role": "user", "content": [{"type": "text", "text": "now",
                                        "cache_control": {"type": "ephemeral"}}]}

    def original(agent, messages, *, system_message=None):
        calls.append((agent, system_message))
        return messages, None, []

    loop = _host(monkeypatch, original)
    engine = OptChatEngine()
    agent = types.SimpleNamespace(context_compressor=engine, _use_prompt_caching=True)
    shim.install(engine)
    wrapped = loop._redecorate_prompt_cache_for_provider
    shim.install(engine)
    assert loop._redecorate_prompt_cache_for_provider is wrapped
    result = wrapped(agent, [view, end], system_message="system")
    assert calls == [(agent, "system")]
    assert result[0][0]["content"] == [
        {"type": "text", "text": "<chat>\n"},
        {"type": "text", "text": "0+1|x\n", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "</chat>"}]
    assert result[0][1] == end
    assert isinstance(view["content"], str)
    assert end["content"][0]["cache_control"] == {"type": "ephemeral"}
    shim.uninstall()
    assert loop._redecorate_prompt_cache_for_provider is original
    shim.uninstall()
    assert loop._redecorate_prompt_cache_for_provider is original


def test_inactive_engine_does_not_mark(monkeypatch):
    def original(_agent, messages):
        return messages, None, []

    loop = _host(monkeypatch, original)
    shim.install(object())
    view = {"role": "assistant", "content": "<chat>\nx\n</chat>"}
    loop._redecorate_prompt_cache_for_provider(
        types.SimpleNamespace(context_compressor=object(), _use_prompt_caching=True), [view])
    assert isinstance(view["content"], str)
    shim.uninstall()


def test_upstream_seam_needs_no_shim(monkeypatch):
    def original(_agent, messages):
        return messages, None, []

    loop = _host(monkeypatch, original)
    context_engine = types.ModuleType("agent.context_engine")
    context_engine.ContextEngine = type("ContextEngine", (), {"cache_plan": lambda self: None})
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine)
    shim.install(object())
    assert loop._redecorate_prompt_cache_for_provider is original


def test_missing_private_function_warns_once_and_noops(monkeypatch, caplog):
    loop = _host(monkeypatch)
    if hasattr(shim._warn_missing_once, "warned"):
        del shim._warn_missing_once.warned
    with caplog.at_level(logging.WARNING, logger="optchat.shim"):
        shim.install(object())
        shim.install(object())
    assert not hasattr(loop, "_redecorate_prompt_cache_for_provider")
    assert sum("cache decorator is unavailable" in record.message for record in caplog.records) == 1


def test_current_engine_replaces_install_time_engine(monkeypatch):
    tools = [{"type": "function", "function": {"name": "current_tool"}}]

    def original(_agent, messages):
        return messages, None, tools

    loop = _host(monkeypatch, original)
    installed, current = OptChatEngine(), OptChatEngine()
    shim.install(installed)
    agent = types.SimpleNamespace(context_compressor=current, _use_prompt_caching=True)
    result = loop._redecorate_prompt_cache_for_provider(
        agent, [{"role": "assistant", "content": "<chat>\n0+1|second session\n</chat>"},
                {"role": "user", "content": "now"}])
    assert "cache_control" in result[0][0]["content"][-2]
    assert current._turn_tools == tools
    assert current._has_turn_tools
    assert not installed._has_turn_tools
    shim.uninstall()


@pytest.mark.parametrize("cache_on, prepared, provider", [
    (False, None, "automatic"), (True, {}, "moa"),
])
def test_host_policy_bypass_keeps_string_view(monkeypatch, cache_on, prepared, provider):
    def original(_agent, messages, **kwargs):
        return messages, kwargs.get("moa_prepared"), []

    loop = _host(monkeypatch, original)
    engine = OptChatEngine()
    shim.install(engine)
    agent = types.SimpleNamespace(context_compressor=engine, _use_prompt_caching=cache_on,
                                  provider=provider)
    view = {"role": "assistant", "content": "<chat>\n0+1|x\n</chat>"}
    result = loop._redecorate_prompt_cache_for_provider(agent, [view], moa_prepared=prepared)
    assert result == ([view], prepared, [])
    assert isinstance(result[0][0]["content"], str)
    assert not engine._has_turn_tools
    shim.uninstall()

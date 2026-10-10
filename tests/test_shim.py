"""The temporary cache shim's installation and message-shape contract."""

import logging
import sys
import types

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
    engine = object()
    agent = types.SimpleNamespace(context_compressor=engine)
    shim.install(engine)
    wrapped = loop._redecorate_prompt_cache_for_provider
    shim.install(engine)
    assert loop._redecorate_prompt_cache_for_provider is wrapped
    result = wrapped(agent, [view, end], system_message="system")
    assert calls == [(agent, "system")]
    assert result[0][0]["content"] == [{"type": "text", "text": "<chat>\n0+1|x\n</chat>",
                                        "cache_control": {"type": "ephemeral"}}]
    assert result[0][1] is end
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
        types.SimpleNamespace(context_compressor=object()), [view])
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

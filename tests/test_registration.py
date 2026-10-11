"""Registration and startup failures are loud, never a silent degrade.

The host logs a failed engine load at debug level and silently keeps its built-in
compressor, so the lost guarantee must be named by the engine itself.
"""

import logging

import pytest

import optchat
import optchat.shim
import optchat.summarizer
from optchat import OptChatEngine


class FakeSummarizer:
    def __call__(self, request):
        return "summary"


class Collector:
    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine


class Manager:
    def __init__(self, home):
        self.home_path = home


class Ctx(Collector):
    def __init__(self, home):
        super().__init__()
        self._manager = Manager(home)


def test_missing_collector_names_the_lost_guarantee(caplog):
    with caplog.at_level(logging.ERROR, logger="optchat"):
        optchat.register(object())
    assert "OptChat is not active" in caplog.text
    assert "compress it lossily" in caplog.text
    assert "register_context_engine" in caplog.text


def test_engine_construction_failure_names_the_lost_guarantee(caplog, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no engine today")

    monkeypatch.setattr(optchat, "OptChatEngine", boom)
    with caplog.at_level(logging.ERROR, logger="optchat"):
        optchat.register(Collector())
    assert "OptChat is not active" in caplog.text
    assert "no engine today" in caplog.text


def test_refused_registration_names_the_lost_guarantee(caplog):
    class Refusing(Collector):
        def register_context_engine(self, engine):
            raise RuntimeError("refused")

    with caplog.at_level(logging.ERROR, logger="optchat"):
        optchat.register(Refusing())
    assert "OptChat is not active" in caplog.text
    assert "refused" in caplog.text


def test_summarizer_failure_is_loud_but_still_registers(caplog, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no aux route")

    monkeypatch.setattr(optchat.summarizer, "make_host_summarizer", boom)
    collector = Collector()
    with caplog.at_level(logging.ERROR, logger="optchat"):
        optchat.register(collector)
    assert isinstance(collector.engine, OptChatEngine)
    assert "cannot summarize" in caplog.text
    assert "no aux route" in caplog.text


def test_shim_failure_is_loud_but_still_registers(caplog, monkeypatch):
    def boom(engine):
        raise RuntimeError("shim exploded")

    monkeypatch.setattr(optchat.shim, "install", boom)
    collector = Collector()
    with caplog.at_level(logging.ERROR, logger="optchat"):
        optchat.register(collector)
    assert isinstance(collector.engine, OptChatEngine)
    assert "cache shim not installed" in caplog.text
    assert "shim exploded" in caplog.text


def test_register_uses_the_manager_home(tmp_path):
    collector = Ctx(tmp_path)
    optchat.register(collector)
    assert isinstance(collector.engine, OptChatEngine)
    assert collector.engine.root == tmp_path / "optchat"


def test_startup_failure_is_loud_and_reraised_once(tmp_path, caplog):
    root = tmp_path / "not-a-dir"
    root.write_text("a file, not a chat directory", encoding="utf-8")
    engine = OptChatEngine(root, summarize=FakeSummarizer())
    with caplog.at_level(logging.ERROR, logger="optchat.engine"):
        with pytest.raises(OSError):
            engine.on_session_start("session")
        with pytest.raises(OSError):
            engine.on_session_start("session")
    assert caplog.text.count("OptChat failed to start") == 1
    assert "no-loss guarantee is not in effect" in caplog.text


def test_unsafe_subagent_session_id_is_loud(tmp_path, caplog):
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    with caplog.at_level(logging.ERROR, logger="optchat.engine"):
        with pytest.raises(ValueError):
            engine.on_session_start("../escape", platform="subagent")
    assert "OptChat failed to start" in caplog.text

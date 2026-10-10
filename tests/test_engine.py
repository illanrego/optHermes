"""Host-independent tests for the OptChat adapter."""

import copy
import json
import logging

from optchat import OptChatEngine
from optchat.prompts import SYSTEM_PROMPT
from optchat.tree import Tree


class FakeSummarizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "summary"


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
    clone.on_turn_complete([{"role": "user", "content": "from clone"}])
    assert engine.log.get(3).text == "from clone"
    assert engine.should_compress() is False
    assert {schema["function"]["name"] for schema in engine.get_tool_schemas()} == {
        "optchat_zoom", "optchat_date"}


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


def test_large_message_uses_summarizer_and_raw_message_survives(tmp_path):
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    text = "x" * 80
    engine.on_turn_complete([{"role": "user", "content": text}])
    assert engine.log.get(0).text == text
    assert engine.tree.get(0, 0).text == "summary"
    assert len(summarizer.prompts) == 1


def _short_chat(n):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_selection_never_starts_a_model_call(tmp_path):
    """select_context runs synchronously inside the host's request, and a summarizer call
    in flight cannot be interrupted by the deadline. So it must not start one: anything
    needing a model is left to the background, and the view may hold a placeholder."""
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    engine.on_turn_complete([{"role": "user", "content": "x" * 80} for _ in range(9)])
    before = len(summarizer.prompts)
    assert not engine.view.settled()

    selected = engine.select_context([{"role": "user", "content": "next"}])

    assert len(summarizer.prompts) == before
    assert "(not summarized yet: zoom it)" in selected[1]["content"]


def test_end_of_turn_compaction_settles_the_view(tmp_path):
    """v1's admission is delivered by the end-of-turn background pass, not by the inline
    settle: after on_turn_complete the view has no placeholder, so the next turn starts
    settled. The pass is bounded at 8 model nodes per turn."""
    summarizer = FakeSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    engine.on_turn_complete([{"role": "user", "content": "x" * 80}])

    assert engine.tree.get(0, 0).text == "summary"
    assert engine.view.settled()
    assert len(summarizer.prompts) == 1


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
    engine.on_turn_complete([{"role": "user", "content": secret},
                             {"role": "assistant", "content": "I will remember it."}])
    selected = engine.select_context([
        {"role": "system", "content": "host"},
        {"role": "user", "content": secret},
        {"role": "assistant", "content": "I will remember it."},
        {"role": "user", "content": "What was the codename?"}])
    assert len(selected) == 3
    assert "ORCHID-17" not in selected[1]["content"]
    assert "zoom" in selected[1]["content"]
    recovered = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert "ORCHID-17" in recovered["text"]


def test_subagent_context_reads_but_never_writes(tmp_path):
    """A subagent must not append to the chat: only its final report returns as a
    ``work`` message to the master. The host reports platform="subagent"."""
    engine = OptChatEngine(tmp_path, summarize=FakeSummarizer())
    engine.on_session_start("session", platform="subagent")
    engine.on_turn_complete([{"role": "user", "content": "subagent step"},
                             {"role": "assistant", "content": "subagent reply"}])
    assert engine.log.count() == 0
    # It still reads: a subagent's first message is the view.
    assert engine.select_context([{"role": "system", "content": "host"},
                                  {"role": "user", "content": "task"}]) is not None


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

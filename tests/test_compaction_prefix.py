"""Measure engine request JSON, before host sanitization or provider serialization.

Run with ``python3 -m pytest -q -s tests/test_compaction_prefix.py`` for numbers.
The format is compact JSON, ensure_ascii=False, UTF-8, keys tools then messages;
message/schema key order is preserved. Routing/timeout arguments are excluded:
these are the engine's prompt-bearing fields, not a claimed provider wire body.
"""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace

from optchat.engine import OptChatEngine


def _serialize(messages, tools):
    return json.dumps({"tools": tools, "messages": messages}, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _shared(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def _report(label, left, right):
    assert left["tools"] == right["tools"]
    a, b = (_serialize(item["messages"], item["tools"]) for item in (left, right))
    overlap = _shared(a, b)
    location = ("identical serialization (end)" if a == b else
                f"messages array boundary after {min(len(left['messages']), len(right['messages']))} rows")
    for index, (first, second) in enumerate(zip(left["messages"], right["messages"])):
        if first == second:
            continue
        assert first["role"] == second["role"]
        content_offset = _shared(first["content"].encode("utf-8"),
                                 second["content"].encode("utf-8"))
        location = f"messages[{index}].content UTF-8 byte {content_offset}"
        break
    print(f"{label}: shared={overlap} bytes; sizes={len(a)}/{len(b)}; "
          f"divergence={location}; next={a[overlap:overlap + 24]!r} / "
          f"{b[overlap:overlap + 24]!r}")
    return overlap


def test_actual_turn_compaction_prefixes_across_merge_and_live_advance(tmp_path, monkeypatch):
    # Only the host/model boundary is fake. The engine, ingestion, tree, maintained
    # contexts, frozen selection and summarizer envelope are the production code.
    calls = []
    client = ModuleType("agent.auxiliary_client")
    client.call_llm = lambda **kwargs: calls.append(copy.deepcopy(kwargs)) or "s" * 512
    client.extract_content_or_reasoning = lambda response: response["content"]
    config = ModuleType("agent.auxiliary_task_config")
    config._get_auxiliary_task_config = lambda _: {"provider": "auto", "model": ""}
    agent = ModuleType("agent")
    agent.__path__ = []
    for name, module in (("agent", agent), (client.__name__, client),
                         (config.__name__, config)):
        monkeypatch.setitem(sys.modules, name, module)
    from optchat.summarizer import make_host_summarizer

    engine = OptChatEngine(tmp_path, summarize=make_host_summarizer())
    engine.on_session_start("prefix-session")
    tools = engine.get_tool_schemas()
    engine._capture_turn_tools(tools)

    def row(number, role="assistant"):
        # Multibyte content makes byte-vs-character mistakes visible. Each root
        # fits 512 bytes; each pair requires a model summary at the real budget.
        return {"role": role, "message_uid": f"row-{number}",
                "content": f"record {number:03d} café " + "x" * 470}

    def select(rows, admitted, turn):
        source_index = next(i for i, item in enumerate(rows)
                            if item["message_uid"] == admitted["message_uid"])
        api = [{"role": "system", "content": "Prefix comparison host."}] + [
            {key: value for key, value in item.items() if key != "message_uid"}
            for item in rows]
        boundary = SimpleNamespace(session_id="prefix-session", turn_id=turn,
                                   reading_scope_id="main", incoming_uid=admitted["message_uid"],
                                   request_start=source_index + 1)
        selected = engine.select_context(api, conversation_messages=rows, tools=tools,
                                         turn_boundary=boundary)
        assert selected is not None
        assert engine.selection_status["conformance"] == "agrees"
        assert selected[2:] == api[source_index + 1:]
        return {"messages": selected, "tools": copy.deepcopy(tools)}

    def compact(target):
        # Construct a fresh real job and capture exactly what make_host_summarizer
        # passes to call_llm. Do not commit its synthetic reply: this leaves the
        # target pending so case (c) can issue a fresh request for that same node.
        assert engine.tree.get(*target) is None
        request = engine.compactor._request(*target)
        assert request is not None
        assert request.system == engine._turn_system
        engine.summarize(request)
        sent = calls[-1]
        assert sent["messages"] == [
            {"role": "system", "content": request.system},
            {"role": "assistant", "content": request.context},
            {"role": "user", "content": request.task}]
        assert sent["tools"] == tools
        return sent

    try:
        # Build older parents through the normal end-of-turn drain, leaving the
        # last sibling pair pending. No view pairs or contexts are manufactured.
        rows = [row(i) for i in range(58)]
        engine.on_turn_complete(rows)
        rows.extend([row(58), row(59)])
        engine.on_message(rows, rows[-1])
        admitted = row(60, "user")
        rows.append(admitted)
        turn_before = select(rows, admitted, "first")
        compact_before = compact((1, 29))  # covers 58 and 59, context ends at 60
        assert compact_before["messages"][1] == turn_before["messages"][1]
        a = _report("(a) turn -> compaction, same state", turn_before, compact_before)

        # The live view advances within this logical turn; its selected view is
        # frozen. The fresh target extends the maintained context past 32 KB and
        # merges it using already-built older parents, down to the 16 KB floor.
        rows.extend(row(i) for i in range(61, 66))
        engine.on_message(rows, rows[-1])
        assert len(engine.view.render().encode("utf-8")) > 32_000
        compact_merged = compact((1, 32))  # covers 64 and 65, context ends at 66
        merged = compact_merged["messages"][1]["content"]
        assert len(merged.encode("utf-8")) <= 16_000
        assert not merged.startswith("<chat>\n0+1|")
        assert engine.view.render().startswith("<chat>\n0+1|")
        turn_after = select(rows, admitted, "first")
        assert turn_before["messages"][1] == turn_after["messages"][1]
        b1 = _report("(b) turn -> merged compaction", turn_before, compact_merged)
        b2 = _report("(b) merged compaction -> same-turn selection", compact_merged, turn_after)
        assert b1 == b2 < a
        _report("(b) turn -> same-turn selection", turn_before, turn_after)

        # A new turn can select the now-advanced live view. Reissuing the earlier
        # still-pending job must retain its original context, not this live view.
        next_admitted = row(66, "user")
        rows.append(next_admitted)
        turn_new = select(rows, next_admitted, "second")
        compact_fresh = compact((1, 29))
        assert turn_new["messages"][1] != turn_before["messages"][1]
        assert compact_fresh == compact_before
        c = _report("(c) advanced turn -> fresh original target", turn_new, compact_fresh)
        assert b1 < c < a
        _report("(c) original compaction -> fresh original target", compact_before, compact_fresh)
        _report("(c) merged compaction -> fresh original target", compact_merged, compact_fresh)
        print("Engine JSON only; live cache_read_tokens for turn->turn, "
              "turn->compaction and compaction->compaction remain unverified.")
    finally:
        engine.log.close()

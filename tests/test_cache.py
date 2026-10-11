"""Complete view boundaries, host prefix preservation, and request-wide budget."""

import copy

import pytest

from optchat.cache import plan_view_cache


def marked(text, ttl="1h"):
    return {"type": "text", "text": text,
            "cache_control": {"type": "ephemeral", "ttl": ttl}}


def mark_positions(messages, tools):
    positions = [("tools", i) for i, tool in enumerate(tools) if "cache_control" in tool]
    for i, message in enumerate(messages):
        if "cache_control" in message:
            positions.append((i, "message"))
        if isinstance(message.get("content"), list):
            positions.extend((i, j) for j, part in enumerate(message["content"])
                             if "cache_control" in part)
    return positions


def host_request(*, tool_prefix=False, lines=2):
    system = [marked("static")]
    system.append({"type": "text", "text": "runtime"} if tool_prefix else marked("runtime"))
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    if tool_prefix:
        tools[0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    view = "<chat>\n" + "".join(f"{i}+1|line {i}\n" for i in range(lines)) + "</chat>"
    return [{"role": "system", "content": system},
            {"role": "assistant", "content": [marked(view)]},
            {"role": "user", "content": [marked("now")]}], tools


@pytest.mark.parametrize("tool_prefix", [False, True])
def test_host_prefixes_survive_with_same_ttl_and_no_input_mutation(tool_prefix):
    messages, tools = host_request(tool_prefix=tool_prefix)
    before = copy.deepcopy((messages, tools))
    planned, planned_tools = plan_view_cache(messages, tools)
    assert planned[0] == messages[0]
    assert planned_tools == tools
    assert len(mark_positions(planned, planned_tools)) == 4
    assert planned[1]["content"][-2] == marked("1+1|line 1\n")
    assert planned[1]["content"][-1] == {"type": "text", "text": "</chat>"}
    assert planned[-1] == messages[-1]
    assert (messages, tools) == before


def test_growth_keeps_old_marked_boundary_at_same_block_position():
    previous, _ = plan_view_cache(*host_request(lines=250))
    grown, _ = plan_view_cache(*host_request(lines=251))
    old_blocks, new_blocks = previous[1]["content"], grown[1]["content"]
    old_mark = next(i for i, block in enumerate(old_blocks) if "cache_control" in block)
    assert old_mark == 250
    assert "cache_control" in new_blocks[old_mark + 1]
    assert [{k: v for k, v in block.items() if k != "cache_control"}
            for block in old_blocks[:old_mark + 1]] == new_blocks[:old_mark + 1]
    assert new_blocks[old_mark]["text"] == old_blocks[old_mark]["text"]
    # The old checkpoint remains in the new breakpoint's lookback window.
    assert len(new_blocks) - 2 - old_mark < 20


def test_unicode_text_is_preserved_and_replanning_is_idempotent():
    messages, tools = host_request()
    text = "<chat>\n0+1|snowman ☃ and separator \u2028 inside a line\n</chat>"
    messages[1]["content"] = [marked(text[:15]), {"type": "text", "text": text[15:]}]
    planned, planned_tools = plan_view_cache(messages, tools)
    assert "".join(part["text"] for part in planned[1]["content"]) == text
    assert len(planned[1]["content"]) == 3
    assert plan_view_cache(planned, planned_tools) == (planned, planned_tools)


@pytest.mark.parametrize("extra_marks", [0, 1, 2, 8, 30])
@pytest.mark.parametrize("tool_prefix", [False, True])
def test_budget_evicts_only_excess_conversation_marks(extra_marks, tool_prefix):
    messages, tools = host_request(tool_prefix=tool_prefix)
    messages[2:2] = [{"role": "user", "content": [marked(str(i))]}
                     for i in range(extra_marks)]
    planned, planned_tools = plan_view_cache(messages, tools)
    assert len(mark_positions(planned, planned_tools)) <= 4
    assert planned[0] == messages[0]
    assert planned_tools == tools
    assert "cache_control" in planned[-1]["content"][-1]


@pytest.mark.parametrize("tail_parts, has_view_mark", [(18, True), (19, False), (40, False)])
def test_lookback_counts_content_blocks_not_message_rows(tail_parts, has_view_mark):
    messages, tools = host_request()
    messages[-1]["content"] = [{"type": "text", "text": str(i)} for i in range(tail_parts)]
    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    planned, planned_tools = plan_view_cache(messages, tools)
    assert any("cache_control" in part for part in planned[1]["content"]) is has_view_mark
    assert "cache_control" in planned[-1]["content"][-1]
    assert len(mark_positions(planned, planned_tools)) <= 4


def test_tool_calls_also_count_towards_lookback():
    messages, tools = host_request()
    messages.insert(2, {"role": "assistant", "content": "work",
                        "tool_calls": [{"id": str(i)} for i in range(18)]})
    planned, _ = plan_view_cache(messages, tools)
    assert all("cache_control" not in part for part in planned[1]["content"])


@pytest.mark.parametrize("native", [False, True])
def test_tool_end_keeps_host_marker_location(native):
    messages, tools = host_request(tool_prefix=native)
    end = {"role": "tool", "tool_call_id": "call", "content": "result"}
    if native:
        end["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    else:
        end["content"] = [marked("result")]
    messages.append(end)
    planned, planned_tools = plan_view_cache(messages, tools)
    assert planned[-1] == end
    assert len(mark_positions(planned, planned_tools)) == 4


def test_unmarked_tool_result_stays_unmarked():
    messages, tools = host_request()
    messages.append({"role": "tool", "tool_call_id": "call", "content": "result"})
    planned, _ = plan_view_cache(messages, tools)
    assert planned[-1] == messages[-1]
    assert "cache_control" in planned[-2]["content"][-1]


def test_scaffold_end_gets_request_endpoint_within_budget():
    messages, tools = host_request()
    messages[-1]["content"] = [marked("scaffold"), {"type": "text", "text": "tail"}]
    planned, planned_tools = plan_view_cache(messages, tools)
    assert "cache_control" in planned[-1]["content"][-1]
    assert len(mark_positions(planned, planned_tools)) == 4


@pytest.mark.parametrize("content", ["ordinary assistant text", "<chat>\nunfinished",
                                     [{"type": "image_url", "image_url": {"url": "x"}}]])
def test_no_view_returns_none(content):
    messages = [{"role": "assistant", "content": content}]
    assert plan_view_cache(messages, []) is None


def test_empty_view_has_no_complete_line_mark():
    planned, _ = plan_view_cache(*host_request(lines=0))
    assert planned[1]["content"] == [{"type": "text", "text": "<chat>\n"},
                                      {"type": "text", "text": "</chat>"}]


def test_undecorated_request_and_view_only_request():
    view = {"role": "assistant", "content": "<chat>\n0+1|x\n</chat>"}
    planned, _ = plan_view_cache([view, {"role": "user", "content": "now"}], [])
    assert mark_positions(planned, []) == [(0, 1), (1, 0)]
    planned, _ = plan_view_cache([view], [])
    assert mark_positions(planned, []) == [(0, 1), (0, 2)]

"""Version-specific seam-host probes; no provider or live-agent execution.

Run real SQLite persistence and the actual hook/clone function bodies in an
isolated subprocess. Extracting those bodies avoids conversation_loop's startup
side effects; the cache policy setup alone is stubbed, not the hook or clone.
"""

import ast
import inspect
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("hook", ["on_message", "on_turn_complete", "select_context", "cache_plan"])
def test_seam_host_uid_at_each_hook(tmp_path, hook):
    host = Path(__file__).resolve().parents[2] / "opthermes-seam"
    if not (host / "agent" / "conversation_loop.py").is_file():
        pytest.skip("version-specific probe requires the sibling opthermes-seam checkout")
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), str(host), str(tmp_path), hook],
        env={"HERMES_HOME": str(tmp_path / "host-home"),
             "PYTHONPATH": os.pathsep.join(sys.path),
             "PYTEST_CURRENT_TEST": f"test_seam_host_uid_at_each_hook[{hook}]"},
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{hook}: identity assertions passed" in result.stdout


def _probe(host, root, hook):
    sys.path.insert(0, str(host))
    from hermes_state import SessionDB
    from agent.message_metadata import append_message, without_persistence_fields
    from agent.transcript_repair import sync_flushed_message_markers

    live = []
    append_message(live, {"role": "user", "content": "task"})
    append_message(live, {"role": "assistant", "content": "answer"})
    assert all("message_uid" not in row for row in live)
    batch = [dict(row) for row in live]
    db = SessionDB(root / "state.db")
    try:
        db.create_session("probe", "cli")
        assert db.append_messages_batch("probe", batch) == 2
        uids = [row["message_uid"] for row in batch]
        assert len(set(uids)) == 2 and all(isinstance(uid, str) and uid for uid in uids)
        sync_flushed_message_markers(live, batch)
        assert [row["message_uid"] for row in live] == uids
        restored = db.get_messages_as_conversation("probe")
        assert [row["message_uid"] for row in restored] == uids
    finally:
        db.close()

    # Compile the real host definitions, preserving their branches and clones.
    wanted = {"_clone_message_for_send", "_engine_overrides_hook", "_engine_hook_accepts_kwarg",
              "_apply_context_engine_selection", "_notify_context_engine_turn_complete",
              "_notify_context_engine_message", "_redecorate_prompt_cache_for_provider"}
    source = host / "agent" / "conversation_loop.py"
    parsed = ast.parse(source.read_text())
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in wanted]
    assert {node.name for node in functions} == wanted
    namespace = {"inspect": inspect, "without_persistence_fields": without_persistence_fields,
                 "logger": logging.getLogger("identity-probe")}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    calls = []

    class Observer:
        def on_message(self, messages, message):
            calls.append((messages, message))

        def on_turn_complete(self, messages, **kwargs):
            calls.append(messages)

        def select_context(self, request, **kwargs):
            calls.append((request, kwargs))
            return None

        def cache_plan(self, messages, tools):
            calls.append(messages)
            return None

    agent = types.SimpleNamespace(context_compressor=Observer(), session_id="probe")
    logger = namespace["logger"]
    if hook == "on_message":
        namespace["_notify_context_engine_message"](agent, restored, restored[-1], logger=logger)
        messages, notified = calls.pop()
        assert [row["message_uid"] for row in messages] == uids
        assert notified["message_uid"] == uids[-1] and notified is not restored[-1]
        # Real append helper does not mint a UID before the notification.
        fresh = append_message(restored, {"role": "tool", "content": "fresh result"})
        namespace["_notify_context_engine_message"](agent, restored, fresh, logger=logger)
        messages, notified = calls.pop()
        assert "message_uid" not in notified and "message_uid" not in messages[-1]
    elif hook == "on_turn_complete":
        namespace["_notify_context_engine_turn_complete"](agent, restored, logger=logger)
        messages = calls.pop()
        assert [row["message_uid"] for row in messages] == uids
        assert all(a is not b for a, b in zip(messages, restored))
    else:
        api = [without_persistence_fields(row) for row in restored]
        assert all("message_uid" not in row for row in api)
        if hook == "select_context":
            namespace["_apply_context_engine_selection"](
                agent, api, restored, restored[0], logger=logger, tools=[])
            request, kwargs = calls.pop()
            assert all("message_uid" not in row for row in request)
            assert [row["message_uid"] for row in kwargs["conversation_messages"]] == uids
            assert kwargs["incoming_message"]["message_uid"] == uids[0]
            assert kwargs["incoming_message"] is not restored[0]
        else:
            # Exercise the actual cache hook with a simple policy. Neither the
            # policy stubs nor this API-only hook can restore stripped UIDs.
            namespace.update({
                "strip_anthropic_cache_control": lambda messages: None,
                "strip_anthropic_tool_cache_control": lambda tools: tools,
                "_ensure_cached_system_prompt_static": lambda *args, **kwargs: None,
                "effective_cache_ttl": lambda *args, **kwargs: "5m",
                "build_prompt_cache_plan": lambda messages, tools, **kwargs:
                    types.SimpleNamespace(messages=messages, tools=tools),
            })
            agent._use_prompt_caching = True
            agent._use_native_cache_layout = False
            agent._cache_ttl = "5m"
            agent.provider, agent.model = "anthropic", "probe"
            namespace["_redecorate_prompt_cache_for_provider"](agent, api, tools_for_api=[])
            assert all("message_uid" not in row for row in calls.pop())
    assert not calls
    print(f"{hook}: identity assertions passed")


if __name__ == "__main__":
    _probe(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])

"""Media stays recoverable outside the text-only log, tree, and view."""

import base64
import json
import logging
import sys
import types

from optchat import OptChatEngine
from optchat.store import Log
from optchat.tools import zoom
from optchat.tree import Tree


class RecordingSummarizer:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return "summary"


def _image(raw: bytes) -> dict:
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def test_image_is_archived_and_kept_out_of_log_view_and_compaction(tmp_path):
    summarizer = RecordingSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    image = b"image bytes for archival"
    encoded = base64.b64encode(image).decode("ascii")
    engine.on_turn_complete([{"role": "user", "content": [
        {"type": "text", "text": "Describe this"}, _image(image)]},
        {"role": "tool", "content": "unrelated text that requires compaction"}])

    result = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert result["media"] == [{"path": "assets/0-1.png", "type": "image"}]
    assert (tmp_path / result["media"][0]["path"]).read_bytes() == image
    assert json.loads((tmp_path / "assets" / "index.json").read_text()) == {
        "0": result["media"]}
    assert result["text"] == "Describe this"
    assert encoded not in engine.log.get(0).text + engine.view.render()
    assert "[image" not in engine.log.get(0).text.lower() + engine.view.render().lower()
    assert summarizer.requests
    assert all(encoded not in request.task and encoded not in request.context
               for request in summarizer.requests)


def test_two_media_parts_keep_content_order_and_media_only_message(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    first, second = b"first", b"second"
    engine.on_turn_complete([{"role": "user", "content": [
        _image(first), {"type": "text", "text": "Two images"}, _image(second)]},
        {"role": "assistant", "content": [_image(b"reply")]}])

    user = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0}))
    assert user["media"] == [
        {"path": "assets/0-0.png", "type": "image"},
        {"path": "assets/0-2.png", "type": "image"},
    ]
    assert [(tmp_path / entry["path"]).read_bytes() for entry in user["media"]] == [first, second]
    reply = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 1}))
    assert reply["text"] == ""
    assert reply["media"] == [{"path": "assets/1-0.png", "type": "image"}]


def test_undecodable_payload_warns_and_does_not_interrupt_ingestion(tmp_path, caplog):
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    with caplog.at_level(logging.WARNING, logger="optchat.media"):
        engine.on_turn_complete([{"role": "user", "content": [
            {"type": "text", "text": "See image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
        ]}])
    assert "Skipped media part 1 on message 0" in caplog.text
    assert json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0}))["media"] == []
    assert engine.log.get(0).text == "See image"


def test_audio_file_and_anthropic_image_payloads(tmp_path):
    local = tmp_path / "source.pdf"
    local.write_bytes(b"%PDF-sample")
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    engine.on_turn_complete([{"role": "user", "content": [
        {"type": "audio", "data": base64.b64encode(b"sound").decode(),
         "mime_type": "audio/wav"},
        {"type": "file", "path": str(local), "mime_type": "application/pdf"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                      "data": base64.b64encode(b"picture").decode()}},
    ]}])
    media = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0}))["media"]
    assert [entry["type"] for entry in media] == ["audio", "file", "image"]
    assert [(tmp_path / entry["path"]).read_bytes() for entry in media] == [
        b"sound", b"%PDF-sample", b"picture"]


def test_input_image_and_input_audio_shapes_do_not_leak_media_text(tmp_path):
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    image = base64.b64encode(b"picture").decode()
    audio = base64.b64encode(b"voice").decode()
    engine.on_turn_complete([{"role": "user", "content": [
        {"type": "input_image", "image_url": f"data:image/png;base64,{image}",
         "text": "media field must not be logged"},
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
    ]}])
    result = json.loads(engine.handle_tool_call("optchat_zoom", {"id": 0}))
    assert result["text"] == ""
    assert result["media"] == [
        {"path": "assets/0-0.png", "type": "image"},
        {"path": "assets/0-1.wav", "type": "audio"},
    ]
    assert [(tmp_path / entry["path"]).read_bytes() for entry in result["media"]] == [
        b"picture", b"voice"]


def _multimodal_host(monkeypatch):
    """Install the host's envelope predicate so the engine sees a supporting host."""
    agent = types.ModuleType("agent")
    agent.__path__ = []
    helpers = types.ModuleType("agent.tool_dispatch_helpers")
    helpers._is_multimodal_tool_result = lambda value: (
        isinstance(value, dict) and value.get("_multimodal") is True
        and isinstance(value.get("content"), list))
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.tool_dispatch_helpers", helpers)


def test_zoom_uses_the_host_envelope_when_supported(tmp_path, monkeypatch):
    _multimodal_host(monkeypatch)
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    image = b"envelope image bytes"
    engine.on_turn_complete([{"role": "user", "content": [
        {"type": "text", "text": "Describe this"}, _image(image)]}])

    result = engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1})

    assert result["_multimodal"] is True
    text_part, image_part = result["content"]
    assert text_part == {"type": "text", "text": result["text_summary"]}
    # The text channel is today's JSON payload, byte for byte.
    assert json.loads(result["text_summary"]) == {
        "id": 0, "n": 1, "kind": "user", "scope": None, "text": "Describe this",
        "media": [{"path": "assets/0-1.png", "type": "image"}]}
    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == image


def test_envelope_is_only_for_messages_with_media(tmp_path, monkeypatch):
    _multimodal_host(monkeypatch)
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    engine.on_turn_complete([{"role": "user", "content": "no media here"}])

    result = engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1})

    assert isinstance(result, str)
    assert json.loads(result)["media"] == []


def test_envelope_keeps_audio_and_file_references_in_the_json(tmp_path, monkeypatch):
    _multimodal_host(monkeypatch)
    local = tmp_path / "source.pdf"
    local.write_bytes(b"%PDF-sample")
    engine = OptChatEngine(tmp_path, summarize=RecordingSummarizer())
    engine.on_turn_complete([{"role": "user", "content": [
        {"type": "audio", "data": base64.b64encode(b"sound").decode(),
         "mime_type": "audio/wav"},
        {"type": "file", "path": str(local), "mime_type": "application/pdf"},
        _image(b"picture")]}])

    result = engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1})

    assert result["_multimodal"] is True
    payload = json.loads(result["text_summary"])
    assert [entry["type"] for entry in payload["media"]] == ["audio", "file", "image"]
    assert len(result["content"]) == 2  # the JSON text part and the one image part
    assert result["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_envelope_passes_a_remote_reference_through(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        index = log.append("user", "remote picture")
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "index.json").write_text(json.dumps({
            "0": [{"path": "https://example.invalid/picture.png", "type": "image"}]}),
            encoding="utf-8")
        result = zoom(log, tree, None, index, 1, envelope=True)

    assert result["_multimodal"] is True
    assert result["content"][1] == {
        "type": "image_url", "image_url": {"url": "https://example.invalid/picture.png"}}


def test_envelope_skips_unresolvable_references(tmp_path):
    with Log(tmp_path) as log:
        tree = Tree(tmp_path)
        index = log.append("user", "lost picture")
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "index.json").write_text(json.dumps({
            "0": [{"path": "assets/0-0.png", "type": "image"}]}), encoding="utf-8")
        result = zoom(log, tree, None, index, 1, envelope=True)

    # Nothing to attach: today's JSON string, references intact.
    assert isinstance(result, str)
    assert json.loads(result)["media"] == [{"path": "assets/0-0.png", "type": "image"}]


def test_envelope_media_still_never_enters_log_view_or_compaction(tmp_path, monkeypatch):
    _multimodal_host(monkeypatch)
    summarizer = RecordingSummarizer()
    engine = OptChatEngine(tmp_path, node_bytes=32, summarize=summarizer)
    image = b"round trip image"
    message = {"role": "user", "content": [
        {"type": "text", "text": "Describe this"}, _image(image)]}
    engine.on_turn_complete([message])

    result = engine.handle_tool_call("optchat_zoom", {"id": 0, "n": 1})
    # The host stores the envelope's content list as the tool message; ingest it.
    engine.on_turn_complete([
        message,
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "function": {"name": "optchat_zoom", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": result["content"]}])

    encoded = base64.b64encode(image).decode("ascii")
    assert engine.log.count() == 3
    assert all(encoded not in record.text
               for record in engine.log.range(0, engine.log.count() - 1))
    assert encoded not in engine.view.render()
    assert all(encoded not in request.task and encoded not in request.context
               for request in summarizer.requests)

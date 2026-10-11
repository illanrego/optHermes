"""Media stays recoverable outside the text-only log, tree, and view."""

import base64
import json
import logging

from optchat import OptChatEngine


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

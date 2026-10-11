"""Archive host media content parts separately from the text-only chat log."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes


logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    "image": "image", "image_url": "image", "input_image": "image",
    "audio": "audio", "input_audio": "audio",
    "file": "file", "input_file": "file",
}

_EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "image/bmp": ".bmp", "image/tiff": ".tiff",
    "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/ogg": ".ogg", "audio/webm": ".webm",
    "application/pdf": ".pdf",
}


def has_media_parts(content: Any) -> bool:
    """Whether a host content list contains a recognized media part."""
    return isinstance(content, list) and any(
        isinstance(part, dict) and isinstance(part.get("type"), str)
        and part["type"] in MEDIA_TYPES for part in content)


def _data_url(value: str) -> tuple[bytes, str]:
    header, separator, body = value.partition(",")
    if not separator or not header.startswith("data:"):
        raise ValueError("invalid data URL")
    mime = header[5:].split(";", 1)[0].lower()
    if header.lower().endswith(";base64"):
        return base64.b64decode(body, validate=True), mime
    return unquote_to_bytes(body), mime


def _payload(part: dict[str, Any], kind: str) -> tuple[bytes, str]:
    source = part.get("source")
    nested = part.get("input_audio") if kind == "audio" else None
    reference = False
    candidate = (part.get("image_url") if kind == "image" else None)
    if candidate is None:
        candidate = part.get("file_data") if kind == "file" else None
    if candidate is None:
        candidate = part.get("data")
    if candidate is None:
        candidate = part.get("url") or part.get("path") or part.get("image")
        reference = candidate is not None
    if candidate is None and isinstance(nested, dict):
        candidate = nested.get("data")
    if candidate is None and isinstance(source, dict):
        candidate = source.get("data") or source.get("url") or source.get("path")
        reference = source.get("data") is None
    if isinstance(candidate, dict):
        reference = candidate.get("data") is None
        candidate = candidate.get("data") or candidate.get("url") or candidate.get("path")
    mime = part.get("mime_type") or part.get("media_type")
    if not mime and isinstance(source, dict):
        mime = source.get("media_type")
    if not isinstance(mime, str):
        mime = ""
    mime = mime.lower().split(";", 1)[0]
    if isinstance(candidate, (bytes, bytearray)):
        return bytes(candidate), mime
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("missing media payload")
    if candidate.startswith("data:"):
        raw, url_mime = _data_url(candidate)
        return raw, url_mime or mime
    if candidate.startswith(("http://", "https://")):
        raise ValueError("remote URL has no local payload")
    if candidate.startswith("file://"):
        candidate = candidate[7:]
        reference = True
    if reference:
        return Path(candidate).expanduser().read_bytes(), mime
    if isinstance(nested, dict) and nested.get("format") == "wav":
        mime = mime or "audio/wav"
    try:
        return base64.b64decode(candidate, validate=True), mime
    except binascii.Error as exc:
        raise ValueError("invalid base64 media payload") from exc


def _index(root: Path) -> dict[str, list[dict[str, str]]]:
    path = root / "assets" / "index.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        raise ValueError("media index is not an object")
    except FileNotFoundError:
        return {}


def archive_media(root: Path, message_id: int, content: Any) -> None:
    """Persist decodable media; failures warn and leave the message ingestible."""
    if not isinstance(content, list):
        return
    assets = Path(root) / "assets"
    entries: list[dict[str, str]] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get("type"), str) or part["type"] not in MEDIA_TYPES:
            continue
        kind = MEDIA_TYPES[part["type"]]
        try:
            raw, mime = _payload(part, kind)
            if not raw:
                raise ValueError("empty media payload")
            extension = _EXTENSIONS.get(mime, ".bin")
            relative = f"assets/{message_id}-{part_index}{extension}"
            assets.mkdir(parents=True, exist_ok=True)
            path = Path(root) / relative
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            entries.append({"path": relative, "type": kind})
        except (OSError, ValueError, TypeError, binascii.Error) as exc:
            logger.warning("Skipped media part %d on message %d: %s", part_index, message_id, exc)
    if not entries:
        return
    try:
        mapping = _index(Path(root))
        mapping[str(message_id)] = entries
        path = assets / "index.json"
        temporary = assets / "index.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(mapping, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not index media for message %d: %s", message_id, exc)


def media_for(root: Path, message_id: int) -> list[dict[str, str]]:
    """Return indexed archive references for one message."""
    try:
        entries = _index(Path(root)).get(str(message_id), [])
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read media index for message %d: %s", message_id, exc)
        return []


# Reverse of the image half of ``_EXTENSIONS``: an archived file's suffix -> its
# media type for the envelope's data URL.
_IMAGE_MIME = {extension: mime for mime, extension in _EXTENSIONS.items()
               if mime.startswith("image/")}


def envelope_media_parts(root: Path, message_id: int) -> list[dict[str, Any]]:
    """Host-envelope content parts for one message's archived images.

    The host carries images in tool results as ``{"type": "image_url",
    "image_url": {"url": ...}}`` parts. A local archive reference is resolved to
    an inline data URL; an http(s) reference is passed through, which the host
    forwards to the provider as a remote-URL image. Anything else - an unknown
    suffix, an unreadable or escaping path - is skipped, and its reference in the
    payload still names it. Only images: the host converts tool-result parts as
    text and image, so audio and file entries stay references.
    """
    root = Path(root)
    parts: list[dict[str, Any]] = []
    for entry in media_for(root, message_id):
        if not isinstance(entry, dict) or entry.get("type") != "image":
            continue
        reference = entry.get("path")
        if not isinstance(reference, str) or not reference:
            continue
        if reference.startswith(("http://", "https://")):
            parts.append({"type": "image_url", "image_url": {"url": reference}})
            continue
        mime = _IMAGE_MIME.get(Path(reference).suffix.lower())
        if mime is None:
            continue
        try:
            path = (root / reference).resolve()
            path.relative_to(root.resolve())
            raw = path.read_bytes()
        except (OSError, ValueError):
            continue
        if not raw:
            continue
        encoded = base64.b64encode(raw).decode("ascii")
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return parts

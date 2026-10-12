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

from .persistence import atomic_json


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


class MediaPayloadError(ValueError):
    """A media part that could not be decoded, carrying its source reference.

    The reference (a path, a data URL, a remote URL or an embedded payload) is
    preserved verbatim so a later pass can retry it; the failure is never a
    reason to drop what the part pointed at.
    """

    def __init__(self, message: str, reference: str | None = None):
        super().__init__(message)
        self.reference = reference


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
        raise MediaPayloadError("missing media payload")
    if candidate.startswith("data:"):
        try:
            raw, url_mime = _data_url(candidate)
        except ValueError as exc:
            raise MediaPayloadError(f"invalid data URL ({exc})", candidate) from exc
        return raw, url_mime or mime
    if candidate.startswith(("http://", "https://")):
        raise MediaPayloadError("remote URL has no local payload", candidate)
    if candidate.startswith("file://"):
        candidate = candidate[7:]
        reference = True
    if reference:
        try:
            return Path(candidate).expanduser().read_bytes(), mime
        except OSError as exc:
            raise MediaPayloadError(f"cannot read {candidate!r} ({exc})", candidate) from exc
    if isinstance(nested, dict) and nested.get("format") == "wav":
        mime = mime or "audio/wav"
    try:
        return base64.b64decode(candidate, validate=True), mime
    except binascii.Error as exc:
        raise MediaPayloadError("invalid base64 media payload", candidate) from exc


def _index(root: Path) -> dict[str, list[dict[str, str]]]:
    path = root / "assets" / "index.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        raise ValueError("media index is not an object")
    except FileNotFoundError:
        return {}


def _write_asset(root: Path, message_id: int, part_index: int, kind: str,
                 raw: bytes, mime: str) -> dict[str, str]:
    """Write one decoded media payload and return its index entry."""
    extension = _EXTENSIONS.get(mime, ".bin")
    relative = f"assets/{message_id}-{part_index}{extension}"
    assets = Path(root) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = Path(root) / relative
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": relative, "type": kind}


def _merge_index(root: Path, message_id: int, entries: list[dict[str, str]]) -> None:
    """Merge archived entries into the media index without dropping earlier ones."""
    mapping = _index(Path(root))
    existing = mapping.get(str(message_id))
    if not isinstance(existing, list):
        existing = []
    known = {item.get("path") for item in existing if isinstance(item, dict)}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("path") not in known:
            existing.append(entry)
            known.add(entry.get("path"))
    mapping[str(message_id)] = existing
    path = Path(root) / "assets" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, mapping)


def archive_media(root: Path, message_id: int, content: Any) -> None:
    """Persist decodable media; a failure records its reference for a retry.

    Decode/read/index failures must not be swallowed while the row checkpoint
    advances: the failed part's reference and its failure are recorded durably
    in ``assets/pending.json`` so a later pass can try again, and the message
    stays ingestible either way.
    """
    if not isinstance(content, list):
        return
    entries: list[dict[str, str]] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get("type"), str) or part["type"] not in MEDIA_TYPES:
            continue
        kind = MEDIA_TYPES[part["type"]]
        try:
            raw, mime = _payload(part, kind)
            if not raw:
                raise MediaPayloadError("empty media payload")
            entries.append(_write_asset(Path(root), message_id, part_index, kind, raw, mime))
        except (OSError, ValueError, TypeError, binascii.Error) as exc:
            logger.warning("Skipped media part %d on message %d: %s (reference recorded "
                           "for retry)", part_index, message_id, exc)
            record_pending_media(Path(root), message_id, part_index, kind,
                                 getattr(exc, "reference", None), exc, part=part)
    if not entries:
        return
    try:
        _merge_index(Path(root), message_id, entries)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not index media for message %d: %s", message_id, exc)
        record_pending_media(Path(root), message_id, None, "index", None, exc, entries=entries)


def media_for(root: Path, message_id: int) -> list[dict[str, str]]:
    """Return indexed archive references for one message."""
    try:
        entries = _index(Path(root)).get(str(message_id), [])
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read media index for message %d: %s", message_id, exc)
        return []


def _pending_path(root: Path) -> Path:
    return Path(root) / "assets" / "pending.json"


def _pending_entries(root: Path) -> list[dict[str, Any]]:
    """The recorded failed media parts, oldest first."""
    try:
        data = json.loads(_pending_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _save_pending(root: Path, entries: list[dict[str, Any]]) -> None:
    path = _pending_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, entries)


def record_pending_media(root: Path, message_id: int, part_index: int | None,
                         kind: str, reference: str | None, error: object,
                         part: dict[str, Any] | None = None,
                         entries: list[dict[str, str]] | None = None) -> None:
    """Record a failed media part durably so a later pass can retry it.

    The reference survives verbatim (a remote URL, a path, a data URL or an
    embedded payload) together with the failure; a retry never guesses. The part
    itself is kept when it is JSON-safe, so the retry can rerun the exact
    extraction. Re-recording the same part replaces its entry and keeps the
    attempt count.
    """
    record: dict[str, Any] = {"message_id": message_id, "part_index": part_index,
                              "type": kind, "reference": reference,
                              "error": str(error), "attempts": 1}
    if isinstance(part, dict):
        try:
            json.dumps(part)
            record["part"] = part
        except (TypeError, ValueError):
            pass
    if isinstance(entries, list):
        record["entries"] = entries
    try:
        pending = _pending_entries(root)
        previous = next((item for item in pending
                         if (item.get("message_id"), item.get("part_index"), item.get("type"))
                         == (message_id, part_index, kind)), None)
        if isinstance(previous, dict) and type(previous.get("attempts")) is int:
            record["attempts"] = max(previous["attempts"], 1)
        pending = [item for item in pending if item is not previous]
        pending.append(record)
        _save_pending(root, pending)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not record pending media for message %s: %s", message_id, exc)


def _retry_part(reference: object) -> dict[str, Any] | None:
    """A minimal part that re-runs the original extraction for a reference."""
    if not isinstance(reference, str) or not reference:
        return None
    if reference.startswith("data:"):
        return {"data": reference}
    if reference.startswith(("http://", "https://")):
        return None  # remote bytes are a separate, open capability
    return {"path": reference}


def retry_pending_media(root: Path) -> dict[str, int]:
    """Try every recorded failed media reference again.

    A local path or an embedded payload is read or decoded again and archived on
    success; a remote URL cannot be fetched yet (fetching remote bytes is a
    separate, open capability), so its reference is kept exactly as recorded.
    Returns ``{"archived": n, "pending": n, "remote": n}``.
    """
    root = Path(root)
    entries = _pending_entries(root)
    if not entries:
        return {"archived": 0, "pending": 0, "remote": 0}
    remaining: list[dict[str, Any]] = []
    archived = remote = 0
    for entry in entries:
        kind = entry.get("type")
        message_id = entry.get("message_id")
        part_index = entry.get("part_index")
        if kind == "index":
            try:
                stored = entry.get("entries")
                if not isinstance(stored, list) or not isinstance(message_id, int):
                    raise ValueError("no retryable index entries")
                _merge_index(root, message_id, stored)
                archived += 1
                continue
            except (OSError, TypeError, ValueError) as exc:
                entry["attempts"] = int(entry.get("attempts", 1)) + 1
                entry["error"] = str(exc)
                remaining.append(entry)
                continue
        reference = entry.get("reference")
        if isinstance(reference, str) and reference.startswith(("http://", "https://")):
            # A retry would fetch the bytes; fetching remote bytes is out of scope
            # today, so the reference is preserved untouched for that pass.
            remote += 1
            remaining.append(entry)
            continue
        part = entry.get("part") if isinstance(entry.get("part"), dict) else _retry_part(reference)
        try:
            if (not isinstance(part, dict) or not isinstance(kind, str) or not kind
                    or not isinstance(message_id, int) or not isinstance(part_index, int)):
                raise ValueError("no retryable reference")
            raw, mime = _payload(part, kind)
            if not raw:
                raise MediaPayloadError("empty media payload")
            _merge_index(root, message_id,
                         [_write_asset(root, message_id, part_index, kind, raw, mime)])
            archived += 1
        except (OSError, ValueError, TypeError, binascii.Error) as exc:
            entry["attempts"] = int(entry.get("attempts", 1)) + 1
            entry["error"] = str(exc)
            remaining.append(entry)
    try:
        _save_pending(root, remaining)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Could not update pending media for %s: %s", root, exc)
    return {"archived": archived, "pending": len(remaining), "remote": remote}


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

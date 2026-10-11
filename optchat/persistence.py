"""Append-only JSONL recovery and durable replacement of derived metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def read_jsonl(path: Path, *, seal: bool = True):
    """Yield (byte offset, JSON value), sealing only an interrupted final line.

    Valid JSON needs just its newline. Invalid tail bytes are retained and sealed
    with NUL + newline: the NUL makes that fragment permanently invalid, even if
    it ended at a JSON boundary. Only these explicitly sealed fragments may be
    ignored on subsequent loads; ordinary mid-file corruption raises.

    ``seal=False`` is the read-only mode: the file is never written. A complete
    record missing only its newline is still yielded, and an unparseable tail
    fragment is skipped - a later writer seals it.
    """
    with path.open("r+b" if seal else "rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                return
            tail = not raw.endswith(b"\n")
            try:
                data = json.loads(raw)
            except (UnicodeError, ValueError) as exc:
                if tail:
                    if seal:
                        stream.write(b"\x00\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    return
                if raw.endswith(b"\x00\n"):
                    continue
                raise ValueError(f"Corrupt JSONL record at {path}:{offset}") from exc
            if tail and seal:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            yield offset, data

"""Append-only, day-partitioned chat log."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .persistence import read_jsonl


KINDS = frozenset({"user", "unii", "tool", "echo", "work", "note"})


@dataclass(frozen=True)
class Record:
    id: int
    kind: str
    text: str
    size: int
    date: str


class LogLockedError(RuntimeError):
    """Another Log instance owns this chat directory."""


class Log:
    def __init__(self, root: Path, *, read_only: bool = False):
        self.root = Path(root)
        self._read_only = bool(read_only)
        if self._read_only:
            # A lock-free read handle for another chat: the chat's owner may hold
            # the exclusive lock in this process or another one. It never writes,
            # so it neither locks, creates directories, nor seals a tail.
            self._lock_fd = None
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self._lock_fd = os.open(self.root / "lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(self._lock_fd)
                self._lock_fd = None
                raise LogLockedError(f"Chat log is locked: {self.root}") from exc

        self._index: list[tuple[Path, int]] = []
        try:
            self._load_index()
        except BaseException:
            self.close()
            raise

    def _load_index(self) -> None:
        main = self.root / "main"
        found: dict[int, tuple[Path, int]] = {}
        for path in sorted(main.glob("????-??-??.jsonl")):
            # read_jsonl seals a torn tail: a complete record missing only its
            # newline is indexed (and the file made whole), while an unparseable
            # fragment is marked so it can never parse later. Mid-file corruption
            # raises, because ids are the identity every other module keys on.
            # A read-only handle skips the repair and only reports what it sees.
            for offset, data in read_jsonl(path, seal=not self._read_only):
                try:
                    record = Record(**data)
                    valid = (
                        isinstance(record.id, int)
                        and record.id >= 0
                        and record.kind in KINDS
                        and isinstance(record.text, str)
                        and isinstance(record.date, str)
                        and record.date[:10] == path.stem
                        and record.size == len(record.text.encode("utf-8"))
                    )
                except (TypeError, KeyError):
                    continue
                if not valid:
                    continue
                if record.id in found:
                    # First-wins would make the later record unreachable and let
                    # the next append reuse its id, losing it for good.
                    raise ValueError(
                        f"Chat log has two records with id {record.id}: "
                        f"{found[record.id][0].name} and {path.name}"
                    )
                found[record.id] = (path, offset)
        if set(found) != set(range(len(found))):
            raise ValueError("Chat log has a gap in its record IDs")
        self._index = [found[i] for i in range(len(found))]

    def append(self, kind: str, text: str) -> int:
        if self._read_only:
            raise ValueError("Log is read-only")
        if kind not in KINDS:
            raise ValueError(f"Invalid record kind: {kind!r}")
        if not isinstance(text, str):
            raise TypeError("Record text must be a string")
        if self._lock_fd is None:
            raise ValueError("Log is closed")

        date = datetime.now(timezone.utc).isoformat()
        record = Record(len(self._index), kind, text, len(text.encode("utf-8")), date)
        path = self.root / "main" / f"{date[:10]}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            offset = stream.tell()
            if offset:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
                    offset += 1
            stream.seek(0, os.SEEK_END)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self._index.append((path, offset))
        return record.id

    def get(self, i: int) -> Record | None:
        if i < 0 or i >= len(self._index):
            return None
        path, offset = self._index[i]
        with path.open("rb") as stream:
            stream.seek(offset)
            return Record(**json.loads(stream.readline()))

    def range(self, lo: int, hi: int) -> list[Record]:
        if hi < lo:
            return []
        return [self.get(i) for i in range(max(lo, 0), min(hi + 1, self.count()))]

    def count(self) -> int:
        return len(self._index)

    def close(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self) -> Log:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_lock_fd", None) is not None:
            self.close()

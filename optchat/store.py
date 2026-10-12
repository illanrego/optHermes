"""Append-only, day-partitioned chat log."""

from __future__ import annotations

import fcntl
import json
import os
import warnings
from contextlib import contextmanager
from threading import RLock
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .persistence import read_jsonl, sync_directory


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
        self._append_lock = RLock()
        self._transaction = False
        self._append_failed = False
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

        self._index: list[tuple[Path, int] | None] = []
        try:
            self._load_index()
        except BaseException:
            self.close()
            raise

    def _load_index(self) -> None:
        main = self.root / "main"
        found: dict[int, tuple[Path, int]] = {}
        refused: dict[int, tuple[Path, int]] = {}
        for path in sorted(main.glob("????-??-??.jsonl")):
            # read_jsonl seals a torn tail: a complete record missing only its
            # newline is indexed (and the file made whole), while an unparseable
            # fragment is marked so it can never parse later. Mid-file corruption
            # raises, because ids are the identity every other module keys on.
            # A read-only handle skips the repair and only reports what it sees.
            for offset, data in read_jsonl(path, seal=not self._read_only):
                try:
                    record = Record(**data)
                except (TypeError, KeyError) as exc:
                    self._refuse(path, offset, data, f"invalid record schema ({exc})", refused)
                    continue
                reason = self._record_reason(record, path)
                if reason is not None:
                    self._refuse(path, offset, data, reason, refused)
                    continue
                if record.id in found:
                    # First-wins would make the later record unreachable and let
                    # the next append reuse its id, losing it for good.
                    raise ValueError(
                        f"Chat log has two records with id {record.id}: "
                        f"{found[record.id][0].name} and {path.name}"
                    )
                found[record.id] = (path, offset)
        seen = set(found) | set(refused)
        if seen != set(range(len(seen))):
            raise ValueError("Chat log has a gap in its record IDs")
        # A refused id keeps its position: the index knows it was seen, ``get``
        # returns ``None`` for it, and ``append`` never reuses it.
        self._index = [found.get(i) for i in range(len(seen))]

    @staticmethod
    def _refuse(path: Path, offset: int, data: object, reason: str,
                refused: dict[int, tuple[Path, int]]) -> None:
        """Warn about a refused line and keep its id known so it is never reused.

        Loud, never a silent skip: a line that parses but does not validate is
        refused. When the line names an id, that id keeps its position in the
        index and the next append can never reuse it.
        """
        warnings.warn(
            f"OptChat chat log line at {path}:{offset} is refused: {reason}",
            RuntimeWarning, stacklevel=2)
        refused_id = data.get("id") if isinstance(data, dict) else None
        if isinstance(refused_id, int) and refused_id >= 0:
            refused.setdefault(refused_id, (path, offset))

    @staticmethod
    def _record_reason(record: Record, path: Path) -> str | None:
        """Why this parsed record does not validate, or ``None`` when it does."""
        if not isinstance(record.id, int) or record.id < 0:
            return "record id must be a nonnegative integer"
        if record.kind not in KINDS:
            return f"unknown record kind {record.kind!r}"
        if not isinstance(record.text, str):
            return "record text must be a string"
        if not isinstance(record.date, str):
            return "record date must be a string"
        if record.date[:10] != path.stem:
            return "record date does not match the file's day"
        if record.size != len(record.text.encode("utf-8")):
            return "record size does not match its text"
        return None

    @contextmanager
    def transaction(self, *, ingestion: bool = False):
        """Exclude other append paths, including reentrant note imports."""
        with self._append_lock:
            if self._transaction:
                raise RuntimeError("Log append transaction cannot be nested")
            if self._append_failed:
                raise RuntimeError("Log append durability unavailable; reopen the log")
            self._transaction = "ingestion" if ingestion else "external"
            try:
                yield
            finally:
                self._transaction = False

    def append(self, kind: str, text: str) -> int:
        with self._append_lock:
            if self._append_failed:
                raise RuntimeError("Log append durability unavailable; reopen the log")
            if self._transaction != "ingestion":
                self.check_ingestion_idle()
            return self._append(kind, text)

    def check_ingestion_idle(self) -> None:
        """An unresolved durable intent owns the tail even outside a hook call."""
        if (self.root / "ingest-unavailable.json").exists():
            raise RuntimeError("Append unavailable: unresolved ingestion recovery")
        checkpoint = self.root / "ingest.json"
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            raise RuntimeError("Append unavailable: unreadable ingestion checkpoint") from exc
        if (not isinstance(state, dict) or state.get("inflight") is not None
                or state.get("lost") is not None or state.get("failure") is not None):
            raise RuntimeError("Append unavailable: unresolved ingestion transaction")

    def _append(self, kind: str, text: str) -> int:
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
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            sync_directory(self.root)
        new_file = not path.exists()
        payload = (json.dumps(record.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
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
            if new_file:
                sync_directory(path.parent)
            self._index.append((path, offset))
        except BaseException:
            self._append_failed = True
            raise
        return record.id

    def get(self, i: int) -> Record | None:
        if i < 0 or i >= len(self._index):
            return None
        entry = self._index[i]
        if entry is None:
            return None  # a refused line's position, never a record
        path, offset = entry
        with path.open("rb") as stream:
            stream.seek(offset)
            return Record(**json.loads(stream.readline()))

    def range(self, lo: int, hi: int) -> list[Record | None]:
        """The records for ids ``lo..hi`` inclusive; ``None`` at a refused id."""
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

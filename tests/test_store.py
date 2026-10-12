import json
import re
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from optchat.store import Log, LogLockedError


def test_day_boundary_ids_lookup_and_utf8_size(tmp_path):
    dates = iter(
        [
            datetime(2026, 10, 10, 23, 59, tzinfo=timezone.utc),
            datetime(2026, 10, 11, 0, 1, tzinfo=timezone.utc),
        ]
    )

    class Clock:
        @staticmethod
        def now(_timezone):
            return next(dates)

    with patch("optchat.store.datetime", Clock):
        with Log(tmp_path) as log:
            assert log.append("user", "hello") == 0
            assert log.append("unii", "café ☕") == 1
            assert log.count() == 2
            assert [record.id for record in log.range(0, 1)] == [0, 1]
            assert log.range(-10, 0)[0].text == "hello"
            assert log.range(2, 1) == []
            assert log.get(1).size == len("café ☕".encode("utf-8"))
            assert log.get(2) is None

    first = tmp_path / "main" / "2026-10-10.jsonl"
    second = tmp_path / "main" / "2026-10-11.jsonl"
    assert json.loads(first.read_text())["id"] == 0
    assert json.loads(second.read_text())["id"] == 1
    with Log(tmp_path) as reopened:
        assert reopened.count() == 2
        assert [record.text for record in reopened.range(0, 1)] == ["hello", "café ☕"]


def test_torn_tail_is_skipped_without_rewriting_it(tmp_path):
    with Log(tmp_path) as log:
        assert log.append("user", "first") == 0
        path = tmp_path / "main" / (log.get(0).date[:10] + ".jsonl")

    torn = b'{"id":1,"kind":"user"'
    with path.open("ab") as stream:
        stream.write(torn)
    original = path.read_bytes()

    with Log(tmp_path) as log:
        assert log.count() == 1
        assert log.append("note", "after") == 1
        assert log.get(1).text == "after"

    assert path.read_bytes().startswith(original)
    with Log(tmp_path) as reopened:
        assert [record.text for record in reopened.range(0, 1)] == ["first", "after"]


def test_exclusive_lock_and_invalid_kind(tmp_path):
    with Log(tmp_path) as first:
        with pytest.raises(LogLockedError):
            Log(tmp_path)
        with pytest.raises(ValueError):
            first.append("thought", "not a record")
        assert first.count() == 0
    with Log(tmp_path) as reopened:
        assert reopened.append("tool", "{}") == 0


def test_reload_uses_ids_when_calendar_moves_backward(tmp_path):
    dates = iter(
        [
            datetime(2026, 10, 11, tzinfo=timezone.utc),
            datetime(2026, 10, 10, tzinfo=timezone.utc),
        ]
    )

    class Clock:
        @staticmethod
        def now(_timezone):
            return next(dates)

    with patch("optchat.store.datetime", Clock):
        with Log(tmp_path) as log:
            log.append("user", "first")
            log.append("unii", "second")
    with Log(tmp_path) as reopened:
        assert [record.text for record in reopened.range(0, 1)] == ["first", "second"]


def test_parseable_torn_tail_is_recovered_and_never_collides(tmp_path):
    with Log(tmp_path) as log:
        assert log.append("user", "first") == 0
        date = log.get(0).date
        path = tmp_path / "main" / (date[:10] + ".jsonl")

    # A complete record whose newline never landed. It parses, so skipping it
    # would let the next append reuse id 1 and make this record unreachable.
    torn = json.dumps({"id": 1, "kind": "note", "text": "torn", "size": 4,
                       "date": date}).encode("utf-8")
    with path.open("ab") as stream:
        stream.write(torn)

    with Log(tmp_path) as log:
        assert log.count() == 2
        assert log.get(1).text == "torn"
        assert log.append("user", "after") == 2
        assert log.get(2).text == "after"

    with Log(tmp_path) as reopened:
        assert [record.text for record in reopened.range(0, 2)] == ["first", "torn", "after"]


def test_duplicate_record_ids_are_rejected(tmp_path):
    with Log(tmp_path) as log:
        log.append("user", "first")
        date = log.get(0).date
        path = tmp_path / "main" / (date[:10] + ".jsonl")

    duplicate = json.dumps({"id": 0, "kind": "user", "text": "copy", "size": 4,
                            "date": date}) + "\n"
    with path.open("ab") as stream:
        stream.write(duplicate.encode("utf-8"))

    with pytest.raises(ValueError, match="two records with id 0"):
        Log(tmp_path)


def test_mid_file_corruption_is_loud(tmp_path):
    with Log(tmp_path) as log:
        log.append("user", "first")
        log.append("note", "second")
        date = log.get(0).date
        path = tmp_path / "main" / (date[:10] + ".jsonl")

    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(1, b'{"broken"\n')
    path.write_bytes(b"".join(lines))

    with pytest.raises(ValueError, match="Corrupt JSONL record"):
        Log(tmp_path)


def test_parseable_invalid_records_are_loud_and_never_reused(tmp_path):
    with Log(tmp_path) as log:
        log.append("user", "first")
        log.append("unii", "second")
        date = log.get(0).date
        path = tmp_path / "main" / (date[:10] + ".jsonl")

    # Valid JSON, invalid schema and size: silently skipping these would leave
    # the ids contiguous and let the next append reuse 2 and 3.
    bad_kind = json.dumps({"id": 2, "kind": "thought", "text": "refused",
                           "size": 7, "date": date})
    bad_size = json.dumps({"id": 3, "kind": "note", "text": "also refused",
                           "size": 999, "date": date})
    with path.open("ab") as stream:
        stream.write((bad_kind + "\n" + bad_size + "\n").encode("utf-8"))

    with pytest.warns(RuntimeWarning) as caught:
        log = Log(tmp_path)
    messages = [str(item.message) for item in caught]
    assert any(re.search(rf"{re.escape(str(path))}:\d+ is refused: unknown record kind",
                         message) for message in messages)
    assert any(re.search(rf"{re.escape(str(path))}:\d+ is refused: record size does not match",
                         message) for message in messages)

    assert log.count() == 4                  # the refused ids keep their positions
    assert log.get(2) is None
    assert log.get(3) is None
    assert log.append("note", "after") == 4  # the refused ids are never reused
    assert log.get(4).text == "after"
    assert [record.text if record else None for record in log.range(0, 4)] == [
        "first", "second", None, None, "after"]
    log.close()

    with Log(tmp_path) as reopened:
        assert reopened.count() == 5
        assert [record.text for record in reopened.range(0, 1)] == ["first", "second"]
        assert reopened.get(2) is None
        assert reopened.get(4).text == "after"


def test_new_day_file_syncs_directory_before_append_returns(tmp_path, monkeypatch):
    import optchat.store as store

    synced = []
    real_sync = store.sync_directory

    def sync(path):
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(store, "sync_directory", sync)
    with Log(tmp_path) as log:
        log.append("user", "first")
        assert synced == [tmp_path, tmp_path / "main"]
        log.append("user", "same day")
        assert synced == [tmp_path, tmp_path / "main"]


def test_day_directory_sync_failure_requires_reopen(tmp_path, monkeypatch):
    import optchat.store as store

    real_sync = store.sync_directory

    def fail_main(path):
        if path == tmp_path / "main":
            raise OSError("injected directory sync failure")
        real_sync(path)

    with Log(tmp_path) as log:
        monkeypatch.setattr(store, "sync_directory", fail_main)
        with pytest.raises(OSError, match="directory sync"):
            log.append("user", "surviving bytes")
        with pytest.raises(RuntimeError, match="durability unavailable"):
            log.append("user", "must not reuse the id")
    monkeypatch.setattr(store, "sync_directory", real_sync)
    with Log(tmp_path) as log:
        assert log.get(0).text == "surviving bytes"
        assert log.append("user", "next") == 1

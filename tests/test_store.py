import json
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

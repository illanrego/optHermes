"""UTF-8 byte-bounded text splitting."""

import pytest

from optchat.text import split_text


def test_split_prefers_the_last_whitespace_in_the_second_half():
    text = "alpha beta gamma"
    assert split_text(text, 11) == ["alpha beta ", "gamma"]


def test_split_hard_breaks_when_whitespace_is_too_early():
    assert split_text("ab cdefghij", 8) == ["ab cdefg", "hij"]


def test_chunks_fit_the_byte_limit_and_preserve_non_ascii_text():
    text = "🙂é漢字 café 🙂é漢字"
    chunks = split_text(text, 7)
    assert "".join(chunks) == text
    assert all(chunk and len(chunk.encode("utf-8")) <= 7 for chunk in chunks)
    assert all(chunk.encode("utf-8").decode("utf-8") == chunk for chunk in chunks)


def test_short_and_empty_text_are_unchanged():
    assert split_text("café", 5) == ["café"]
    assert split_text("", 5) == [""]


def test_a_character_larger_than_the_limit_is_rejected():
    with pytest.raises(ValueError, match="cannot hold a character"):
        split_text("🙂", 3)

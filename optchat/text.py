"""Split logged text into UTF-8 byte-bounded messages."""


def kind_prefix(kind: str) -> str:
    """The ``{kind}: `` tag a seeded node carries (the spec's level-0 <input> form)."""
    return f"{kind}: "


def split_for_kind(text: str, max_bytes: int, kind: str) -> list[str]:
    """Split so every chunk still fits ``max_bytes`` once tagged with ``kind: ``.

    A node's text is ``{kind}: {text}`` and that whole string is what the byte
    budget measures, so the raw text must be split by the remaining budget. Split
    by the full budget instead and no chunk would ever fit, so every message would
    need a model call.
    """
    budget = max_bytes - len(kind_prefix(kind).encode("utf-8"))
    if budget <= 0:
        raise ValueError("max_bytes must exceed the kind tag")
    return split_text(text, budget)


def split_text(text: str, max_bytes: int) -> list[str]:
    """Preserve all text while preferring a late whitespace boundary."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start
        size = 0
        whitespace_end = None
        while end < len(text):
            char_size = len(text[end].encode("utf-8"))
            if size + char_size > max_bytes:
                break
            size += char_size
            end += 1
            if text[end - 1].isspace() and size * 2 >= max_bytes:
                whitespace_end = end
        if end == start:
            raise ValueError("max_bytes cannot hold a character in text")
        if end == len(text):
            chunks.append(text[start:end])
            break
        cut = whitespace_end if whitespace_end is not None else end
        chunks.append(text[start:cut])
        start = cut
    return chunks

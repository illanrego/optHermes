"""Split logged text into UTF-8 byte-bounded messages."""


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

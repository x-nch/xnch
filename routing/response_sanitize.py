"""Strip model reasoning / thinking blocks from chat output."""

from __future__ import annotations

import re

_OPEN_TAGS = ("<think>", "<thinking>", "<think>")
_CLOSE_TAGS = ("</think>", "</thinking>", "</think>")
_BLOCK_RE = re.compile(
    r"<(?:redacted_thinking|think(?:ing)?)>.*?</(?:redacted_thinking|think(?:ing)?)>",
    re.DOTALL | re.IGNORECASE,
)


def _find_tag(text: str, tags: tuple[str, ...]) -> tuple[int, int] | None:
    lower = text.lower()
    best: tuple[int, int] | None = None
    for tag in tags:
        idx = lower.find(tag.lower())
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, len(tag))
    return best


def _partial_suffix_len(text: str, tags: tuple[str, ...]) -> int:
    lower = text.lower()
    max_keep = 0
    for tag in tags:
        tag_lower = tag.lower()
        for i in range(1, len(tag_lower)):
            if lower.endswith(tag_lower[:i]):
                max_keep = max(max_keep, i)
    return max_keep


def strip_thinking(text: str) -> str:
    """Remove thinking blocks and orphan pre-close-tag reasoning from model output."""
    if not text:
        return ""

    cleaned = _BLOCK_RE.sub("", text)

    while True:
        close = _find_tag(cleaned, _CLOSE_TAGS)
        if close is None:
            break
        cleaned = cleaned[close[0] + close[1] :]

    open_tag = _find_tag(cleaned, _OPEN_TAGS)
    if open_tag is not None:
        cleaned = cleaned[: open_tag[0]]

    return cleaned.strip()


class ThinkingStripFilter:
    """Incremental filter for streaming model output.

  Buffers until a thinking close tag (or complete block) is seen, then streams
  only the post-thinking content. If the stream ends without thinking markers,
  emits the full buffered text.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False
        self._tail = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        if self._started:
            return self._feed_started(chunk)

        self._buffer += chunk

        close = _find_tag(self._buffer, _CLOSE_TAGS)
        if close is not None:
            remainder = self._buffer[close[0] + close[1] :]
            self._buffer = ""
            self._started = True
            return self._feed_started(remainder)

        if _BLOCK_RE.search(self._buffer):
            cleaned = strip_thinking(self._buffer)
            self._buffer = ""
            self._started = True
            return cleaned

        return ""

    def _feed_started(self, chunk: str) -> str:
        self._tail += chunk
        keep = _partial_suffix_len(self._tail, _OPEN_TAGS + _CLOSE_TAGS)
        if keep < len(self._tail):
            out = self._tail[:-keep] if keep else self._tail
            self._tail = self._tail[-keep:] if keep else ""
            return out
        return ""

    def flush(self) -> str:
        if self._started:
            out = self._tail
            self._tail = ""
            return out

        cleaned = strip_thinking(self._buffer)
        self._buffer = ""
        return cleaned

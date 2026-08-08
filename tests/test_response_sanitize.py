"""Tests for model response thinking-block sanitization."""

from xnch.routing.response_sanitize import ThinkingStripFilter, strip_thinking


def test_strip_redacted_thinking_block():
    raw = (
        "<think>Let me think about this greeting.</think>\n\n"
        "ck-san. I'm here."
    )
    assert strip_thinking(raw) == "ck-san. I'm here."


def test_strip_orphan_close_tag():
    raw = (
        'The user said "Hello". I should respond concisely.\n'
        "</think>\n\n"
        "ck-san. You've already said hello."
    )
    assert strip_thinking(raw) == "ck-san. You've already said hello."


def test_strip_thinking_tag_variant():
    raw = "<thinking>internal reasoning</thinking>\n\nHello!"
    assert strip_thinking(raw) == "Hello!"


def test_strip_empty_and_plain_text():
    assert strip_thinking("") == ""
    assert strip_thinking("Hello there.") == "Hello there."


def test_stream_filter_skips_thinking_block():
    filt = ThinkingStripFilter()
    assert filt.feed("<think>reasoning") == ""
    assert filt.feed("</think>\n\nHi") == "\n\nHi"
    assert filt.flush() == ""


def test_stream_filter_orphan_close():
    filt = ThinkingStripFilter()
    assert filt.feed("Some reasoning\n") == ""
    assert filt.feed("</think>\n\nAnswer") == "\n\nAnswer"

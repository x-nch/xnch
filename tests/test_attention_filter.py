"""Unit tests for AttentionFilter (no hardware required)."""
import time

import pytest

from xnch.perception.attention_filter import AttentionFilter


@pytest.fixture
def filter_() -> AttentionFilter:
    return AttentionFilter(
        silence_threshold_s=1.5,
        screen_diff_threshold=0.15,
    )


class TestVoiceTranscriptRule:
    def test_forwards_to_gateway_when_transcript_and_silence(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript="hello world",
            silence_duration_s=2.0,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert any(a["rule"] == "voice_transcript" and a["action"] == "forward_to_gateway" for a in actions)

    def test_skips_when_no_transcript(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=2.0,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert not any(a["rule"] == "voice_transcript" for a in actions)

    def test_skips_when_silence_too_short(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript="hello",
            silence_duration_s=0.5,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert not any(a["rule"] == "voice_transcript" for a in actions)

    def test_skips_when_empty_transcript(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript="",
            silence_duration_s=2.0,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert not any(a["rule"] == "voice_transcript" for a in actions)


class TestScreenChangeRule:
    def test_encodes_episode_when_diff_exceeds_threshold(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.3,
            file_saved=False,
        )
        assert any(a["rule"] == "screen_change" and a["action"] == "encode_and_store_episode" for a in actions)

    def test_skips_when_diff_below_threshold(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.05,
            file_saved=False,
        )
        assert not any(a["rule"] == "screen_change" for a in actions)

    def test_skips_when_diff_equals_threshold(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.15,
            file_saved=False,
        )
        assert not any(a["rule"] == "screen_change" for a in actions)


class TestFileSavedRule:
    def test_triggers_file_watcher_when_file_saved(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.0,
            file_saved=True,
        )
        assert any(a["rule"] == "vault_file_saved" and a["action"] == "trigger_file_watcher" for a in actions)

    def test_skips_when_no_file_saved(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert not any(a["rule"] == "vault_file_saved" for a in actions)


class TestMultipleRules:
    def test_multiple_rules_can_trigger_simultaneously(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript="test",
            silence_duration_s=2.0,
            screen_pixel_diff=0.5,
            file_saved=True,
        )
        rules = {a["rule"] for a in actions}
        assert "voice_transcript" in rules
        assert "screen_change" in rules
        assert "vault_file_saved" in rules

    def test_returns_empty_when_no_rules_match(self, filter_: AttentionFilter):
        actions = filter_.evaluate(
            voice_transcript=None,
            silence_duration_s=0.0,
            screen_pixel_diff=0.0,
            file_saved=False,
        )
        assert actions == []


class TestTouch:
    def test_touch_updates_last_activity(self, filter_: AttentionFilter):
        before = filter_._last_activity
        time.sleep(0.01)
        filter_.touch()
        assert filter_._last_activity > before

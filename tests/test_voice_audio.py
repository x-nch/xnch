"""Unit tests for xnch/voice/audio.py."""

from __future__ import annotations

import pytest

from tests.fixtures.voice.make_tone import make_tone_wav
from xnch.voice.audio import AudioValidationError, decode_audio_input, pcm_to_wav


def test_decode_wav_tone():
    wav = make_tone_wav()
    decoded = decode_audio_input(wav, fmt="wav")
    assert decoded.sample_rate == 16000
    assert decoded.duration_s == pytest.approx(0.5, abs=0.01)
    assert len(decoded.pcm_s16le) > 0


def test_pcm_roundtrip():
    wav = make_tone_wav()
    decoded = decode_audio_input(wav, fmt="wav")
    out = pcm_to_wav(decoded.pcm_s16le, sample_rate=decoded.sample_rate)
    again = decode_audio_input(out, fmt="wav")
    assert again.duration_s == pytest.approx(decoded.duration_s, abs=0.01)


def test_reject_empty():
    with pytest.raises(AudioValidationError):
        decode_audio_input(b"", fmt="wav")


def test_reject_too_short():
    pcm = b"\x00\x00" * 2
    with pytest.raises(AudioValidationError):
        decode_audio_input(pcm, fmt="pcm", sample_rate=16000)

"""Audio validation and WAV encoding for voice endpoints."""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass

from xnch.config import settings


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedAudio:
    pcm_s16le: bytes
    sample_rate: int
    duration_s: float


def pcm_to_wav(pcm: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise AudioValidationError("WAV must be 16-bit PCM")
        channels = wf.getnchannels()
        if channels != 1:
            raise AudioValidationError("WAV must be mono")
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


def decode_audio_input(
    raw: bytes,
    *,
    fmt: str = "wav",
    sample_rate: int = 16000,
) -> DecodedAudio:
    if not raw:
        raise AudioValidationError("Empty audio payload")
    if len(raw) > settings.voice_max_audio_bytes:
        raise AudioValidationError(
            f"Audio exceeds {settings.voice_max_audio_bytes} bytes"
        )

    fmt_norm = (fmt or "wav").lower()
    if fmt_norm == "wav":
        pcm, sr = _wav_to_pcm(raw)
    elif fmt_norm in {"pcm", "pcm_s16le"}:
        pcm = raw
        sr = sample_rate
        if sr <= 0:
            raise AudioValidationError("sample_rate required for PCM input")
        if len(pcm) % 2 != 0:
            raise AudioValidationError("PCM length must be even (16-bit samples)")
    else:
        raise AudioValidationError(f"Unsupported audio format: {fmt}")

    duration_s = len(pcm) / (2 * sr)
    if duration_s > settings.voice_max_audio_duration_s:
        raise AudioValidationError(
            f"Audio exceeds {settings.voice_max_audio_duration_s}s limit"
        )
    if duration_s < 0.05:
        raise AudioValidationError("Audio too short")

    return DecodedAudio(pcm_s16le=pcm, sample_rate=sr, duration_s=duration_s)


def wav_duration_s(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())

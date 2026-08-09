"""Speech-to-text via faster-whisper (lazy-loaded, CPU by default)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from xnch.config import settings
from xnch.voice.audio import pcm_to_wav

logger = logging.getLogger(__name__)

_whisper_model = None
_model_lock = asyncio.Lock()


class SttError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str
    duration_s: float


async def _get_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _model_lock:
        if _whisper_model is not None:
            return _whisper_model

        def _load():
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SttError(
                    "faster-whisper not installed; pip install faster-whisper"
                ) from exc
            return WhisperModel(
                settings.voice_stt_model,
                device=settings.voice_stt_device,
                compute_type=settings.voice_stt_compute_type,
            )

        loop = asyncio.get_running_loop()
        _whisper_model = await loop.run_in_executor(None, _load)
        logger.info(
            "Loaded STT model %s on %s",
            settings.voice_stt_model,
            settings.voice_stt_device,
        )
        return _whisper_model


async def transcribe_wav(
    wav_bytes: bytes,
    *,
    duration_s: float | None = None,
) -> TranscriptResult:
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SttError("WAV must be 16-bit PCM")
        pcm = wf.readframes(wf.getnframes())

    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    model = await _get_model()
    loop = asyncio.get_running_loop()

    def _run() -> tuple[str, str]:
        segments, info = model.transcribe(
            audio_np,
            beam_size=1,
            language=settings.voice_stt_language or None,
        )
        text = " ".join(seg.text for seg in segments).strip()
        lang = info.language or settings.voice_stt_language or "en"
        return text, lang

    text, language = await loop.run_in_executor(None, _run)
    if not text:
        raise SttError("Empty transcript (silence or unintelligible audio)")

    return TranscriptResult(
        text=text,
        language=language,
        duration_s=duration_s or 0.0,
    )


async def transcribe_pcm(
    pcm: bytes,
    *,
    sample_rate: int,
    duration_s: float,
) -> TranscriptResult:
    wav = pcm_to_wav(pcm, sample_rate=sample_rate)
    return await transcribe_wav(wav, duration_s=duration_s)

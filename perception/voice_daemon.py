import asyncio
import json
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from ..config import settings


class VoiceDaemon:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = aioredis.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
        )
        self._vad_model = None
        self._whisper_model = None
        self._recording = False
        self._audio_buffer: list[bytes] = []

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def load_models(self) -> None:
        loop = asyncio.get_running_loop()

        def _load_vad():
            import silero_vad
            return silero_vad.load_silero_vad()

        def _load_whisper():
            from faster_whisper import WhisperModel
            return WhisperModel(
                "base",
                device="cuda",
                compute_type="float16",
                num_workers=1,
            )

        self._vad_model = await loop.run_in_executor(None, _load_vad)
        self._whisper_model = await loop.run_in_executor(None, _load_whisper)

    async def start_recording(self) -> dict[str, Any]:
        if self._recording:
            return {"status": "already_recording"}
        self._recording = True
        self._audio_buffer = []
        return {"status": "recording_started", "session_id": str(uuid.uuid4())}

    async def stop_recording(self) -> dict[str, Any]:
        if not self._recording:
            return {"status": "not_recording"}
        self._recording = False
        transcript = await self._transcribe()
        perception_id = str(uuid.uuid4())
        payload = {
            "id": perception_id,
            "transcript": transcript,
            "timestamp": time.time(),
            "type": "voice",
        }
        await self._redis.setex(
            f"perception:voice:{perception_id}",
            60,
            json.dumps(payload),
        )
        self._audio_buffer = []
        return {
            "status": "stopped",
            "perception_id": perception_id,
            "transcript": transcript,
        }

    async def push_audio(self, chunk: bytes) -> None:
        if not self._recording:
            return
        self._audio_buffer.append(chunk)

    async def _transcribe(self) -> str:
        if self._whisper_model is None or not self._audio_buffer:
            return ""
        import numpy as np
        raw = b"".join(self._audio_buffer)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_running_loop()

        def _run():
            segments, _ = self._whisper_model.transcribe(audio_np, beam_size=1, language="en")
            return " ".join(seg.text for seg in segments).strip()

        return await loop.run_in_executor(None, _run)

    async def detect_silence(self, chunk: bytes, sample_rate: int = 16000) -> bool:
        if self._vad_model is None:
            return True
        import numpy as np
        audio_int16 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_running_loop()

        def _check():
            return self._vad_model.is_speech(audio_int16, sample_rate)

        return not await loop.run_in_executor(None, _check)

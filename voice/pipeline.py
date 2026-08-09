"""Voice chat orchestration: STT → Nexi chat → TTS."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from nexi.pipeline.context_assembler import assemble_context
from nexi.proactivity.engine import ProactivityEngine
from xnch.config import settings
from xnch.routing.classifier import classify_request
from xnch.security.injection_guard import scan_input
from xnch.security.memory_guard import validate_memory_write
from xnch.security.trust_model import get_trust_level
from xnch.voice.audio import DecodedAudio, pcm_to_wav
from xnch.voice.stt import SttError, transcribe_pcm
from xnch.voice.tts import TtsError, synthesize_speech

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceChatResult:
    transcript: str
    response: str
    session_id: str
    model_used: str
    language: str
    duration_s: float
    audio_wav: bytes | None = None

    def to_dict(self, *, include_audio: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transcript": self.transcript,
            "response": self.response,
            "session_id": self.session_id,
            "model_used": self.model_used,
            "language": self.language,
            "duration_s": self.duration_s,
        }
        if include_audio and self.audio_wav:
            payload["audio_base64"] = base64.b64encode(self.audio_wav).decode("ascii")
            payload["audio_format"] = "wav"
        return payload


def _get_proactivity(app: Any) -> ProactivityEngine:
    if not hasattr(app, "_nexi_proactivity"):
        redis = app.kv_cache.redis_client
        app._nexi_proactivity = ProactivityEngine(redis)
    return app._nexi_proactivity


async def _agent_lessons_for_chat(app: Any, message: str) -> list[str]:
    if not settings.am_prefetch_enabled:
        return []
    from xnch.memory.agentmemory_prefetch import prefetch_agent_lessons
    from xnch.routing.recall_intent import recall_query

    query = recall_query(message) or message
    return await prefetch_agent_lessons(app, query)


def _invalidate_system_prompt_cache(app: Any) -> None:
    from xnch.routes.nexi_gateway import _invalidate_system_prompt_cache as _invalidate

    _invalidate(app)


async def run_nexi_chat(
    app: Any,
    *,
    message: str,
    session_id: str,
    actor_role: str,
    voice_mode: bool = False,
) -> tuple[str, str]:
    """Run the standard Nexi chat path; returns (response_text, model_name)."""
    result = scan_input(message, app.event_log)
    if not result.is_clean:
        raise HTTPException(status_code=400, detail="Input rejected by injection guard")

    from xnch.routing.recall_intent import recall_query

    ctx = await assemble_context(
        session_id=session_id,
        raw_input=message,
        working_memory=app.working_memory,
        pg_episodic=app.pg_episodic,
        graph_store=app.graph_store,
        relationship_store=app.relationship_store,
        sensory_buffer=app.sensory_buffer,
        proactivity_engine=_get_proactivity(app),
        recall_query=recall_query(message),
        agent_lessons=await _agent_lessons_for_chat(app, message),
        voice_mode=voice_mode,
    )

    route = classify_request(message, actor_role, {})
    messages = ctx.to_messages(message)
    model_name = route.model_name

    await app.working_memory.append_turn(session_id, "user", message)

    from xnch_mcp.chat_tools import chat_with_tools

    try:
        response_text = await chat_with_tools(
            app,
            messages,
            model_name,
            session_id=session_id,
            actor_role="nexi",
        )
    except Exception as exc:
        logger.error("LiteLLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LiteLLM unavailable") from exc

    await app.working_memory.append_turn(session_id, "assistant", response_text)

    episode_text = f"{message}\n{response_text}"
    validation = validate_memory_write(
        content=episode_text,
        actor_role=actor_role,
        trust_level=get_trust_level(actor_role),
    )
    if not validation[0]:
        logger.warning("Memory write blocked by guard: %s", validation[1])
    elif await app.pg_episodic.has_identical_recent(episode_text, hours=24):
        logger.info("Skipping duplicate episode store for session %s", session_id)
    else:
        await app.pg_episodic.store_episode(
            type_="conversation",
            raw_text=episode_text,
            summary=f"{message[:80]} → {response_text[:120]}",
        )

    _invalidate_system_prompt_cache(app)
    return response_text, model_name


async def run_voice_chat(
    app: Any,
    audio: DecodedAudio,
    *,
    session_id: str,
    actor_role: str = "operator",
    return_audio: bool = True,
) -> VoiceChatResult:
    if not settings.voice_enabled:
        raise HTTPException(status_code=503, detail="Voice subsystem disabled")

    try:
        stt = await transcribe_pcm(
            audio.pcm_s16le,
            sample_rate=audio.sample_rate,
            duration_s=audio.duration_s,
        )
    except SttError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transcript = stt.text
    await app.sensory_buffer.write_perception(
        "voice",
        transcript,
        ttl=60,
    )

    response_text, model_name = await run_nexi_chat(
        app,
        message=transcript,
        session_id=session_id,
        actor_role=actor_role,
        voice_mode=True,
    )

    audio_wav: bytes | None = None
    if return_audio:
        try:
            audio_wav = await synthesize_speech(response_text)
        except TtsError as exc:
            logger.error("TTS failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"TTS unavailable: {exc}") from exc

    return VoiceChatResult(
        transcript=transcript,
        response=response_text,
        session_id=session_id,
        model_used=model_name,
        language=stt.language,
        duration_s=audio.duration_s,
        audio_wav=audio_wav,
    )


async def transcribe_audio(audio: DecodedAudio) -> dict[str, Any]:
    if not settings.voice_enabled:
        raise HTTPException(status_code=503, detail="Voice subsystem disabled")
    try:
        stt = await transcribe_pcm(
            audio.pcm_s16le,
            sample_rate=audio.sample_rate,
            duration_s=audio.duration_s,
        )
    except SttError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "transcript": stt.text,
        "duration_s": audio.duration_s,
        "language": stt.language,
    }


async def speak_text(text: str) -> bytes:
    if not settings.voice_enabled:
        raise HTTPException(status_code=503, detail="Voice subsystem disabled")
    try:
        return await synthesize_speech(text)
    except TtsError as exc:
        raise HTTPException(status_code=502, detail=f"TTS unavailable: {exc}") from exc


def decoded_to_wav(audio: DecodedAudio) -> bytes:
    return pcm_to_wav(audio.pcm_s16le, sample_rate=audio.sample_rate)

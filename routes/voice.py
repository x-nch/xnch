"""Nexi voice endpoints — STT, TTS, and full voice chat loop."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from xnch.config import settings
from xnch.voice.audio import AudioValidationError, decode_audio_input
from xnch.voice.pipeline import (
    run_voice_chat,
    speak_text,
    transcribe_audio,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/nexi/voice", tags=["nexi-voice"])


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None


async def _read_upload(audio: UploadFile) -> bytes:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    return data


@router.post("/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    format: str = Form("wav"),
    sample_rate: int = Form(16000),
) -> dict:
    if not settings.voice_enabled:
        raise HTTPException(status_code=503, detail="Voice subsystem disabled")
    try:
        raw = await _read_upload(audio)
        decoded = decode_audio_input(raw, fmt=format, sample_rate=sample_rate)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await transcribe_audio(decoded)


@router.post("/speak")
async def voice_speak(body: SpeakRequest) -> Response:
    wav = await speak_text(body.text)
    return Response(content=wav, media_type="audio/wav")


@router.post("/speak/upload")
async def voice_speak_raw(
    text: str = Form(...),
) -> Response:
    wav = await speak_text(text)
    return Response(content=wav, media_type="audio/wav")


@router.post("/chat")
async def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    actor_role: str = Form("operator"),
    return_audio: bool = Form(True),
) -> dict:
    try:
        raw = await _read_upload(audio)
        decoded = decode_audio_input(raw, fmt="wav")
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await run_voice_chat(
        request.app.state,
        decoded,
        session_id=session_id,
        actor_role=actor_role,
        return_audio=return_audio,
    )
    return result.to_dict(include_audio=return_audio)

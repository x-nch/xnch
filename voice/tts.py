"""Text-to-speech via Piper CLI (subprocess) or espeak-ng fallback."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from xnch.config import settings

logger = logging.getLogger(__name__)


class TtsError(RuntimeError):
    pass


def _truncate_for_tts(text: str) -> str:
    limit = settings.voice_max_tts_chars
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _synthesize_piper(text: str) -> bytes:
    piper_bin = shutil.which("piper")
    if piper_bin is None:
        raise TtsError("piper binary not found in PATH")

    model = settings.voice_tts_voice_path.expanduser()
    config = settings.voice_tts_config_path.expanduser()
    if not model.is_file():
        raise TtsError(f"Piper voice model not found: {model}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = Path(out_f.name)

    try:
        cmd = [
            piper_bin,
            "--model",
            str(model),
            "--output_file",
            str(out_path),
        ]
        if config.is_file():
            cmd.extend(["--config", str(config)])

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise TtsError(f"piper failed: {stderr.strip() or proc.returncode}")

        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def _synthesize_espeak(text: str) -> bytes:
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak is None:
        raise TtsError("No TTS engine available (install piper or espeak-ng)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = Path(out_f.name)

    try:
        proc = subprocess.run(
            [espeak, "-w", str(out_path), text],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise TtsError(f"espeak failed: {stderr.strip() or proc.returncode}")
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def _synthesize_sync(text: str) -> bytes:
    trimmed = _truncate_for_tts(text)
    if not trimmed:
        raise TtsError("Empty text for TTS")

    engine = settings.voice_tts_engine.lower()
    if engine == "piper":
        try:
            return _synthesize_piper(trimmed)
        except TtsError as exc:
            logger.warning("Piper TTS failed (%s); trying espeak fallback", exc)
            return _synthesize_espeak(trimmed)
    if engine == "espeak":
        return _synthesize_espeak(trimmed)
    raise TtsError(f"Unknown TTS engine: {engine}")


async def synthesize_speech(text: str) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _synthesize_sync, text)

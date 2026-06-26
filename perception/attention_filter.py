import asyncio
import logging
import time
from typing import Any

from ..config import settings


logger = logging.getLogger(__name__)


class AttentionFilter:
    def __init__(
        self,
        silence_threshold_s: float | None = None,
        screen_diff_threshold: float | None = None,
        idle_timeout_s: int | None = None,
    ) -> None:
        self._silence_threshold = silence_threshold_s or settings.attention_silence_threshold_s
        self._screen_diff_threshold = screen_diff_threshold or settings.attention_screen_diff_threshold
        self._idle_timeout = idle_timeout_s or settings.attention_idle_timeout_s
        self._last_activity: float = time.time()

    def touch(self) -> None:
        self._last_activity = time.time()

    def evaluate(
        self,
        voice_transcript: str | None,
        silence_duration_s: float,
        screen_pixel_diff: float,
        file_saved: bool,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        now = time.time()

        # Rule 1: Voice transcript present + silence > threshold
        if voice_transcript and silence_duration_s > self._silence_threshold:
            self.touch()
            actions.append({
                "rule": "voice_transcript",
                "action": "forward_to_gateway",
                "payload": {"transcript": voice_transcript},
            })

        # Rule 2: Screen pixel diff > threshold
        if screen_pixel_diff > self._screen_diff_threshold:
            self.touch()
            actions.append({
                "rule": "screen_change",
                "action": "encode_and_store_episode",
                "payload": {"pixel_diff": screen_pixel_diff, "layer": 2},
            })

        # Rule 3: File saved in vault
        if file_saved:
            self.touch()
            actions.append({
                "rule": "vault_file_saved",
                "action": "trigger_file_watcher",
                "payload": {},
            })

        # Rule 4: User idle > timeout
        idle_duration = now - self._last_activity
        if idle_duration > self._idle_timeout:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                from ..jobs.consolidation import run_consolidation
                task = asyncio.create_task(run_consolidation())
                task.add_done_callback(lambda t: t.exception() and logger.error("consolidation failed: %s", t.exception()))
            actions.append({
                "rule": "user_idle",
                "action": "suppress_responses_and_consolidate",
                "payload": {"idle_duration_s": idle_duration},
            })

        return actions

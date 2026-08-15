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
    ) -> None:
        self._silence_threshold = silence_threshold_s or settings.attention_silence_threshold_s
        self._screen_diff_threshold = screen_diff_threshold or settings.attention_screen_diff_threshold
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

        # Consolidation no longer runs in-process on idle: it used to spawn
        # CPU-bound llama.cpp inference on the server event loop, freezing the
        # API. It now runs only via the systemd/k3s timer.

        return actions

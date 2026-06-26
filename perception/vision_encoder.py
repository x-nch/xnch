import asyncio
import json
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from ..config import settings


class VisionEncoder:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = aioredis.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
        )
        self._model = None
        self._model_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def load_model(self) -> None:
        async with self._model_lock:
            if self._model is not None:
                return

            def _load():
                import moondream
                return moondream.vl(model_id="moondream2")

            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(None, _load)

    async def encode_screenshot(self, image_bytes: bytes) -> dict[str, Any]:
        await self.load_model()
        loop = asyncio.get_running_loop()

        def _encode():
            from PIL import Image as PILImage
            import io
            image = PILImage.open(io.BytesIO(image_bytes))
            return self._model.caption(image)["caption"]

        description = await loop.run_in_executor(None, _encode)
        perception_id = str(uuid.uuid4())
        payload = {
            "id": perception_id,
            "description": description,
            "timestamp": time.time(),
            "type": "vision",
        }
        await self._redis.setex(
            f"perception:vision:{perception_id}",
            60,
            json.dumps(payload),
        )
        return payload

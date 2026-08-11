"""beeAI ChatModel backend — routed through the same LiteLLM proxy as the rest of xnch.

Production path: beeAI OpenAI-compatible ChatModel -> LiteLLM :4000 -> vLLM (Ornith).
Keeps a single inference gateway and lets LiteLLM handle model routing/auth.
"""
from __future__ import annotations

from typing import Any

from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelOutput
from beeai_framework.backend.chat import load_model

from ...config import settings


def build_chat_model(
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatModel:
    """Build an OpenAI-compatible ChatModel pointed at the LiteLLM proxy.

    Overridable per-call so tests and the swarm demo can inject a stub.
    """
    OpenAIChatModel = load_model("openai")
    return OpenAIChatModel(
        model_id=model_id or settings.beeai_model,
        base_url=base_url or settings.litellm_proxy_url,
        api_key=api_key or settings.beeai_api_key,
    )


class StaticChatModel(ChatModel):
    """Deterministic ChatModel stub — returns a fixed answer without a backend.

    Used by tests and by the degraded/demo path (``XNCH_BEEAI_DEMO_MODE``).
    """

    provider_id = "static"
    model_id = "static/fixed"

    def __init__(self, response: str = "beeAI demo response (no LLM configured)") -> None:
        super().__init__()
        self._response = response

    async def _create(self, input: Any, run: Any) -> ChatModelOutput:
        return ChatModelOutput(
            output=[AssistantMessage(content=self._response)],
            finish_reason="end_turn",
        )

    async def _create_stream(self, input: Any, run: Any) -> Any:
        yield await self._create(input, run)

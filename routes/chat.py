"""OpenAI-compatible /v1/chat/completions endpoint — forwards to Nexi."""
import logging
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> dict[str, Any]:
    """OpenAI-compatible chat completions endpoint."""
    app = request.app.state

    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    raw_input = body.messages[-1].content

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    actor_id = app.token_verifier.verify_bearer(auth_header)
    if not actor_id:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    actor = await app.governance.resolve_actor(actor_id)
    if not actor:
        raise HTTPException(status_code=401, detail=f"Unknown actor: {actor_id}")

    state_version = await app.get_state_version()
    policy_version = await app.get_policy_version()

    session_id = str(uuid4())
    trace_id = str(uuid4())
    session_context = {
        "session_id": session_id,
        "trace_id": trace_id,
        "actor": actor.to_dict(),
        "system_state_version": state_version,
        "policy_version": policy_version,
        "idempotency_key": str(uuid4()),
        "raw_input": raw_input,
        "priority": "NORMAL",
    }

    try:
        async with httpx.AsyncClient(base_url=settings.nexi_base_url, timeout=120.0) as client:
            resp = await client.post("/session/start", json=session_context)
        nexi_response = resp.json()
    except Exception as exc:
        logger.error("Nexi /session/start failed: %s", exc)
        raise HTTPException(status_code=502, detail="Nexi unavailable")

    await app.working_memory.append_turn(session_id, "user", raw_input)
    response_text = nexi_response.get("status", "") or nexi_response.get("error", "")
    await app.working_memory.append_turn(session_id, "assistant", response_text)

    return {
        "id": f"chatcmpl-{session_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
    }

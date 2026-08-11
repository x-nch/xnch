"""beeAI orchestration path tests — demo (StaticChatModel) runs and route gating."""
import pytest
from types import SimpleNamespace

from xnch_mcp.context import ActorContext
from xnch.agents.beeai.backend import StaticChatModel, build_chat_model
from xnch.agents.beeai.runtime import run_agent, run_swarm


def _actor() -> ActorContext:
    return ActorContext(actor_role="operator", trace_id="test-beeai", session_id="s-1")


def _app_state() -> SimpleNamespace:
    events: list[dict] = []

    def emit(*args, **kwargs) -> None:
        events.append({"args": args, "kwargs": kwargs})

    return SimpleNamespace(registry=None, event_log=SimpleNamespace(emit=emit))


@pytest.mark.asyncio
async def test_run_agent_demo_mode_returns_normalized_result() -> None:
    """Demo-mode run_agent should return text, tool_count and duration_ms."""
    actor = _actor()
    app_state = _app_state()
    result = await run_agent(
        "what tools do you have?",
        app_state=app_state,
        actor=actor,
        event_log=app_state.event_log,
        approve=True,
        llm=StaticChatModel(),
    )
    assert result["text"].startswith("beeAI demo response")
    assert result["tool_count"] > 0
    assert result["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_run_swarm_demo_mode_returns_final_answer() -> None:
    """Demo-mode swarm should hand off through both bees and return text."""
    actor = _actor()
    app_state = _app_state()
    result = await run_swarm(
        "hello",
        app_state=app_state,
        actor=actor,
        event_log=app_state.event_log,
        approve=True,
        llm=StaticChatModel(),
    )
    assert result["text"].startswith("beeAI demo response")
    assert result["tool_count"] > 0
    assert result["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_run_agent_without_approval_still_returns_text() -> None:
    """Without approval the policy gate denies mutating tools but the agent
    still completes (static demo model answers directly)."""
    actor = _actor()
    app_state = _app_state()
    result = await run_agent(
        "do something",
        app_state=app_state,
        actor=actor,
        event_log=app_state.event_log,
        approve=False,
        llm=StaticChatModel(),
    )
    assert isinstance(result["text"], str) and result["text"]


@pytest.mark.asyncio
async def test_static_chat_model_returns_output() -> None:
    """StaticChatModel should return an AssistantMessage with the fixed text."""
    model = StaticChatModel(response="fixed answer")
    from beeai_framework.backend import UserMessage

    out = await model.run([UserMessage("ping")])
    assert out.get_text_content() == "fixed answer"


def test_build_chat_model_production_path() -> None:
    """Production build_chat_model should point at the LiteLLM proxy."""
    model = build_chat_model()
    assert model.provider_id == "openai"
    assert model.model_id == "ornith"


def test_beeai_router_returns_404_when_disabled() -> None:
    """The /beeai routes 404 when the engine is disabled."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from xnch.config import settings
    from xnch.agents.beeai.route import beeai_router

    app = FastAPI()
    app.include_router(beeai_router)
    client = TestClient(app)

    settings.beeai_enabled = False
    resp = client.post("/beeai/chat", json={"message": "hi"})
    assert resp.status_code == 404
    resp = client.get("/beeai/health")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

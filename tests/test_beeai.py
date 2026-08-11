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
    # xnch MCP tools (operator) + framework built-ins (think/meteo/wiki/ddg)
    assert result["tool_count"] >= 5
    assert result["duration_ms"] >= 0


def test_build_tools_includes_framework_builtins_for_agent_and_swarm() -> None:
    """Both chat and swarm get Think/OpenMeteo (+ Wikipedia/DuckDuckGo when installed)."""
    from xnch.agents.beeai.tools import FRAMEWORK_TOOL_NAMES, build_tools

    tools = build_tools(_actor())
    names = {t.name for t in tools}
    assert "think" in names
    assert "OpenMeteoTool" in names
    # extras installed in this env
    assert "Wikipedia" in names
    assert "DuckDuckGo" in names
    assert names >= {"xnch_status", "xnch_memory_recall"}
    assert FRAMEWORK_TOOL_NAMES <= names


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


def test_beeai_router_maps_agent_error_to_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iteration-limit AgentError becomes a stable 422, not an unhandled 500."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from beeai_framework.agents.errors import AgentError

    from xnch.config import settings
    from xnch.agents.beeai import route as beeai_route

    async def _boom(*_args: object, **_kwargs: object) -> dict:
        raise AgentError("Agent was not able to resolve the task in 9 iterations.")

    monkeypatch.setattr(beeai_route, "run_agent", _boom)
    settings.beeai_enabled = True

    app = FastAPI()
    app.include_router(beeai_route.beeai_router)
    client = TestClient(app)

    resp = client.post("/beeai/chat", json={"message": "ok"})
    assert resp.status_code == 422
    assert "iteration limit" in resp.json()["detail"]


def test_beeai_router_maps_timeout_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wall-clock timeout becomes 504 so the client can retry without a restart."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from xnch.config import settings
    from xnch.agents.beeai import route as beeai_route
    from xnch.agents.beeai.runtime import BeeaiTimeoutError

    async def _slow(*_args: object, **_kwargs: object) -> dict:
        raise BeeaiTimeoutError("beeAI run exceeded timeout of 60s")

    monkeypatch.setattr(beeai_route, "run_agent", _slow)
    settings.beeai_enabled = True

    app = FastAPI()
    app.include_router(beeai_route.beeai_router)
    client = TestClient(app)

    resp = client.post("/beeai/chat", json={"message": "hi"})
    assert resp.status_code == 504
    assert "timeout" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_run_agent_timeout_cancels_hung_run() -> None:
    """asyncio.wait_for must abort a hung agent.run so the GPU slot is released."""
    import asyncio

    class _HungAgent:
        async def run(self, *_args: object, **_kwargs: object) -> None:
            await asyncio.sleep(3600)

    actor = _actor()
    app_state = _app_state()

    from xnch.agents.beeai import runtime as beeai_runtime
    from xnch.agents.beeai.runtime import BeeaiTimeoutError

    original_build = beeai_runtime.build_orchestrator

    def _fake_build(**_kwargs: object) -> _HungAgent:
        return _HungAgent()

    beeai_runtime.build_orchestrator = _fake_build  # type: ignore[assignment]
    try:
        with pytest.raises(BeeaiTimeoutError, match="timeout"):
            await run_agent(
                "hang",
                app_state=app_state,
                actor=actor,
                event_log=app_state.event_log,
                llm=StaticChatModel(),
                timeout_s=0.05,
            )
    finally:
        beeai_runtime.build_orchestrator = original_build  # type: ignore[assignment]

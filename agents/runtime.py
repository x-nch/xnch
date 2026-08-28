from __future__ import annotations

from typing import Protocol

from xnch.agents.model_selector import select_model
from xnch.agents.roster import get_agent


class Dispatcher(Protocol):
    def dispatch(
        self,
        agent_key: str,
        persona: str,
        tools: list[str],
        model: str,
        request: str,
    ) -> str: ...


def _estimate_tokens(request: str) -> int:
    return max(1, len(request) // 4)


def invoke_agent(key: str, request: str, dispatcher: Dispatcher | None = None) -> dict:
    agent = get_agent(key)
    if agent is None:
        raise ValueError(f"unknown agent: {key}")
    model = select_model(agent.model_policy, _estimate_tokens(request))
    if dispatcher is None:
        from xnch.agents import _default_dispatcher

        dispatcher = _default_dispatcher
    run_id = dispatcher.dispatch(
        agent_key=key,
        persona=agent.persona,
        tools=agent.allowed_tools,
        model=model,
        request=request,
    )
    return {"run_id": run_id, "model": model}

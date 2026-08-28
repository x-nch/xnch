from __future__ import annotations

from xnch.agents.runtime import invoke_agent


class FakeDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, agent_key, persona, tools, model, request):
        self.calls.append((agent_key, model, request))
        return "run-123"


def test_invoke_uses_selector_and_dispatches():
    d = FakeDispatcher()
    out = invoke_agent("finance", "How was this month?", dispatcher=d)
    assert out["run_id"] == "run-123"
    assert out["model"].startswith("openrouter:")
    assert d.calls[0][0] == "finance"

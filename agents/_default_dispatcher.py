from __future__ import annotations

import os

import httpx

# xnch.config.settings has no XNCH_DISPATCH_URL / XNCH_GATEWAY_SECRET, so we
# read these from the environment with sensible dev defaults.
DISPATCH_URL = os.environ.get("XNCH_DISPATCH_URL", "http://localhost:8001")
GATEWAY_SECRET = os.environ.get("XNCH_GATEWAY_SECRET", "")


class DefaultDispatcher:
    def dispatch(
        self,
        agent_key: str,
        persona: str,
        tools: list[str],
        model: str,
        request: str,
    ) -> str:
        resp = httpx.post(
            f"{DISPATCH_URL}/execution/execute",
            headers={"Authorization": f"Bearer {GATEWAY_SECRET}"},
            json={
                "agent_key": agent_key,
                "persona": persona,
                "tools": tools,
                "model": model,
                "request": request,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("run_id", "pending")


_default_dispatcher = DefaultDispatcher()

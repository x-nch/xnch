"""Governance auth gate + weight-promotion regression gate.

Covers: (1) POST /governance/weights/propose, /weights/approve and /actors
require a valid gateway token when app.state.gateway_secret is set;
(2) approve_weights runs the fitness regression gate against the active
config — measured regressions are rejected unless force=true, skips are
recorded; (3) unit coverage of evaluate_weight_candidate pass/block/skip
paths with injected fakes.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xnch.config import settings
from xnch.learning.evolution.promotion_gate import evaluate_weight_candidate
from xnch.memory.db import init_db
from xnch.security.gateway_token import mint_gateway_token

XNCH_ROOT = Path(__file__).resolve().parent.parent


def _load_governance():
    """Import xnch.routes.governance without executing routes/__init__.py —
    the full router bundle pulls voice/chat (litellm et al.) that these tests
    never touch. Mirrors the controlled-loading style of test_agent_routes."""
    root = str(XNCH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if "xnch" not in sys.modules:
        pkg = types.ModuleType("xnch")
        pkg.__path__ = [root]
        sys.modules["xnch"] = pkg
    routes_pkg = types.ModuleType("xnch.routes")
    routes_pkg.__path__ = [str(XNCH_ROOT / "routes")]
    sys.modules["xnch.routes"] = routes_pkg
    return importlib.import_module("xnch.routes.governance")


governance = _load_governance()

SECRET = "test-secret"

_GOOD_WEIGHTS = {
    "policy_score": 0.05,
    "outcome_score": 0.05,
    "risk_score": 0.85,
    "context_fit_score": 0.05,
}
_BAD_WEIGHTS = {
    "policy_score": 0.85,
    "outcome_score": 0.05,
    "risk_score": 0.05,
    "context_fit_score": 0.05,
}


def _token_header() -> dict[str, str]:
    return {"X-Gateway-Token": mint_gateway_token(SECRET)}


def _make_episodes(n: int = 10) -> list[dict[str, Any]]:
    """Episodes where risk_score aligns with outcome and policy_score anti-aligns."""
    episodes = []
    for i in range(n):
        success = i % 2 == 0
        episodes.append({
            "intent_class": "EXECUTION",
            "outcome": "SUCCESS" if success else "FAILURE",
            "scores_json": json.dumps({
                "policy_score": -1.0 if success else 1.0,
                "outcome_score": 0.0,
                "risk_score": 1.0 if success else -1.0,
                "context_fit_score": 0.0,
            }),
        })
    return episodes


def _async_fetch(episodes: list[dict[str, Any]]):
    async def _fetch(intent_class: str, lookback_days: int) -> list[dict[str, Any]]:
        return episodes

    return _fetch


def _async_current(current: dict[str, Any]):
    async def _current(intent_class: str) -> dict[str, Any]:
        return current

    return _current


# ---------------------------------------------------------------------- #
# Unit: evaluate_weight_candidate                                         #
# ---------------------------------------------------------------------- #

@pytest.fixture()
def good_current() -> dict[str, Any]:
    return {"version": "wc-active", "weights": dict(_GOOD_WEIGHTS)}


async def test_gate_blocks_regression(good_current) -> None:
    result = await evaluate_weight_candidate(
        intent_class="EXECUTION",
        proposed_weights={"policy_score": 0.85, "outcome_score": 0.05,
                          "risk_score": 0.05, "context_fit_score": 0.05},
        fetch_fn=_async_fetch(_make_episodes()),
        current_weights_fn=_async_current({"version": "v", "weights": dict(_GOOD_WEIGHTS)}),
    )
    assert result["status"] == "block"
    assert result["current_fitness"] > result["proposed_fitness"]


async def test_gate_pass_when_better() -> None:
    result = await evaluate_weight_candidate(
        intent_class="EXECUTION",
        proposed_weights=dict(_GOOD_WEIGHTS),
        fetch_fn=_async_fetch(_make_episodes()),
        current_weights_fn=_async_current({"version": "v", "weights": dict(_BAD_WEIGHTS)}),
    )
    assert result["status"] == "pass"
    assert result["proposed_fitness"] == 1.0


async def test_gate_skips_insufficient_data(good_current) -> None:
    result = await evaluate_weight_candidate(
        intent_class="EXECUTION",
        proposed_weights=dict(_GOOD_WEIGHTS),
        fetch_fn=_async_fetch(_make_episodes(4)),
        current_weights_fn=_async_current(good_current),
    )
    assert result["status"] == "skipped"
    assert "insufficient_data" in result["reason"]


async def test_gate_skips_without_baseline() -> None:
    result = await evaluate_weight_candidate(
        intent_class="EXECUTION",
        proposed_weights=dict(_GOOD_WEIGHTS),
        fetch_fn=_async_fetch(_make_episodes()),
        current_weights_fn=_async_current({}),
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "no_active_baseline"


async def test_gate_skips_on_infra_error(good_current) -> None:
    async def boom(intent: str, days: int) -> list[dict[str, Any]]:
        raise ConnectionError("pg down")

    result = await evaluate_weight_candidate(
        intent_class="EXECUTION",
        proposed_weights=dict(_GOOD_WEIGHTS),
        fetch_fn=boom,
        current_weights_fn=_async_current(good_current),
    )
    assert result["status"] == "skipped"
    assert result["reason"].startswith("eval_unavailable")


async def test_default_current_weights_relative_imports_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the evolution-package relative-import depth (`...config`, not `..config`)."""
    from xnch.learning.evolution.promotion_gate import _default_current_weights

    monkeypatch.setattr(settings, "base_dir", tmp_path)
    await init_db(tmp_path / "xnch.db")
    assert await _default_current_weights("EXECUTION") == {}


# ---------------------------------------------------------------------- #
# HTTP surface: auth + gated approval                                     #
# ---------------------------------------------------------------------- #

@pytest.fixture()
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "base_dir", tmp_path)
    await init_db(tmp_path / "xnch.db")

    app = FastAPI()

    async def _noop_increment() -> str:
        return "v-test"

    app.state.increment_state_version = _noop_increment
    app.state.gateway_secret = SECRET
    app.include_router(governance.router)
    return TestClient(app)


def _propose(tc: TestClient, weights: dict[str, float]) -> str:
    r = tc.post(
        "/governance/weights/propose",
        json={
            "intent_class": "EXECUTION",
            "weights": weights,
            "episode_batch": "test-batch",
            "proposed_by": "test",
        },
        headers=_token_header(),
    )
    assert r.status_code == 200, r.text
    return r.json()["version"]


def test_governance_writes_require_token(client: TestClient) -> None:
    assert client.post(
        "/governance/weights/propose", json={"intent_class": "EXECUTION", "weights": {}}
    ).status_code == 401
    assert client.post(
        "/governance/weights/approve?version=wc-x"
    ).status_code == 401
    assert client.post(
        "/governance/actors", json={"actor_id": "a1", "role": "nexi"}
    ).status_code == 401


def test_approve_happy_path_records_skipped_eval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _propose(client, dict(_GOOD_WEIGHTS))
    monkeypatch.setattr(
        governance, "evaluate_weight_candidate",
        _fake_gate({"status": "skipped", "reason": "no_active_baseline"}),
    )

    r = client.post(f"/governance/weights/approve?version={version}", headers=_token_header())
    assert r.status_code == 200, r.text
    assert r.json()["eval"]["status"] == "skipped"

    active = client.get("/governance/weights?intent_class=EXECUTION")
    assert active.status_code == 200
    assert active.json()["version"] == version


def test_approve_blocks_regression(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _propose(client, dict(_BAD_WEIGHTS))
    monkeypatch.setattr(
        governance, "evaluate_weight_candidate",
        _fake_gate({"status": "block", "proposed_fitness": 0.2,
                    "current_fitness": 0.8, "episodes": 10}),
    )

    r = client.post(f"/governance/weights/approve?version={version}", headers=_token_header())
    assert r.status_code == 422
    assert r.json()["detail"]["eval"]["status"] == "block"

    # Pending row survives the block: a non-forced retry is blocked again.
    retry = client.post(f"/governance/weights/approve?version={version}", headers=_token_header())
    assert retry.status_code == 422


def test_approve_force_overrides_block_and_records(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _propose(client, dict(_BAD_WEIGHTS))
    monkeypatch.setattr(
        governance, "evaluate_weight_candidate",
        _fake_gate({"status": "block", "proposed_fitness": 0.2,
                    "current_fitness": 0.8, "episodes": 10}),
    )

    r = client.post(
        f"/governance/weights/approve?version={version}&force=true",
        headers=_token_header(),
    )
    assert r.status_code == 200
    assert r.json()["eval"]["status"] == "block"
    assert "[FORCED]" in r.json()["description"]

    active = client.get("/governance/weights?intent_class=EXECUTION")
    assert active.json()["version"] == version


def _fake_gate(result: dict[str, Any]):
    async def _evaluate(**kwargs: Any) -> dict[str, Any]:
        return result

    return _evaluate

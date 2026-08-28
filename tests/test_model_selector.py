from __future__ import annotations

from xnch.agents.model_selector import select_model, within_budget
from xnch.agents.roster import ComplexityPolicy, ModelPolicy


def _policy(tiers: list[str], thr: int, ceil: float) -> ModelPolicy:
    return ModelPolicy(
        tiers=tiers,
        complexity=ComplexityPolicy(
            tokens_threshold=thr, latency_budget_s=30, price_ceiling_usd=ceil
        ),
        default_tier=tiers[0],
    )


def test_small_uses_default() -> None:
    p = _policy(["openrouter:auto", "openrouter:mid", "openrouter:frontier"], 4000, 0.05)
    assert select_model(p, 500) == "openrouter:auto"


def test_large_steps_up() -> None:
    p = _policy(["openrouter:auto", "openrouter:mid", "openrouter:frontier"], 4000, 0.05)
    assert select_model(p, 9000) == "openrouter:frontier"


def test_budget_blocks() -> None:
    p = _policy(["openrouter:auto"], 4000, 0.02)
    assert within_budget(p, 1000, 0.05) is False
    assert within_budget(p, 1000, 0.01) is True

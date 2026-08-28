from __future__ import annotations

from .roster import ComplexityPolicy, ModelPolicy


def select_model(
    policy: ModelPolicy,
    estimated_tokens: int,
    latency_budget_s: int | None = None,
) -> str:
    tiers = policy.tiers or [policy.default_tier]
    if estimated_tokens <= policy.complexity.tokens_threshold:
        return tiers[0]
    overshoot = estimated_tokens // max(policy.complexity.tokens_threshold, 1)
    idx = min(overshoot, len(tiers) - 1)
    return tiers[idx]


def within_budget(policy: ModelPolicy, estimated_tokens: int, price_usd: float) -> bool:
    if price_usd > policy.complexity.price_ceiling_usd:
        return False
    return True

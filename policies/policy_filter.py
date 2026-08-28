"""Per-agent spend-budget gate for the policy layer."""

from xnch.agents.roster import get_agent
from xnch.agents.model_selector import within_budget


def check_agent_budget(agent_key: str, estimated_tokens: int, price_usd: float) -> bool:
    agent = get_agent(agent_key)
    if agent is None:
        return False
    return within_budget(agent.model_policy, estimated_tokens, price_usd)

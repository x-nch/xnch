from xnch.agents.roster import get_agent, load_roster


def test_loads_eighteen():
    assert len(load_roster()) == 18


def test_get_known_agent():
    agent = get_agent("chief_of_staff")
    assert agent is not None
    assert agent.name == "Chief of staff"
    assert agent.model_policy.default_tier.startswith("openrouter:")


def test_missing_agent_none():
    assert get_agent("nope") is None

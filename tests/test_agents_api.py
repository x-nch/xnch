"""Tests for the /agents roster HTTP endpoints."""
from fastapi.testclient import TestClient

from xnch.main import app

client = TestClient(app)


def test_list_agents():
    resp = client.get("/agents")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 18
    assert body[0]["key"] == "chief_of_staff"


def test_get_agent_detail():
    resp = client.get("/agents/finance")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Finance"


def test_get_agent_404():
    resp = client.get("/agents/nope")
    assert resp.status_code == 404

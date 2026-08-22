"""Classifier hard-reasoning override for optillm operational mode."""
from __future__ import annotations

from xnch.routing.classifier import classify_request


def test_hard_reasoning_routes_to_ornith_reasoned(monkeypatch) -> None:
    monkeypatch.setattr("xnch.routing.classifier._cache_lookup", lambda *_: None)
    monkeypatch.setattr("xnch.routing.classifier._cache_store", lambda *a, **k: None)
    route = classify_request("think hard", "operator", {"hard_reasoning": True})
    assert route.model_name == "ornith-reasoned"
    assert "hard_reasoning" in route.reason


def test_default_still_qwen(monkeypatch) -> None:
    monkeypatch.setattr("xnch.routing.classifier._cache_lookup", lambda *_: None)
    monkeypatch.setattr("xnch.routing.classifier._cache_store", lambda *a, **k: None)
    route = classify_request("hello", "operator", {})
    assert route.model_name == "qwen2.5-vl-7b"

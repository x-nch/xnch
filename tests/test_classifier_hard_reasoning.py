"""Classifier routing — all requests resolve to the local ornith model."""
from __future__ import annotations

from xnch.routing.classifier import classify_request


def test_hard_reasoning_routes_to_ornith(monkeypatch) -> None:
    monkeypatch.setattr("xnch.routing.classifier._cache_lookup", lambda *_: None)
    monkeypatch.setattr("xnch.routing.classifier._cache_store", lambda *a, **k: None)
    route = classify_request("think hard", "operator", {"hard_reasoning": True})
    assert route.model_name == "ornith"
    assert route.reason == "default route: ornith"


def test_default_routes_to_ornith(monkeypatch) -> None:
    monkeypatch.setattr("xnch.routing.classifier._cache_lookup", lambda *_: None)
    monkeypatch.setattr("xnch.routing.classifier._cache_store", lambda *a, **k: None)
    route = classify_request("hello", "operator", {})
    assert route.model_name == "ornith"

"""Shared test fixtures.

Tests exercise routers against minimal apps; most set a real gateway secret.
``allow_open_gateway`` is forced True suite-wide so the fail-closed default
(503 when no secret is configured) doesn't leak into unrelated tests — the
dedicated fail-closed tests opt back out via monkeypatch.
"""
from __future__ import annotations

import pytest

from xnch.config import settings


@pytest.fixture(autouse=True)
def _allow_open_gateway_in_tests():
    settings.allow_open_gateway = True
    yield
    settings.allow_open_gateway = False

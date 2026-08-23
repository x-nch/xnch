"""Short-lived HMAC tokens minted by the web proxy, verified by xnch.

Hybrid-B gate (spec §4): write endpoints under /workflows/* and /approvals/*
require a valid token OR the shared service key. Legacy routes are untouched.
Token format: ``<expiry_epoch>.<hex_hmac(secret, expiry_epoch)>`` — constant-
time compared. Stdlib only.
"""
from __future__ import annotations

import hmac
import hashlib
import time


def mint_gateway_token(secret: str, ttl_s: int = 300) -> str:
    expiry = str(int(time.time()) + ttl_s)
    sig = hmac.new(secret.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def verify_gateway_token(secret: str, token: str, *, now: float | None = None) -> bool:
    if not secret or not token:
        return False
    try:
        expiry_raw, sig = token.split(".", 1)
        expiry = int(expiry_raw)
    except (ValueError, AttributeError):
        return False
    now = now if now is not None else time.time()
    if expiry < now:
        return False
    expected = hmac.new(secret.encode(), expiry_raw.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def verify_service_key(secret: str, presented: str | None) -> bool:
    """Service identity check for nexi (Phase 2): exact shared key."""
    if not secret or not presented:
        return False
    return hmac.compare_digest(secret, presented)

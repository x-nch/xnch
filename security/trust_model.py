from __future__ import annotations

import enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request


class TrustLevel(enum.IntEnum):
    UNTRUSTED = 1
    EXTERNAL_AGENT = 2
    TRUSTED_AGENT = 3
    OWNER = 4
    SYSTEM = 5


ACTOR_TRUST_MAP: dict[str, TrustLevel] = {
    "nexi": TrustLevel.SYSTEM,
    "openclaw": TrustLevel.OWNER,
    "claude_code": TrustLevel.TRUSTED_AGENT,
    "opencode": TrustLevel.TRUSTED_AGENT,
    "perception_daemon": TrustLevel.TRUSTED_AGENT,
    "consolidation_job": TrustLevel.TRUSTED_AGENT,
    "external": TrustLevel.UNTRUSTED,
}


def get_trust_level(actor_role: str) -> TrustLevel:
    return ACTOR_TRUST_MAP.get(actor_role, TrustLevel.UNTRUSTED)


def requires_trust(minimum: TrustLevel):
    def decorator(func: Any) -> Any:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Request | None = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                for _, v in kwargs.items():
                    if isinstance(v, Request):
                        request = v
                        break
            if request is None:
                raise HTTPException(status_code=500, detail="Request object not found")

            actor_role = request.headers.get("X-Actor-Role", "")
            level = get_trust_level(actor_role)
            if level.value < minimum.value:
                raise HTTPException(
                    status_code=403,
                    detail=f"Actor '{actor_role}' (trust level {level.name}) "
                           f"does not meet minimum trust level {minimum.name}",
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

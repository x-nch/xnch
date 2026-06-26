from __future__ import annotations

from dataclasses import dataclass

from xnch.security.trust_model import TrustLevel, get_trust_level


@dataclass
class ActorCapabilities:
    can_write_memory: bool = False
    can_read_all_memory: bool = False
    can_trigger_jobs: bool = False
    can_modify_policies: bool = False
    can_access_perception: bool = False


CAPABILITY_MAP: dict[TrustLevel, ActorCapabilities] = {
    TrustLevel.SYSTEM: ActorCapabilities(
        can_write_memory=True,
        can_read_all_memory=True,
        can_trigger_jobs=True,
        can_modify_policies=True,
        can_access_perception=True,
    ),
    TrustLevel.OWNER: ActorCapabilities(
        can_write_memory=True,
        can_read_all_memory=True,
        can_trigger_jobs=True,
        can_modify_policies=False,
        can_access_perception=True,
    ),
    TrustLevel.TRUSTED_AGENT: ActorCapabilities(
        can_write_memory=True,
        can_read_all_memory=False,
        can_trigger_jobs=True,
        can_modify_policies=False,
        can_access_perception=False,
    ),
    TrustLevel.EXTERNAL_AGENT: ActorCapabilities(
        can_write_memory=False,
        can_read_all_memory=False,
        can_trigger_jobs=False,
        can_modify_policies=False,
        can_access_perception=False,
    ),
    TrustLevel.UNTRUSTED: ActorCapabilities(
        can_write_memory=False,
        can_read_all_memory=False,
        can_trigger_jobs=False,
        can_modify_policies=False,
        can_access_perception=False,
    ),
}


def get_capabilities(actor_role: str) -> ActorCapabilities:
    level = get_trust_level(actor_role)
    return CAPABILITY_MAP.get(level, CAPABILITY_MAP[TrustLevel.UNTRUSTED])

from .admin import router as admin_router
from .session import router as session_router
from .memory import router as memory_router
from .policy import router as policy_router
from .verdict import router as verdict_router
from .execution import router as execution_router
from .governance import router as governance_router
from .auth import router as auth_router
from .nexi_gateway import router as nexi_gateway_router
from .chat import router as chat_router
from .voice import router as voice_router
from .goals import router as goal_router
from .pipeline import router as pipeline_router
from .workflows import approvals_router, router as workflows_router
from .agents import router as agents_router

__all__ = [
    "session_router", "memory_router", "policy_router",
    "verdict_router", "execution_router", "governance_router",
    "auth_router", "nexi_gateway_router", "chat_router", "admin_router",
    "voice_router", "goal_router", "pipeline_router",
    "workflows_router", "approvals_router", "agents_router",
]

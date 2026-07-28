"""Deep Agents CompositeBackend for XNCH memory consolidation."""
import os
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

DATABASE_URL = os.environ.get("XNCH_POSTGRES_URL", "postgresql://localhost:5432/xnch")


def create_memory_backend(runtime):
    """Create CompositeBackend with persistent routes for episodes and patterns.

    Default (StateBackend): ephemeral working files, session-scoped.
    /episodes/: persistent episodic memory (cross-session).
    /patterns/: persistent pattern store (cross-session).
    /entities/: persistent entity memory (cross-session).
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/episodes/": StoreBackend(runtime),
            "/patterns/": StoreBackend(runtime),
            "/entities/": StoreBackend(runtime),
        },
    )


# Production: use PostgresStore instead of InMemoryStore
# from langgraph.store.postgres import PostgresStore
# store = PostgresStore.from_conn_string(DATABASE_URL)

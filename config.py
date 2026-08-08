from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XNCH_", env_file=".env")

    # Paths
    base_dir: Path = Path("~/.xnch").expanduser()

    @property
    def keys_dir(self) -> Path:
        return self.base_dir / "keys"

    @property
    def audit_dir(self) -> Path:
        return self.base_dir / "audit"

    @property
    def db_path(self) -> Path:
        return self.base_dir / "xnch.db"

    @property
    def governance_dir(self) -> Path:
        return self.base_dir / "governance"

    @property
    def policies_dir(self) -> Path:
        return self.base_dir / "policies"

    @property
    def weights_dir(self) -> Path:
        return self.base_dir / "weights"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    auth_secret: str = "dev-secret-change-in-production"
    token_ttl_ms: int = 30_000

    # Session
    session_ttl_s: int = 120
    rate_limit_per_minute: int = 10

    # Nexi callback
    nexi_base_url: str = "http://localhost:8000"

    # PostgreSQL / pgvector
    postgres_url: str = "postgresql://xnch:cf00d3e9a10c400f9083b424b94f0cf7@localhost:5432/xnch"

    # Learning
    pattern_min_observations: int = 10
    score_adapter_accuracy_threshold: float = 0.6

    # Memory recall (context assembly)
    memory_recall_min_score: float = 0.35

    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # LiteLLM proxy
    litellm_proxy_url: str = "http://litellm:4000"

    # Graph extractor
    graph_extractor_model: str = "ollama/phi3:mini"

    # Perception
    vault_dir: Path = Path("~/.xnch/vault").expanduser()
    perception_redis_db: int = 0
    attention_silence_threshold_s: float = 1.5
    attention_screen_diff_threshold: float = 0.15
    attention_idle_timeout_s: int = 600

    # Read-only filesystem (Nexi MCP tools)
    fs_policy_path: Path = Path("~/.xnch/fs-policy.yaml").expanduser()
    fs_local_host: str = "node-a"
    fs_agent_node_b_url: str = "http://192.168.50.2:8003"
    fs_agent_token: str = ""
    fs_max_read_bytes: int = 2_097_152
    fs_max_list_entries: int = 1000
    fs_max_glob_results: int = 200

    # Governed command execution (Nexi MCP tools)
    exec_policy_path: Path = Path("~/.xnch/exec-policy.yaml").expanduser()
    exec_local_host: str = "node-a"
    exec_agent_node_b_url: str = "http://192.168.50.2:8004"
    exec_agent_token: str = ""

    # External MCP bridge (federated MCP servers for Nexi runtime)
    mcp_bridge_enabled: bool = True
    mcp_servers_path: Path = Path("~/.xnch/mcp-servers.yaml").expanduser()
    mcp_max_tool_rounds: int = 3
    mcp_max_tool_rounds_with_bridge: int = 5

    # Anonymous web search (Nexi MCP tools via self-hosted SearXNG)
    web_search_policy_path: Path = Path("~/.xnch/web-search.yaml").expanduser()
    searxng_url: str = "http://127.0.0.1:8888"

    # Memory routing (episodic vs agentmemory)
    memory_routing_policy_path: Path = Path("~/.xnch/memory-routing.yaml").expanduser()
    am_prefetch_enabled: bool = False


settings = Settings()

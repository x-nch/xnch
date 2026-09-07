from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScraperSettings(BaseSettings):
    """Scraper-specific config. Env vars prefixed with SCRAPER_."""

    model_config = SettingsConfigDict(env_prefix="SCRAPER_")

    default_tier: str = "auto"
    max_concurrent: int = 5
    request_timeout: float = 30.0
    instagram_session: str | None = None
    twitter_username: str | None = None
    twitter_password: str | None = None
    twitter_email: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XNCH_")

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

    # Execution sandbox (node-b)
    executor_url: str = "http://192.168.50.2:8083"

    # Self (xnch) base URL — used by background jobs that POST to own API
    self_base_url: str = "http://localhost:8001"

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

    # Prometheus instrumentation
    metrics_enabled: bool = True
    metrics_allow_cidrs: list[str] = Field(default_factory=lambda: ["127.0.0.1", "::1", "192.168.50.0/24"])
    deep_health_interval_s: float = 30.0

    # Alertmanager webhook receiver (POST /admin/alerts). Includes the docker
    # bridge range so the alertmanager container can reach host-gateway.
    alert_webhook_allow_cidrs: list[str] = Field(
        default_factory=lambda: [
            "127.0.0.1", "::1", "192.168.50.0/24", "172.16.0.0/12",
        ]
    )
    recent_alerts_capacity: int = 200

    # OpenCode Go API (hosted DeepSeek V4)
    opencode_go_api_url: str = "https://opencode.ai/zen/go/v1"
    opencode_go_api_key: str = ""
    opencode_go_api_timeout_s: float = 60.0

    # LiteLLM proxy (legacy; kept for rollback)
    litellm_proxy_url: str = ""

    # LLM availability probe (OpenCode Go hosted)
    llm_status_url: str = "https://opencode.ai/zen/go/v1/models"
    # Which model the gateway asks nexi's chat tool-loop to use. "nexi-default"
    # is an alias meaning "let nexi's model router decide"; set any concrete
    # model id (e.g. anthropic/claude-sonnet-4) to pin it per-deploy.
    llm_model_id: str = "nexi-default"
    llm_probe_timeout_s: float = 5.0

    # Prometheus (operator UI summarizer; runs co-located on Node A)
    prometheus_url: str = "http://127.0.0.1:9090"
    prometheus_timeout_s: float = 4.0

    # Graph extractor (OpenCode Go hosted DeepSeek V4)
    graph_extractor_model: str = "deepseek-v4-pro"
    graph_extractor_provider_hint: str = ""

    # Session ingest (OpenCode SQLite logs -> episodic + semantic tiers)
    session_ingest_db_path: Path = Path(
        "~/.local/share/opencode/opencode.db"
    ).expanduser()
    session_ingest_model: str = "deepseek-v4-pro"
    session_ingest_max_tokens: int = 4096
    session_ingest_project_dirs: str = ""
    session_ingest_scheduled: bool = True
    session_ingest_cron_minute: int = 15

    # Goal-step auto-dispatch (v1): cron files a goal_step APPROVAL;
    # human approve spawns the agent_run. Default off.
    goal_dispatch_enabled: bool = False
    goal_dispatch_cron_minute: int = 30
    goal_dispatch_goal_id: str = "2c821d69-c7c3-42ff-96cb-d1aaddc245b0"

    # Direct agent dispatch (POST /agents/dispatch) bypasses the approval
    # gate, so it is deny-by-default (2026-08-24 audit F7). Enable only if
    # the muse-style manual dispatch path is wanted.
    agents_direct_dispatch_enabled: bool = False
    # Comma-separated keywords matched against plan-entry "action" text.
    # Fail-closed: only matching actions file risk_class='low'; anything
    # else — including an EMPTY allowlist — files 'elevated', which the
    # decide route gates to X-Actor-Role: admin.
    goal_dispatch_allowed_actions: str = ""

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

    # HITL gate on the LangGraph decision pipeline
    langgraph_pipeline: bool = False
    hitl_execution_mode: str = "always"
    hitl_risk_threshold: float = 0.5

    # Gateway token (Hybrid-B): shared HMAC secret with the web proxy.
    # Empty + allow_open_gateway=False ⇒ gated routes 503 (fail-closed).
    # Set XNCH_ALLOW_OPEN_GATEWAY=1 only for throwaway dev instances.
    gateway_secret: str = ""
    allow_open_gateway: bool = False

    # Workflow executor (P2): when True, approving a step leaves it APPROVED
    # for nexi to claim+execute; False keeps v1 approve⇒DONE semantics.
    workflow_executor_enabled: bool = False
    workflow_step_claim_lease_s: int = 120

    # Voice (STT + TTS on gate7 CPU)
    voice_enabled: bool = True
    voice_stt_model: str = "base"
    voice_stt_device: str = "cpu"
    voice_stt_compute_type: str = "int8"
    voice_stt_language: str = "en"
    voice_tts_engine: str = "piper"
    voice_tts_voice_path: Path = Path("~/.xnch/voice/en_US-lessac-medium.onnx").expanduser()
    voice_tts_config_path: Path = Path(
        "~/.xnch/voice/en_US-lessac-medium.onnx.json"
    ).expanduser()
    voice_max_audio_duration_s: float = 60.0
    voice_max_audio_bytes: int = 10_485_760
    voice_max_tts_chars: int = 2000
    voice_models_dir: Path = Path("~/.xnch/voice/models").expanduser()

    # Scraper
    scraper: ScraperSettings = ScraperSettings()


settings = Settings()

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


settings = Settings()

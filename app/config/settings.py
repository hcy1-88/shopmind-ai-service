"""Application settings and configuration management."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """对应.env中的配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="shopmind-ai-service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Nacos Configuration
    nacos_server_addr: str = Field(
        default="127.0.0.1:8848",
        description="Nacos server address",
    )
    nacos_namespace: str = Field(
        default="public",
        description="Nacos namespace",
    )
    nacos_group: str = Field(
        default="DEFAULT_GROUP",
        description="Nacos group",
    )
    nacos_data_id: str = Field(
        default="shopmind-ai-service.yaml",
        description="Nacos config data ID",
    )
    nacos_username: Optional[str] = Field(
        default=None,
        description="Nacos username",
    )
    nacos_password: Optional[str] = Field(
        default=None,
        description="Nacos password",
    )

    # Service Registration
    service_name: str = Field(
        default="ai-service",
        description="Service name for registration",
    )
    service_port: int = Field(
        default=8000,
        description="Service port",
    )
    service_cluster: str = Field(
        default="DEFAULT",
        description="Service cluster",
    )
    service_metadata: dict = Field(
        default_factory=lambda: {"version": "0.1.0"},
        description="Service metadata",
    )

    # PostgreSQL (优先 Nacos，若无 则取 .env 中的配置)
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(default="postgres", description="PostgreSQL password")
    postgres_db: str = Field(default="shopmind-dev", description="PostgreSQL database")
    postgres_pool_size: int = Field(default=10, description="Connection pool size")
    postgres_max_overflow: int = Field(default=20, description="Max overflow connections")

    # Milvus (优先 Nacos，若无 则取 .env 中的配置)
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    milvus_user: Optional[str] = Field(default=None, description="Milvus user")
    milvus_password: Optional[str] = Field(default=None, description="Milvus password")
    milvus_db_name: str = Field(default="default", description="Milvus database name")

    # RocketMQ (优先 Nacos，若无 则取 .env 中的配置)
    rocketmq_namesrv_addr: str = Field(
        default="127.0.0.1:9876",
        description="RocketMQ NameServer address",
    )
    rocketmq_access_key: Optional[str] = Field(
        default=None,
        description="RocketMQ access key",
    )
    rocketmq_secret_key: Optional[str] = Field(
        default=None,
        description="RocketMQ secret key",
    )
    rocketmq_group_id: str = Field(
        default="ai-service-group",
        description="RocketMQ producer group ID",
    )

    # LLM Configuration (优先 Nacos，若无 则取 .env 中的配置)
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, tongyi, etc.)",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_api_base: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )
    openai_vision_model: str = Field(
        default="gpt-4o",
        description="OpenAI vision model name",
    )
    llm_temperature: float = Field(
        default=0.7,
        description="LLM temperature",
    )
    llm_max_tokens: int = Field(
        default=2000,
        description="LLM max tokens",
    )
    llm_timeout: int = Field(
        default=60,
        description="LLM request timeout in seconds",
    )

    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """
    加载 .env 中的配置，得到 settings 实例

    Returns:
        Settings instance
    """
    return Settings()

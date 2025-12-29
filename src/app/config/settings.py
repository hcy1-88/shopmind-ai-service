"""Application settings and configuration management."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.app.utils.ip import get_local_ip

# 模块级别的单例实例（避免与 Pydantic 字段系统冲突）
_settings_instance: Optional["Settings"] = None


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
    service_ip: str = Field(
        default_factory=get_local_ip, description="Service IP"
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


    # Milvus (优先 Nacos，若无 则取 .env 中的配置)
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    milvus_user: Optional[str] = Field(default=None, description="Milvus user")
    milvus_password: Optional[str] = Field(default=None, description="Milvus password")
    milvus_db_name: str = Field(default="default", description="Milvus database name")


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

    @classmethod
    def get_instance(cls) -> "Settings":
        """
        获取 Settings 单例实例.

        Returns:
            Settings 实例
        """
        global _settings_instance
        if _settings_instance is None:
            _settings_instance = cls()
        return _settings_instance



def get_settings() -> Settings:
    """
    获取 Settings 单例实例.

    Returns:
        Settings 单例实例
    """
    return Settings.get_instance()

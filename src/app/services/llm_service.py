"""统一的大模型服务入口，提供单例 LLM 服务."""

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from app.config.nacos_client import get_nacos_client
from app.provider.llm_provider import OpenAIProvider, LLMProvider, TongyiProvider
from app.utils.logger import app_logger as logger


class LLMService:
    """统一的大模型服务，提供不同的 LLM 模型实例."""

    _instance: Optional["LLMService"] = None

    def __init__(self):
        """Initialize LLM service."""
        self.provider: Optional[LLMProvider] = None
        self._config: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "LLMService":
        """
        获取 LLM 服务单例实例.

        Returns:
            LLMService 实例
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """从 Nacos 初始化 LLM 服务配置."""
        try:
            # 从 Nacos 获取 LLM 配置
            nacos_client = get_nacos_client()
            self._config = nacos_client.get_llm_config()

            # 根据配置初始化提供商（默认使用 OpenAI）
            provider_name = self._config.get("provider", "openai").lower()

            if provider_name == "openai":
                self.provider = OpenAIProvider(self._config.get("openai", {}))
            elif provider_name == "tongyi":
                self.provider = TongyiProvider(self._config.get("tongyi", {}))
            else:
                logger.warning(
                    f"未知的提供商 {provider_name}，回退到 OpenAI",
                )
                self.provider = OpenAIProvider(self._config.get("openai", {}))
            self.provider_name = provider_name

            logger.info(
                "LLM provider 初始化完毕！",
                extra={"provider": provider_name},
            )

        except Exception as e:
            logger.error(f"初始化 LLM 服务失败: {e}")
            raise

    def get_chat_model(self, **kwargs) -> BaseChatModel:
        """
        获取聊天模型实例.

        Args:
            **kwargs: 额外的模型参数

        Returns:
            BaseChatModel 实例
        """
        if not self.provider:
            raise RuntimeError("LLM Provider 未初始化")

        return self.provider.get_chat_model(**kwargs)

    def get_vision_model(self, **kwargs) -> BaseChatModel:
        """
        获取视觉模型实例.

        Args:
            **kwargs: 额外的模型参数

        Returns:
            BaseChatModel 实例
        """
        if not self.provider:
            raise RuntimeError("LLM Provider 未初始化")

        return self.provider.get_vision_model(**kwargs)

    def get_reasoning_model(self, **kwargs) -> BaseChatModel:
        """
        获取推理模型实例.

        Args:
            **kwargs: 额外的模型参数

        Returns:
            BaseChatModel 实例，若未配置则降级返回 chat_model
        """
        if not self.provider:
            raise RuntimeError("LLM Provider 未初始化")

        return self.provider.get_reasoning_model(**kwargs)

    def reload_config(self) -> None:
        """从 Nacos 重新加载配置."""
        logger.info("重新加载 LLM 配置")
        self._initialize()


def get_llm_service() -> LLMService:
    """
    获取 LLM 服务单例实例.

    Returns:
        LLMService 单例实例
    """
    return LLMService.get_instance()


async def init_llm_service() -> None:
    llm_service = get_llm_service()
    logger.info(f"大语言模型初始化成功！提供商：{llm_service.provider_name}")
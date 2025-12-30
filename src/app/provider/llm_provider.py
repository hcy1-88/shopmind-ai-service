"""LLM provider 抽象层，支持多种 LLM 提供商."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.utils.logger import app_logger as logger


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(self, **kwargs) -> BaseChatModel:
        """
        Get chat model instance.

        Returns:
            BaseChatModel instance
        """
        pass

    @abstractmethod
    def get_vision_model(self, **kwargs) -> BaseChatModel:
        """
        Get vision model instance for image understanding.

        Returns:
            BaseChatModel instance
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: OpenAI configuration
        """
        self.config = config

    def get_chat_model(self, **kwargs) -> BaseChatModel:
        """Get OpenAI chat model."""
        model_kwargs = {
            "model": kwargs.get("model", self.config.get("model", "gpt-4o-mini")),
            "temperature": kwargs.get(
                "temperature",
                self.config.get("temperature", 0.7),
            ),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 2000)),
            "timeout": kwargs.get("timeout", self.config.get("timeout", 60)),
        }

        # Set API key and base URL
        if self.config.get("api_key"):
            model_kwargs["api_key"] = self.config["api_key"]
        if self.config.get("api_base"):
            model_kwargs["base_url"] = self.config["api_base"]

        return ChatOpenAI(**model_kwargs)

    def get_vision_model(self, **kwargs) -> BaseChatModel:
        """Get OpenAI vision model."""
        model_kwargs = {
            "model": kwargs.get(
                "model",
                self.config.get("vision_model", "gpt-4o"),
            ),
            "temperature": kwargs.get(
                "temperature",
                self.config.get("temperature", 0.7),
            ),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 2000)),
            "timeout": kwargs.get("timeout", self.config.get("timeout", 60)),
        }

        if self.config.get("api_key"):
            model_kwargs["api_key"] = self.config["api_key"]
        if self.config.get("api_base"):
            model_kwargs["base_url"] = self.config["api_base"]

        return ChatOpenAI(**model_kwargs)


class TongyiProvider(LLMProvider):
    """Tongyi (Qwen) LLM provider - placeholder for future implementation."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Tongyi provider.

        Args:
            config: Tongyi configuration
        """
        self.config = config

    def get_chat_model(self, **kwargs) -> BaseChatModel:
        """Get Tongyi chat model."""
        return OpenAIProvider(self.config).get_chat_model(**kwargs)

    def get_vision_model(self, **kwargs) -> BaseChatModel:
        """Get Tongyi vision model."""
        return OpenAIProvider(self.config).get_vision_model(**kwargs)

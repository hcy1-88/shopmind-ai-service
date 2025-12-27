"""LangChain chain for product text generation (description, summary, etc.)."""
import asyncio
import base64
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_llm_service
from app.utils.image_util import load_image_from_url
from app.utils.logger import app_logger as logger


class BaseProductTextChain(ABC):
    """负责商品相关的文本生成链 (description, summary, etc.)."""

    def __init__(self):
        """Initialize base text generation chain."""
        self.llm_service = get_llm_service()
        self.text_prompt = self._create_text_prompt()

    @abstractmethod
    def _create_text_prompt(self) -> ChatPromptTemplate:
        """
        Create the text-only prompt template.

        Subclasses must implement this method to define their specific prompts.

        Returns:
            ChatPromptTemplate for text-only generation
        """
        pass

    @abstractmethod
    def _create_vision_prompt_text(self, title: str, category: str) -> str:
        """
        Create the prompt text for vision model.

        Subclasses must implement this method to define their specific vision prompts.

        Args:
            title: Product title
            category: Product category

        Returns:
            Prompt text for vision model
        """
        pass

    @abstractmethod
    def _get_log_prefix(self) -> str:
        """
        Get the log prefix for this chain.

        Returns:
            Log prefix string (e.g., "Description", "Summary")
        """
        pass

    @abstractmethod
    def _get_fallback_text(self, title: str, category: str) -> str:
        """
        Get fallback text when generation fails.

        Args:
            title: Product title
            category: Product category

        Returns:
            Fallback text
        """
        pass

    async def generate_with_images(
            self,
            title: str,
            category: str,
            image_urls: list[str],
    ) -> str:
        """
        Generate text based on multiple images.

        Args:
            title: Product title
            category: Product category
            image_urls: List of product image URLs

        Returns:
            Generated text
        """
        if not image_urls:
            logger.warning(f"{self._get_log_prefix()}: No images provided, falling back to text-only generation")
            return await self.generate_text_only(title, category)

        try:
            # Get vision model
            vision_model = self.llm_service.get_vision_model()

            # Load all images concurrently as base64 for better performance
            tasks = [load_image_from_url(url) for url in image_urls]
            image_base64_list = await asyncio.gather(*tasks)

            # Build content: text first, then all images
            prompt_text = self._create_vision_prompt_text(title, category)
            content = [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ]

            for base64_str in image_base64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_str},
                })

            message = HumanMessage(content=content)

            # Call model
            response = await vision_model.ainvoke([message])
            result_text = response.content.strip()

            logger.info(
                f"{self._get_log_prefix()} generated with images",
                extra={
                    "title": title[:50],
                    "image_count": len(image_urls),
                },
            )

            return result_text

        except Exception as e:
            logger.error(f"{self._get_log_prefix()} generation with images failed: {e}", exc_info=True)
            # Fallback to text-only generation
            return await self.generate_text_only(title, category)

    async def generate_text_only(self, title: str, category: str) -> str:
        """
        Generate text using only text input (no images).

        Args:
            title: Product title
            category: Product category

        Returns:
            Generated text
        """
        try:
            # Get chat model
            llm = self.llm_service.get_chat_model()

            # Create chain
            chain = self.text_prompt | llm

            # Run chain
            response = await chain.ainvoke(
                {
                    "title": title,
                    "category": category,
                },
            )

            result_text = response.content.strip()

            logger.info(
                f"{self._get_log_prefix()} generated (text-only)",
                extra={"title": title[:50]},
            )

            return result_text

        except Exception as e:
            logger.error(f"Error generating text-only {self._get_log_prefix().lower()}: {e}")
            # Return fallback text
            return self._get_fallback_text(title, category)

    async def generate(
        self,
        title: str,
        category: str,
        image_urls: list[str],
    ) -> str:
        """
        Generate text (description, summary, etc.).

        Args:
            title: 商品标题
            category: 商品分类
            image_urls: 图片

        Returns:
            生成文本
        """
        # Try to generate with images first
        if image_urls:
            return await self.generate_with_images(title, category, image_urls)
        else:
            return await self.generate_text_only(title, category)


# ==================== Concrete Implementations ====================


class DescriptionGenerationChain(BaseProductTextChain):
    """生成商品描述的 chain."""

    _instance: Optional["DescriptionGenerationChain"] = None

    def _create_text_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for description generation (text-only)."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商文案撰写专家。你的任务是根据商品标题和类目生成吸引人的商品描述。

文案要求：
1. 突出商品的核心卖点和特色
2. 使用生动、吸引人的语言
3. 符合电商规范，避免虚假宣传
4. 长度控制在100-300字之间
5. 结构清晰，易于阅读
6. 适当使用分段和符号增强可读性

请直接返回生成的商品描述文本，不要包含任何额外的说明或格式。""",
                ),
                (
                    "human",
                    "商品标题：{title}\n商品类目：{category}\n\n请生成商品描述：",
                ),
            ],
        )

    def _create_vision_prompt_text(self, title: str, category: str) -> str:
        """Create prompt text for description generation with images."""
        return f"""你是一个专业的电商文案撰写专家。请根据以下信息生成吸引人的商品描述：

商品标题：{title}
商品类目：{category}

文案要求：
1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的视觉特点
2. 突出商品的核心卖点和特色
3. 使用生动、吸引人的语言
4. 符合电商规范，避免虚假宣传
5. 长度控制在100-300字之间
6. 结构清晰，易于阅读

请直接返回生成的商品描述文本。"""

    def _get_log_prefix(self) -> str:
        """Get log prefix for description generation."""
        return "Description"

    def _get_fallback_text(self, title: str, category: str) -> str:
        """Get fallback description text."""
        return f"这是一款来自{category}类目的优质商品：{title}。详情请查看商品图片和详细信息。"

    @classmethod
    def get_instance(cls) -> "DescriptionGenerationChain":
        """
        单例

        Returns:
            DescriptionGenerationChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class SummaryGenerationChain(BaseProductTextChain):
    """Chain for generating product summaries."""

    _instance: Optional["SummaryGenerationChain"] = None

    def _create_text_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for summary generation (text-only)."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商文案撰写专家。你的任务是根据商品标题和类目生成简洁的商品摘要。

文案要求：
1. 高度概括商品的核心特点和卖点
2. 语言精炼，突出重点
3. 符合电商规范，避免虚假宣传
4. 长度严格控制在200字以内
5. 适合用作商品简介或短描述

请直接返回生成的商品摘要文本，不要包含任何额外的说明或格式。""",
                ),
                (
                    "human",
                    "商品标题：{title}\n商品类目：{category}\n\n请生成商品摘要（最多200字）：",
                ),
            ],
        )

    def _create_vision_prompt_text(self, title: str, category: str) -> str:
        """Create prompt text for summary generation with images."""
        return f"""你是一个专业的电商文案撰写专家。请根据以下信息生成简洁的商品摘要：

商品标题：{title}
商品类目：{category}

文案要求：
1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的核心特点
2. 高度概括商品的核心特点和卖点
3. 语言精炼，突出重点
4. 符合电商规范，避免虚假宣传
5. 长度严格控制在200字以内
6. 适合用作商品简介或短描述

请直接返回生成的商品摘要文本。"""

    def _get_log_prefix(self) -> str:
        """Get log prefix for summary generation."""
        return "Summary"

    def _get_fallback_text(self, title: str, category: str) -> str:
        """Get fallback summary text."""
        return f"{category}类目商品：{title}"

    @classmethod
    def get_instance(cls) -> "SummaryGenerationChain":
        """
        单例 SummaryGenerationChain.

        Returns:
            SummaryGenerationChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

"""LangChain chain for product text generation (description, summary, etc.)."""
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List
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
        创建仅基于文本的提示模板。

        子类必须实现此方法以定义其特定的提示。

        返回：
            仅用于文本生成的ChatPromptTemplate
        """
        pass

    @abstractmethod
    def _create_vision_prompt_text(self, title: str, description: Optional[str]) -> str:
        """
        根据标题和描述创建用于视觉模型的提示文本。

        子类必须实现此方法以定义其特定的视觉提示。

        参数：
            title: 商品标题
            description: 商品描述

        返回：
            视觉模型的提示文本
        """
        pass

    @abstractmethod
    def _get_log_prefix(self) -> str:
        """
        获取该链的日志前缀。

        返回：
            日志前缀字符串（例如："Description", "Summary"）
        """
        pass

    @abstractmethod
    def _get_fallback_text(self, title: str, description: Optional[str]) -> str:
        """
        当生成失败时获取回退文本。

        参数：
            title: 商品标题
            description: 商品描述

        返回：
            回退文本
        """
        pass

    async def generate_with_images(
            self,
            title: str,
            description: Optional[str],
            image_urls: List[str],
    ) -> str:
        """
        基于多个图片生成文本。

        参数：
            title: 商品标题
            description: 商品描述
            image_urls: 图片URL列表

        返回：
            生成的文本
        """
        if not image_urls:
            logger.warning(f"{self._get_log_prefix()}: 没有提供图片，将使用仅基于文本的方式生成")
            return await self.generate_text_only(title, description)

        try:
            # 获取视觉模型
            vision_model = self.llm_service.get_vision_model()

            # 并发加载所有图片为base64格式以提高性能
            tasks = [load_image_from_url(url) for url in image_urls]
            image_base64_list = await asyncio.gather(*tasks)

            # 构建内容：先文本后所有图片
            prompt_text = self._create_vision_prompt_text(title, description)
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

            # 调用模型
            response = await vision_model.ainvoke([message])
            result_text = response.content.strip()

            logger.info(
                f"{self._get_log_prefix()} 使用图片生成",
                extra={
                    "title": title[:50],
                    "image_count": len(image_urls),
                },
            )

            return result_text

        except Exception as e:
            logger.error(f"{self._get_log_prefix()} 使用图片生成失败: {e}", exc_info=True)
            # 回退到仅基于文本的生成
            return await self.generate_text_only(title, description)

    async def generate_text_only(self, title: str, description: Optional[str]) -> str:
        """
        仅使用文本输入生成文本（不使用图片）。

        参数：
            title: 商品标题
            description: 商品描述

        返回：
            生成的文本
        """
        try:
            # 获取聊天模型
            llm = self.llm_service.get_chat_model()

            # 创建链
            chain = self.text_prompt | llm

            # 运行链
            response = await chain.ainvoke(
                {
                    "title": title,
                    "description": description,
                },
            )

            result_text = response.content.strip()

            logger.info(
                f"{self._get_log_prefix()} 仅基于文本生成",
                extra={"title": title[:50]},
            )

            return result_text

        except Exception as e:
            logger.error(f"仅基于文本生成{self._get_log_prefix().lower()}时出错: {e}")
            # 返回回退文本
            return self._get_fallback_text(title, description)

    async def generate(
            self,
            title: str,
            description: Optional[str],
            image_urls: List[str],
    ) -> str:
        """
        生成文本（描述、摘要等）。

        参数：
            title: 商品标题
            description: 商品描述
            image_urls: 图片URL列表

        返回：
            生成文本
        """
        # 尝试首先使用图片生成
        if image_urls:
            return await self.generate_with_images(title, description, image_urls)
        else:
            return await self.generate_text_only(title, description)


# ==================== 具体实现 ====================


class DescriptionGenerationChain(BaseProductTextChain):
    """生成商品描述的chain."""

    _instance: Optional["DescriptionGenerationChain"] = None

    def _create_text_prompt(self) -> ChatPromptTemplate:
        """创建用于描述生成的提示模板（仅基于文本）。"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商文案撰写专家。你的任务是根据商品标题生成吸引人的商品描述。

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
                    "商品标题：{title}\n\n请生成商品描述：",
                ),
            ],
        )

    def _create_vision_prompt_text(self, title: str, description: Optional[str]) -> str:
        """创建带有图片的商品描述生成提示文本。"""
        desc_part = "" if description is None else f"\n商品描述：{description}"
        return f"""你是一个专业的电商文案撰写专家。请根据以下信息生成吸引人的商品描述：

商品标题：{title}{desc_part}

文案要求：
1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的视觉特点
2. 突出商品的核心卖点和特色
3. 使用生动、吸引人的语言
4. 符合电商规范，避免虚假宣传
5. 长度控制在100-300字之间
6. 结构清晰，易于阅读

请直接返回生成的商品描述文本。"""

    def _get_log_prefix(self) -> str:
        """获取商品描述生成的日志前缀。"""
        return "Description"

    def _get_fallback_text(self, title: str, description: Optional[str]) -> str:
        """获取回退描述文本。"""
        desc_part = "" if description is None else f" {description}"
        return f"这是一款优质商品：{title}{desc_part}。详情请查看商品图片和详细信息。"

    @classmethod
    def get_instance(cls) -> "DescriptionGenerationChain":
        """
        单例模式

        返回：
            DescriptionGenerationChain实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class SummaryGenerationChain(BaseProductTextChain):
    """生成商品摘要的chain。"""

    _instance: Optional["SummaryGenerationChain"] = None

    def _create_text_prompt(self) -> ChatPromptTemplate:
        """创建用于摘要生成的提示模板（仅基于文本）。"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商文案撰写专家。你的任务是根据商品标题和描述生成简洁的商品摘要。

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
                    "商品标题：{title}\n商品描述：{description}\n\n请生成商品摘要（最多200字）：",
                ),
            ],
        )

    def _create_vision_prompt_text(self, title: str, description: Optional[str]) -> str:
        """创建带有图片的商品摘要生成提示文本。"""
        desc_part = "" if description is None else f"\n商品描述：{description}"
        return f"""你是一个专业的电商文案撰写专家。请根据以下信息生成简洁的商品摘要：

商品标题：{title}{desc_part}

文案要求：
1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的核心特点
2. 高度概括商品的核心特点和卖点
3. 语言精炼，突出重点
4. 符合电商规范，避免虚假宣传
5. 长度严格控制在200字以内
6. 适合用作商品简介或短描述

请直接返回生成的商品摘要文本。"""

    def _get_log_prefix(self) -> str:
        """获取商品摘要生成的日志前缀。"""
        return "Summary"

    def _get_fallback_text(self, title: str, description: Optional[str]) -> str:
        """获取回退摘要文本。"""
        desc_part = "" if description is None else f" {description}"
        return f"商品：{title}-{desc_part}"

    @classmethod
    def get_instance(cls) -> "SummaryGenerationChain":
        """
        单例模式 SummaryGenerationChain.

        返回：
            SummaryGenerationChain实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

"""LangChain chain for product description generation."""
import asyncio
import base64
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_llm_service
from app.utils.image_util import load_image_from_url
from app.utils.logger import app_logger as logger


class DescriptionGenerationChain:
    """Chain for generating product descriptions."""

    _instance: Optional["DescriptionGenerationChain"] = None

    def __init__(self):
        """Initialize description generation chain."""
        self.llm_service = get_llm_service()

        # 提示词
        self.text_prompt = ChatPromptTemplate.from_messages(
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

    @classmethod
    def get_instance(cls) -> "DescriptionGenerationChain":
        """
        无状态 chain，单例 DescriptionGenerationChain.

        Returns:
            DescriptionGenerationChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_with_images(
            self,
            title: str,
            category: str,
            image_urls: list[str],
    ) -> str:
        """
        根据多张图片获取商品描述（支持多图输入）。

        Args:
            title: 商品标题
            category: 商品的类目
            image_urls: 商品图片的 URL 列表（建议 1~5 张）

        Returns:
            商品描述
        """
        if not image_urls:
            logger.warning("没有图片，回退到纯文本生成！")
            return await self.generate_text_only(title, category)

        try:
            # 获取视觉模型
            vision_model = self.llm_service.get_vision_model()

            # 并发加载所有图片为 base64（提升性能）
            tasks = [load_image_from_url(url) for url in image_urls]
            image_base64_list = await asyncio.gather(*tasks)

            # 构建 content：先文本，再所有图片
            content = [
                {
                    "type": "text",
                    "text": f"""你是一个专业的电商文案撰写专家。请根据以下信息生成吸引人的商品描述：

    商品标题：{title}
    商品类目：{category}

    文案要求：
    1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的视觉特点
    2. 突出商品的核心卖点和特色
    3. 使用生动、吸引人的语言
    4. 符合电商规范，避免虚假宣传
    5. 长度控制在100-300字之间
    6. 结构清晰，易于阅读

    请直接返回生成的商品描述文本。""",
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
            description = response.content.strip()

            logger.info(
                "使用多张图片生成商品描述",
                extra={
                    "title": title[:50],
                    "image_count": len(image_urls),
                },
            )

            return description

        except Exception as e:
            logger.error(f"使用图片生成商品描述失败: {e}", exc_info=True)
            # 回退到纯文本生成
            return await self.generate_text_only(title, category)

    async def generate_text_only(self, title: str, category: str) -> str:
        """
        不使用图片仅凭文本分析获取商品描述.

        Args:
            title: 商品标题
            category: 商品的类目

        Returns:
            商品描述
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
                    "category": category,
                },
            )

            description = response.content.strip()

            logger.info(
                "Description generated (text-only)",
                extra={"title": title[:50]},
            )

            return description

        except Exception as e:
            logger.error(f"Error generating text-only description: {e}")
            # 返回备用描述
            return f"这是一款来自{category}类目的优质商品：{title}。详情请查看商品图片和详细信息。"

    async def generate(
        self,
        title: str,
        category: str,
        image_urls: list[str],
    ) -> str:
        """
        获取商品描述.

        Args:
            title: 商品标题
            category: 商品的类目
            image_urls: 商品图片的 URL 列表

        Returns:
            商品描述
        """
        # 先尝试使用图片生成描述
        if image_urls:
            return await self.generate_with_images(title, category, image_urls)
        else:
            return await self.generate_text_only(title, category)

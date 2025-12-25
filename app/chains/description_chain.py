"""LangChain chain for product description generation."""

import base64
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_llm_service
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

    async def _load_image_from_url(self, url: str) -> str:
        """
        Load image from URL and convert to base64.

        Args:
            url: Image URL

        Returns:
            Base64 encoded image
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")

    async def generate_with_images(
        self,
        title: str,
        category: str,
        image_urls: list[str],
    ) -> str:
        """
        根据图片获取商品描述.

        Args:
            title: 商品标题
            category: 商品的类目
            image_urls: 商品图片的 URL 列表

        Returns:
            商品描述
        """
        try:
            # 获取视觉模型
            vision_model = self.llm_service.get_vision_model()

            # 准备图片内容（使用第一张图片）
            image_url = image_urls[0]
            image_base64 = await self._load_image_from_url(image_url)

            # 提示词
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""你是一个专业的电商文案撰写专家。请根据以下信息生成吸引人的商品描述：

商品标题：{title}
商品类目：{category}

文案要求：
1. 仔细分析商品图片，提取商品的视觉特点
2. 突出商品的核心卖点和特色
3. 使用生动、吸引人的语言
4. 符合电商规范，避免虚假宣传
5. 长度控制在100-300字之间
6. 结构清晰，易于阅读

请直接返回生成的商品描述文本。""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            )

            # 运行视觉模型
            response = await vision_model.ainvoke([message])
            description = response.content.strip()

            logger.info(
                "Description generated with images",
                extra={
                    "title": title[:50],
                    "image_count": len(image_urls),
                },
            )

            return description

        except Exception as e:
            logger.error(f"Error generating description with images: {e}")
            # Fall back to text-only generation
            return await self.generate_text_only(title, category)

    async def generate_text_only(self, title: str, category: str) -> str:
        """
        不使用图片分析获取商品描述.

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

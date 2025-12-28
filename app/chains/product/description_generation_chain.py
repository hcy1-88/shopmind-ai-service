"""商品描述生成链 - 基于标题和图片生成营销性商品描述."""

import asyncio
from typing import Optional, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_llm_service
from app.utils.image_util import load_image_from_url
from app.utils.logger import app_logger as logger


class DescriptionGenerationChain:
    """
    商品描述生成链.

    功能：根据商品标题和图片生成详细的营销性商品描述。
    输入：标题 + 图片URL列表
    输出：商品描述文本
    """

    _instance: Optional["DescriptionGenerationChain"] = None

    def __init__(self):
        """初始化商品描述生成链."""
        self.llm_service = get_llm_service()

    def _create_text_prompt(self) -> ChatPromptTemplate:
        """创建文本模式的提示模板（无图片时使用）."""
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

    def _create_vision_prompt_text(self, title: str) -> str:
        """创建视觉模式的提示文本（有图片时使用）."""
        return f"""你是一个专业的电商文案撰写专家。请根据以下信息生成吸引人的商品描述：

商品标题：{title}

文案要求：
1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的视觉特点
2. 突出商品的核心卖点和特色
3. 使用生动、吸引人的语言
4. 符合电商规范，避免虚假宣传
5. 长度控制在100-300字之间
6. 结构清晰，易于阅读

请直接返回生成的商品描述文本，不要包含任何额外说明。"""

    async def generate_with_images(
        self,
        title: str,
        image_urls: List[str],
    ) -> str:
        """
        基于标题和图片生成商品描述.

        Args:
            title: 商品标题
            image_urls: 图片URL列表

        Returns:
            生成的商品描述
        """
        if not image_urls:
            logger.warning("Description: 没有提供图片，将使用纯文本模式生成")
            return await self.generate_text_only(title)

        try:
            # 获取视觉模型
            vision_model = self.llm_service.get_vision_model()

            # 并发加载所有图片
            tasks = [load_image_from_url(url) for url in image_urls]
            image_base64_list = await asyncio.gather(*tasks)

            # 构建消息内容：文本 + 所有图片
            prompt_text = self._create_vision_prompt_text(title)
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

            # 调用视觉模型
            response = await vision_model.ainvoke([message])
            result_text = response.content.strip()

            logger.info(
                "商品描述生成完成（使用图片）",
                extra={
                    "title": title[:50],
                    "image_count": len(image_urls),
                    "description_length": len(result_text),
                },
            )

            return result_text

        except Exception as e:
            logger.error(f"使用图片生成商品描述失败: {e}", exc_info=True)
            # 回退到纯文本模式
            return await self.generate_text_only(title)

    async def generate_text_only(self, title: str) -> str:
        """
        仅基于标题生成商品描述（无图片）.

        Args:
            title: 商品标题

        Returns:
            生成的商品描述
        """
        try:
            # 获取聊天模型
            llm = self.llm_service.get_chat_model()

            # 创建链
            chain = self._create_text_prompt() | llm

            # 运行链
            response = await chain.ainvoke({"title": title})
            result_text = response.content.strip()

            logger.info(
                "商品描述生成完成（纯文本模式）",
                extra={
                    "title": title[:50],
                    "description_length": len(result_text),
                },
            )

            return result_text

        except Exception as e:
            logger.error(f"纯文本模式生成商品描述失败: {e}", exc_info=True)
            # 返回回退文本
            return f"这是一款优质商品：{title}。详情请查看商品图片和详细信息。"

    async def generate(
        self,
        title: str,
        image_urls: List[str],
    ) -> str:
        """
        生成商品描述（主入口）.

        Args:
            title: 商品标题
            image_urls: 图片URL列表

        Returns:
            生成的商品描述
        """
        if image_urls:
            return await self.generate_with_images(title, image_urls)
        else:
            return await self.generate_text_only(title)

    @classmethod
    def get_instance(cls) -> "DescriptionGenerationChain":
        """
        获取单例实例.

        Returns:
            DescriptionGenerationChain 实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
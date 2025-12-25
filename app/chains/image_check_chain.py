"""LangChain chain for image compliance checking using vision models."""

import base64
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.schemas.image_check import ImageCheckResponse
from app.services.llm_service import get_llm_service
from app.utils.logger import app_logger as logger


class ImageCheckChain:
    """Chain for checking product image compliance using vision models."""

    _instance: Optional["ImageCheckChain"] = None

    def __init__(self):
        """Initialize image check chain."""
        self.llm_service = get_llm_service()

    async def _load_image_from_url(self, url: str) -> str:
        """
        将图片 url 转换为 base64.

        Args:
            url: Image URL

        Returns:
            Base64 encoded image
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")

    @classmethod
    def get_instance(cls) -> "ImageCheckChain":
        """
        无状态 chain，单例 ImageCheckChain.

        Returns:
            ImageCheckChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _is_base64_image(self, image_data: str) -> bool:
        """
        Check if the image data is base64 encoded.

        Args:
            image_data: Image data string

        Returns:
            True if base64, False otherwise
        """
        return not image_data.startswith(("http://", "https://"))

    async def check(self, image_url: str) -> dict:
        """
        检查图片是否合规.

        Args:
            image_url: Image URL 或 base64 encoded image

        Returns:
            Dictionary with validation result
        """
        try:
            # 这里要获取视觉模型
            vision_model = self.llm_service.get_vision_model()

            # 准备图片
            if self._is_base64_image(image_url):
                # Already base64
                image_base64 = image_url
            else:
                # 转 base64 from URL
                image_base64 = await self._load_image_from_url(image_url)

            # Create message with image
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """你是一个专业的电商平台图片审核专家。请审核这张商品图片是否符合平台规范。

审核标准：
1. 不能包含色情、暴力、血腥内容
2. 不能包含政治敏感内容
3. 不能包含虚假宣传或误导性内容
4. 不能包含恶意广告或水印
5. 图片应清晰、真实地展示商品

请以JSON格式返回审核结果：
{
    "valid": true/false,
    "reason": "不合规原因（如果不合规）"
}

如果图片合规，reason设为null。""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            )
            # 运行和解析输出
            response = await vision_model.ainvoke([message])
            output_parser = JsonOutputParser(pydantic_object=ImageCheckResponse)
            result = output_parser.parse(response.content)

            logger.info(
                "Image check completed",
                extra={
                    "image_url": image_url[:100],
                    "valid": result.get("valid"),
                },
            )

            return result

        except httpx.HTTPError as e:
            logger.error(f"Failed to load image from URL: {e}")
            return {
                "valid": False,
                "reason": f"无法加载图片: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error in image check chain: {e}")
            return {
                "valid": False,
                "reason": f"审核过程出现错误: {str(e)}",
            }

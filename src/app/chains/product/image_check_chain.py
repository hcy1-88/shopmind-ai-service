"""LangChain chain for image compliance checking using vision models."""

from typing import Optional

import httpx
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser

from src.app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator
from src.app.schemas import ImageCheckRequest, ImageCheckResponse
from src.app.utils.logger import app_logger as logger


class ImageCheckChain(VisionAwareAIGenerator[ImageCheckRequest, ImageCheckResponse]):
    """Chain for checking product image compliance using vision models."""

    _instance: Optional["ImageCheckChain"] = None

    def _has_images(self, input_data: ImageCheckRequest) -> bool:
        """图片检查必须有图片"""
        return bool(input_data.image_url)

    def _get_vision_system_prompt(self) -> str:
        """视觉模式的系统提示"""
        return """你是一个专业的电商平台图片审核专家。请审核这张商品图片是否符合平台规范。

审核标准：
1. 不能包含色情、暴力、血腥内容
2. 不能包含政治敏感而引起对立的内容，但可使用"政府补贴"、"政府补助"、"消费补贴"、"以旧换新补贴"等营销表述
3. 不能包含虚假宣传或误导性内容
4. 不能包含恶意广告或水印
5. 图片应清晰、真实地展示商品"""


    def _build_vision_human_message_base_text(self, input_data: ImageCheckRequest) -> str:
        """构建视觉 HumanMessage 中的文本部分"""
        return ""

    def _get_output_parser(self) -> BaseOutputParser:
        """返回输出解析器"""
        return PydanticOutputParser(pydantic_object=ImageCheckResponse)

    def _extract_image_urls(self, input_data: ImageCheckRequest) -> list[str]:
        """从 ImageCheckRequest 中提取图片 URL（单个字符串转为列表）"""
        if input_data.image_url:
            return [input_data.image_url]
        return []

    async def generate(self, input_data: ImageCheckRequest) -> ImageCheckResponse:
        """重写 generate 方法，添加异常处理"""
        try:
            return await super().generate(input_data)
        except httpx.HTTPError as e:
            logger.error(f"Failed to load image from URL: {e}", exc_info=True)
            return ImageCheckResponse(
                valid=False,
                reason=f"无法加载图片: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error in image check chain: {e}", exc_info=True)
            return ImageCheckResponse(
                valid=False,
                reason=f"审核过程出现错误: {str(e)}",
            )

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

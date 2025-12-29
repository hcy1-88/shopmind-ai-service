"""
@File       : description_generator_chain.py
@Description:

@Time       : 2025/12/28 20:47
@Author     : hcy18
"""
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser

from src.app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator
from src.app.schemas import DescriptionGenerateRequest, DescriptionGenerateResponse


class DescriptionGenerationChain(VisionAwareAIGenerator[DescriptionGenerateRequest, DescriptionGenerateResponse]):
    """生成商品描述"""

    _instance: Optional["DescriptionGenerationChain"] = None

    def _has_images(self, input_data: DescriptionGenerateRequest) -> bool:
        return bool(input_data.image_urls)

    def _get_text_only_system_prompt(self) -> str:
        """无图片时，生成的系统提示词"""
        return  """你是一个专业的电商文案撰写专家。你的任务是根据商品标题生成吸引人的商品描述。

            文案要求：
            1. 突出商品的核心卖点和特色
            2. 使用生动、吸引人的语言
            3. 符合电商规范，避免虚假宣传
            4. 长度控制在100-300字之间
            5. 结构清晰，易于阅读
            6. 适当使用分段和符号增强可读性
            
            请直接返回生成的商品描述文本，不要包含任何额外的说明或格式。
            """

    def _get_vision_system_prompt(self) -> str:
        """有图片时，生成的系统提示词"""
        return """你是一个专业的电商文案撰写专家。请根据用户输入的商品标题和图片，按以下要求生成吸引人的商品描述：
            文案要求：
            1. 仔细分析所有提供的商品图片（包括主图、细节、场景等），提取商品的视觉特点
            2. 突出商品的核心卖点和特色
            3. 使用生动、吸引人的语言
            4. 符合电商规范，避免虚假宣传
            5. 长度控制在100-300字之间
            6. 结构清晰，易于阅读
            
            请直接返回生成的商品描述文本，不要包含任何额外说明。
            """

    def _build_text_only_human_message_content(self, input_data: DescriptionGenerateRequest) -> str:
        return f"商品标题：{input_data.title}"

    def _build_vision_human_message_base_text(self, input_data: DescriptionGenerateRequest) -> str:
        return f"商品标题：{input_data.title}"

    def _get_output_parser(self) -> BaseOutputParser:
        return PydanticOutputParser(pydantic_object=DescriptionGenerateResponse)


    @classmethod
    def get_instance(cls) -> "DescriptionGenerationChain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
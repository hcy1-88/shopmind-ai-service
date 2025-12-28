"""
@File       : summary_generator_chain.py
@Description:

@Time       : 2025/12/28 21:02
@Author     : hcy18
"""
from typing import Optional
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator, InputType
from app.schemas import SummaryGenerateRequest, SummaryGenerateResponse


class SummaryGenerationChain(VisionAwareAIGenerator[SummaryGenerateRequest, SummaryGenerateResponse]):
    """商品摘要生成链。只根据 商品标题 和 商品描述，无需图片"""

    _instance: Optional["SummaryGenerationChain"] = None

    def _has_images(self, input_data: InputType) -> bool:
        return False

    def _get_text_only_system_prompt(self) -> str:
        return """你是一个专业的电商数据分析和文案专家。你的任务是根据提供的商品信息生成简洁精炼的商品摘要。

            摘要要求：
            - 高度概括商品的核心特点和卖点
            - 语言精炼，突出重点
            - 符合电商规范，避免虚假宣传
            - 长度严格控制在200字以内
            - 适合用作商品简介或短描述
            
            请直接返回生成的商品摘要文本，不要包含任何额外的说明或格式。"""


    def _build_text_only_human_message_content(self, input_data: SummaryGenerateRequest) -> str:
        return f"商品信息如下：\n标题：{input_data.title}\n描述：{input_data.description}\n\n请生成商品摘要："


    def _get_vision_system_prompt(self) -> str:
        pass

    def _build_vision_human_message_base_text(self, input_data: InputType) -> str:
        pass


    def _get_output_parser(self) -> BaseOutputParser:
        return PydanticOutputParser(pydantic_object=SummaryGenerateResponse)

    @classmethod
    def get_instance(cls) -> "SummaryGenerationChain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
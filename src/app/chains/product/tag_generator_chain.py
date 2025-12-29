"""
@File       : tag_generator_chain.py
@Description:

@Time       : 2025/12/28 19:05
@Author     : hcy18
"""
from typing import Optional
from langchain_core.output_parsers import PydanticOutputParser
from src.app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator
from src.app.schemas.product_tag import GenerateTagsRequest, GenerateTagsResponse


class ProductTagGenChain(VisionAwareAIGenerator[GenerateTagsRequest, GenerateTagsResponse]):
    """生成商品标签的 chain"""

    _instance: Optional["ProductTagGenChain"] = None


    def _has_images(self, input_data: GenerateTagsRequest) -> bool:
        return bool(input_data.image_urls)


    def _get_text_only_system_prompt(self) -> str:
        """系统提示词：无图片时，仅文本生成结果"""
        return """
                你是一个专业的电商智能标签生成助手。请根据用户给出的商品标题和商品描述，生成一组简洁、有区分度、对用户有决策帮助的商品标签。

                要求：
                1. 生成 3 到 6 个标签，覆盖以下可能的维度（按优先级）：
                   - 商品核心功能或显著属性（如“无线充电”、“防水”、“适合敏感肌”）
                   - 用户评价或口碑特征（如“性价比高”、“质量好”、“口碑好”）
                   - 运营状态（仅在描述中明确提及才使用，如“新品”、“限时特价”、“包邮”）
                   - 避免主观臆断；所有标签必须有文本依据或合理推断。
                2. 每个标签名称应为 2-4 个汉字或简短词组，简洁醒目。
                3. 为每个标签推荐一个适合 Web UI 展示的颜色，使用 **6 位十六进制颜色码（格式：#RRGGBB）**，例如 #FF4500、#32CD32、#1E90FF。颜色应与标签语义匹配：
                   - 正向/促销类（如“热销”、“特价”）→ 暖色（红、橙）
                   - 新品/健康/安全类（如“新品”、“适合敏感肌”）→ 冷色或绿色（绿、蓝、青）
                   - 口碑/品质类（如“质量好”、“推荐”）→ 中性或品牌色（紫、深蓝、灰）

                """

    def _get_vision_system_prompt(self) -> str:
        """系统提示词：有图片时，结合图片生成结果"""
        return """
                你是一个专业的电商智能标签生成助手。请根据用户给出的商品标题、商品描述 和 商品图片，生成一组简洁、有区分度、对用户有决策帮助的商品标签。

                要求：
                1. 阅读 商品标题 和 商品描述，分析商品相关的图片，寻找商品的卖点和吸引人眼球的特征
                1. 生成 3 到 6 个标签，覆盖以下可能的维度（按优先级）：
                   - 商品核心功能或显著属性（如“无线充电”、“防水”、“适合敏感肌”）
                   - 用户评价或口碑特征（如“性价比高”、“质量好”、“口碑好”）
                   - 运营状态（仅在描述中明确提及才使用，如“新品”、“限时特价”、“包邮”）
                   - 避免主观臆断；所有标签必须有文本依据或合理推断。
                2. 每个标签名称应为 2-4 个汉字或简短词组，简洁醒目。
                3. 为每个标签推荐一个适合 Web UI 展示的颜色，使用 **6 位十六进制颜色码（格式：#RRGGBB）**，例如 #FF4500、#32CD32、#1E90FF。颜色应与标签语义匹配：
                   - 正向/促销类（如“热销”、“特价”）→ 暖色（红、橙）
                   - 新品/健康/安全类（如“新品”、“适合敏感肌”）→ 冷色或绿色（绿、蓝、青）
                   - 口碑/品质类（如“质量好”、“推荐”）→ 中性或品牌色（紫、深蓝、灰）

                """


    def _build_text_only_human_message_content(self, input_data: GenerateTagsRequest) -> str:
        return f"商品标题：{input_data.title}\n商品描述：{input_data.description}"


    def _build_vision_human_message_base_text(self, input_data: GenerateTagsRequest) -> str:
        return f"商品标题：{input_data.title}\n商品描述：{input_data.description}"


    def _get_output_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=GenerateTagsResponse)

    @classmethod
    def get_instance(cls) -> "ProductTagGenChain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
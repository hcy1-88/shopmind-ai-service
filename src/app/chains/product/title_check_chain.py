"""LangChain chain for title compliance checking."""

from typing import Optional

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser

from app.chains.product.base_ai_generator_chain import VisionAwareAIGenerator
from app.schemas import TitleCheckRequest, TitleCheckResponse
from app.utils.logger import app_logger as logger


class TitleCheckChain(VisionAwareAIGenerator[TitleCheckRequest, TitleCheckResponse]):
    """商品标题检查合规性的 chain."""

    _instance: Optional["TitleCheckChain"] = None

    def _has_images(self, input_data: TitleCheckRequest) -> bool:
        """标题检查不需要图片"""
        return False

    def _get_text_only_system_prompt(self) -> str:
        """纯文本模式的系统提示"""
        return """你是一个专业的电商平台内容审核专家。你的任务是审核商品标题是否符合平台规范。\n

        审核标准：
        1. **禁止违规内容**：不得包含色情、暴力、政治敏感、违法不良信息。
        2. **禁止无依据的绝对化用语**：如"全网最低价"、"史上最低"、"100%有效"、"国家级"、"最高级"、"第一品牌"等违反《广告法》的极限词。
        3. **允许真实促销信息，可略微夸张吸引眼球**：
           - 可使用"政府补贴"、"政府补助"、"消费补贴"、"以旧换新补贴"等表述，前提是该类补贴在当前国家或地方政府政策中真实存在（如2024-2025年消费品以旧换新行动）。
           - 可使用"百亿补贴"、"平台补贴"、"限时优惠"、"直降"等电商平台常见营销术语，但不得虚构补贴金额或来源。
        4. **禁止诱导点击**：如"点击领取"、"必抢"、"手慢无"等强诱导性话术。
        5. **商品信息需基本真实**：对于可判断出信息真实的商品，应给予通过；如果存疑，应倾向于通过；如果能直接判断出商品信息虚假，应判断商品标题不通过。
"""


    def _build_text_only_human_message_content(self, input_data: TitleCheckRequest) -> str:
        """构建纯文本 HumanMessage 的 content"""
        return f"请审核这个商品标题：{input_data.title}"


    def _get_output_parser(self) -> BaseOutputParser:
        """返回输出解析器"""
        return PydanticOutputParser(pydantic_object=TitleCheckResponse)


    async def generate(self, input_data: TitleCheckRequest) -> TitleCheckResponse:
        """重写 generate 方法，添加异常处理"""
        try:
            return await super().generate(input_data)
        except Exception as e:
            logger.error(f"Error in title check chain: {e}", exc_info=True)
            # 安全输出
            return TitleCheckResponse(
                valid=False,
                reason=f"审核过程出现错误: {str(e)}",
                suggestions=["请稍后重试"],
            )

    @classmethod
    def get_instance(cls) -> "TitleCheckChain":
        """
        无状态 chain，单例 TitleCheckChain.

        Returns:
            TitleCheckChain instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

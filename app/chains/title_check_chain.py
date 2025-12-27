"""LangChain chain for title compliance checking."""

from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.product_title_check import TitleCheckResponse
from app.services.llm_service import get_llm_service
from app.utils.logger import app_logger as logger


class TitleCheckChain:
    """商品标题检查合规性的 chain."""

    _instance: Optional["TitleCheckChain"] = None

    def __init__(self):
        """Initialize title check chain."""
        self.llm_service = get_llm_service()

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商平台内容审核专家。你的任务是审核商品标题是否符合平台规范。

审核标准：
1. **禁止违规内容**：不得包含色情、暴力、政治敏感、违法不良信息。
2. **禁止无依据的绝对化用语**：如"全网最低价"、"史上最低"、"100%有效"、"国家级"、"最高级"、"第一品牌"等违反《广告法》的极限词。
3. **允许真实促销信息，可略微夸张吸引眼球**：
   - 可使用"政府补贴"、"政府补助"、"消费补贴"、"以旧换新补贴"等表述，前提是该类补贴在当前国家或地方政府政策中真实存在（如2024-2025年消费品以旧换新行动）。
   - 可使用"百亿补贴"、"平台补贴"、"限时优惠"、"直降"等电商平台常见营销术语，但不得虚构补贴金额或来源。
4. **禁止诱导点击**：如"点击领取"、"必抢"、"手慢无"等强诱导性话术。
5. **商品信息需基本真实**：品牌、型号、核心配置（如"M4芯片"、"13英寸"）必须准确。

请以JSON格式返回审核结果：
{{
    "valid": true/false,
    "reason": "不合规原因（如果不合规）",
    "suggestions": ["建议1", "建议2"]
}}

如果标题合规，reason设为null，suggestions设为空数组。""",
                ),
                ("human", "请审核这个商品标题：{title}"),
            ],
        )

        # Output parser with Pydantic model for validation
        self.output_parser = JsonOutputParser(pydantic_object=TitleCheckResponse)

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

    async def check(self, title: str) -> dict:
        """
        Check if title is compliant.

        Args:
            title: Product title to check

        Returns:
            Dictionary with validation result
        """
        try:
            # Get chat model
            llm = self.llm_service.get_chat_model()

            # Create chain
            chain = self.prompt | llm | self.output_parser

            # Run chain
            result = await chain.ainvoke({"title": title})

            logger.info(
                "Title check completed",
                extra={
                    "title": title[:50],
                    "valid": result.get("valid"),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in title check chain: {e}")
            # 安全输出
            return {
                "valid": False,
                "reason": f"审核过程出现错误: {str(e)}",
                "suggestions": ["请稍后重试"],
            }

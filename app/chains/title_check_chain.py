"""LangChain chain for title compliance checking."""

from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.title_check import TitleCheckResponse
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
1. 不能包含违规词汇（如色情、暴力、政治敏感等）
2. 不能有虚假宣传（如"全网最低价"、"100%有效"等绝对化用语）
3. 不能违反广告法（如"国家级"、"最高级"、"第一"等极限词）
4. 不能包含诱导性词汇（如"点击有礼"、"必买"等）
5. 标题应真实准确描述商品

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

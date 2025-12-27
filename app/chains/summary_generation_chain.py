"""商品摘要生成链 - 基于标题和描述生成摘要."""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_llm_service
from app.utils.logger import app_logger as logger


class SummaryGenerationChain:
    """
    商品摘要生成链.

    功能：根据商品标题和描述生成简洁摘要。
    输入：title (标题) + description (描述)
    输出：summary (摘要字符串)
    """

    _instance: Optional["SummaryGenerationChain"] = None

    def __init__(self):
        """初始化商品摘要生成链."""
        self.llm_service = get_llm_service()
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """创建摘要生成的提示模板."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的电商数据分析和文案专家。你的任务是根据提供的商品信息生成简洁精炼的商品摘要。

摘要要求：
- 高度概括商品的核心特点和卖点
- 语言精炼，突出重点
- 符合电商规范，避免虚假宣传
- 长度严格控制在200字以内
- 适合用作商品简介或短描述

请直接返回生成的商品摘要文本，不要包含任何额外的说明或格式。""",
                ),
                (
                    "human",
                    "商品信息如下：\n标题：{title}\n描述：{description}\n\n请生成商品摘要：",
                ),
            ],
        )

    async def generate(
        self,
        title: str,
        description: str,
    ) -> str:
        """
        生成商品摘要.

        Args:
            title: 商品标题
            description: 商品描述

        Returns:
            商品摘要字符串
        """
        try:
            # 获取聊天模型
            llm = self.llm_service.get_chat_model()

            # 创建链：prompt | llm
            chain = self.prompt | llm

            # 运行链，获取响应
            response = await chain.ainvoke(
                {
                    "title": title,
                    "description": description,
                }
            )
            result_text = response.content.strip()

            logger.info(
                "商品摘要生成完成",
                extra={
                    "title": title[:50],
                    "summary_length": len(result_text),
                },
            )

            return result_text

        except Exception as e:
            logger.error(f"生成商品摘要失败: {e}", exc_info=True)
            # 返回回退结果
            return f"商品：{title}。{description[:50]}..."

    @classmethod
    def get_instance(cls) -> "SummaryGenerationChain":
        """
        获取单例实例.

        Returns:
            SummaryGenerationChain 实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
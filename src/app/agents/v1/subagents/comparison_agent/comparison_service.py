"""
@File       : comparison_service.py
@Description: 比较服务 - 使用 create_agent + 工具挂载模式处理商品比较

@Time       : 2026/4/3
@Author     : hcy18
"""
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from app.agents.v1.schema import ShopmindAssistantContext
from app.services import llm_service
from app.tools.chat_tool import get_product_detail_for_comparison
from app.utils.logger import app_logger as logger


COMPARISON_PROMPT = """你是一个专业的电商导购助手，正在为用户对比商品。

## 你的任务
当用户要求比较商品时，你应该首先调用 get_product_detail 工具获取商品的详细信息，然后基于这些信息生成对比文案和购买建议。

## 工具使用
- get_product_detail: 获取商品详情，需要传入商品ID列表
  例如：get_product_detail(product_ids=[1001, 1002])

## 对比维度
请从以下维度进行对比（根据实际商品信息选择适用的维度）：
1. 价格
2. 品牌
3. 核心参数/规格
4. 适用场景
5. 用户评价/口碑
6. 售后保障

## 输出格式
请按以下格式输出：

## 商品对比
| 维度 | 商品1名称 | 商品2名称 | ... |
|------|----------|----------|-----|
| 价格 | xxx | xxx | ... |
| ... | ... | ... | ... |

## 购买建议
基于用户需求，给出明确的购买建议。如果用户没有明确需求，可以从性价比、适用场景等角度给出建议。

## 注意事项
- 只对比商品详情中确实存在的信息，不要编造
- 如果某个维度信息不全，在对比表中标注"未提供"
- 购买建议要具体、有针对性，不要泛泛而谈
- 务必先调用工具获取商品详情再生成对比
- 你的客户是商场的普通消费者，所以回复中不要带 "商品ID"、某某ID 这样的字眼 
"""


class ComparisonService:
    """
    比较服务（单例模式）

    内部管理 ReAct Agent 实例，避免每次调用创建开销
    """

    _instance: Optional["ComparisonService"] = None

    def __init__(self):
        """初始化比较服务"""
        self.llm = llm_service.get_llm_service().get_chat_model()
        self.agent = None

    @classmethod
    def get_instance(cls) -> "ComparisonService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def build_agent(self):
        """构建比较 Agent"""
        tools = [get_product_detail_for_comparison]
        logger.info(f"[ComparisonService] 初始化中，工具数量: {len(tools)}")

        self.agent = create_agent(
            self.llm,
            tools=tools,
            system_prompt=COMPARISON_PROMPT,
        )

        logger.info("[ComparisonService] 初始化完成")

    async def compare(self, query: str, product_ids: list[int], thread_id: str) -> str:
        """
        处理商品比较

        Args:
            query: 用户查询（如"比较这两款洗发水"）
            product_ids: 待比较的商品ID列表
            thread_id: 用于 checkpoint 的线程ID

        Returns:
            Agent 生成的对比文案和购买建议
        """
        if not self.agent:
            self.build_agent()

        try:
            logger.info(f"[ComparisonAgent] thread_id: {thread_id}, 处理商品比较: product_ids={product_ids}")

            # 构建消息
            user_prompt = f"""用户请求：{query}
待比较商品ID：{product_ids}

请先调用 get_product_detail 工具获取这些商品的详情，然后生成对比文案和购买建议。"""

            messages = [
                SystemMessage(content=COMPARISON_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            result = await self.agent.ainvoke({"messages": messages})
            resp = result["messages"][-1].content
            logger.info(f"[ComparisonService] 完成: {resp[:50]}...")
            return resp

        except Exception as e:
            logger.error(f"[ComparisonService] 失败: {e}", exc_info=True)
            return f"抱歉，处理商品比较时遇到了问题: {str(e)}"


def get_comparison_service() -> ComparisonService:
    """获取比较服务单例"""
    return ComparisonService.get_instance()

"""
@File       : chitchat_agent.py
@Description: 闲聊服务 - 使用 ReAct Agent 处理闲聊、天气查询、联网搜索等

@Time       : 2026/3/23
@Author     : hcy18
"""
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langgraph.types import RunnableConfig

from app.agents.v1.schema import ShopmindAssistantContext
from app.agents.v1.utils import build_history_context
from app.services import llm_service
from app.tools.chat_tool import tavily_search, get_current_weather, get_forecast_weather
from app.utils.logger import app_logger as logger


CHITCHAT_PROMPT = """你是一个友好、热情的导购助手，名字叫「小购」。

## 你的任务
负责和用户闲聊聊天，不过要能在恰当的时机，巧妙地把话题拉回到购物上，因为你是导购助手，期望达成平台交易。

## 你的能力
1. **联网搜索**：可以搜索网络上最新信息来回答问题
2. **天气查询**：可以查询城市实时天气和天气预报
3. **闲聊陪伴**：可以和用户进行友好的闲聊

## 回答风格
- 语言自然、亲切、简洁
- 结合对话历史上下文理解用户意图，重要信息不清楚就要问
- 主动使用工具获取实时信息（天气、新闻等）
- 如果用户询问实时信息，主动调用相应工具

## 工具使用
- tavily_search: 联网搜索，用于获取最新信息、新闻等
- get_current_weather: 查询城市实时天气
- get_forecast_weather: 查询城市天气预报（支持3-30天）

## 注意事项
- 不要编造事实，不确定时主动搜索验证
- 如果用户询问你不知道的信息，建议用户使用工具搜索
"""


class ChitChatService:
    """
    闲聊服务（单例模式）

    内部管理 ReAct Agent 实例，避免每次调用创建开销
    注意：不使用 checkpointer，对话历史通过构建输入注入
    """

    _instance: Optional["ChitChatService"] = None

    def __init__(self):
        """初始化闲聊服务"""
        # 获取 LLM
        self.llm = llm_service.get_llm_service().get_chat_model()
        self.agent = None


    @classmethod
    def get_instance(cls) -> "ChitChatService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def chat(self, query: str, messages: list[BaseMessage], thread_id: str) -> str:
        """
        处理闲聊查询

        Args:
            query: 用户查询（sub_query，重写后的子问题）
            messages: 对话历史（已包含原始 HumanMessage）
            thread_id: 用于 checkpoint 的线程 ID

        Returns:
            Agent 生成的回答
        """
        try:
            logger.info(f"[ChitChatService] 处理闲聊: {query[:50]}...")

            # messages 已包含原始 HumanMessage，直接使用
            # query（sub_query）是 LLM 重写后的版本，作为当前轮次的追加消息
            # 但注意：原始 query 已存在于 messages 中，不能重复追加
            # 因此直接传入 messages，agent 会自动处理

            result = await self.agent.ainvoke({"messages": messages})
            resp = result["messages"][-1].content
            logger.info(f"[ChitChatService] 完成: {resp[:50]}...")
            return resp

        except Exception as e:
            logger.error(f"[ChitChatService] 失败: {e}", exc_info=True)
            return f"抱歉，处理您的问题时遇到了问题: {str(e)}"

    def build_chitchat_agent(self):
        # 工具列表
        tools = [tavily_search, get_current_weather, get_forecast_weather]
        logger.info(f"[ChitChatService] 初始化中，工具数量: {len(tools)}")

        # 创建 ReAct Agent（不使用 checkpointer，对话历史通过输入消息注入）
        self.agent = create_agent(
            self.llm,
            tools=tools,
            system_prompt=CHITCHAT_PROMPT,
        )

        logger.info("[ChitChatService] 初始化完成")


def get_chitchat_service() -> ChitChatService:
    """获取闲聊服务单例"""
    return ChitChatService.get_instance()

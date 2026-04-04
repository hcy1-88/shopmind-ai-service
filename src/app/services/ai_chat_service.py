"""AI 对话服务 - 生产级实现."""

import json
from typing import AsyncGenerator, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agents.v1.nodes.aggregator_node import FINAL_ANSWER_TAG
from app.agents.v1.schema import ShopmindAssistantContext
from app.agents.v1.graph_factory import GraphFactory
from app.config.nacos_client import get_nacos_client
from app.schemas.ai_ask_schema import AIAskRequest
from app.services import llm_service
from app.utils.logger import app_logger as logger
from app.agents.callbacks.agent_trace_callback import AgentTraceCallback


class AIChatService:
    """
    AI 对话服务（单例模式）

    架构说明：
    1. checkpointer: AsyncPostgresSaver，用于 LangGraph checkpoint 存储
    2. conversations_manager: ConversationsManager，用于会话元数据管理
    3. 通过 thread_id（即 session_id）来区分不同会话，实现多会话隔离
    4. thread_id = session_id（前端传递）
    5. 支持流式输出和对话历史管理
    """

    _instance: Optional["AIChatService"] = None

    def __init__(self, checkpointer, conversations_manager):
        """初始化 AI 对话服务

        Args:
            checkpointer: AsyncPostgresSaver 实例
            conversations_manager: ConversationsManager 实例
        """
        self._graph = None
        self.checkpointer = checkpointer
        self.conversations_manager = conversations_manager

        # 获取聊天配置
        chat_config = get_nacos_client().get_chat_config()

        # 提取其他配置
        self.max_clarification_count = chat_config.get("max_clarification_count", 3)
        self.max_history_task_count = chat_config.get("max_history_task_count", 3)
        self.max_search_loop = chat_config.get("max_search_loop", 3)
        self.max_tool_loop = chat_config.get("max_tool_loop", 3)

        logger.info("AIChatService 初始化完成")

    def _ensure_graph(self):
        """懒加载初始化 graph"""
        if self._graph is None:
            self._graph = GraphFactory.build_all(self.checkpointer).get_graph()
        return self._graph

    @classmethod
    def get_instance(cls) -> "AIChatService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    # 内部 pipeline 节点，不向前端暴露 thinking 过程
    _INTERNAL_NODES = {"query_rewrite_node", "intent_decomposer_node"}

    # 需要向前端展示的节点名称（这些节点会产生前端 UI 步骤）
    _NODE_NAMES = {
        "intent_decomposer_node",
        "searching_subgraph_node",
        "filter_node",
        "comparison_subgraph_node",
        "platform_node",
        "chitchat_node",
        "aggregator_node",
    }

    # 需要向前端展示的工具名称（这些工具会产生前端 UI 工具气泡）
    _TOOL_NAMES = {
        # 搜索类
        "search_product",
        "platform_knowledge_search",
        "tavily_search",
        # 商品类
        "get_new_product",
        "get_product_detail",
        # 天气类
        "get_current_weather",
        "get_forecast_weather",
    }

    async def chat_stream(self, request: AIAskRequest) -> AsyncGenerator[str, None]:
        """
        流式对话（完整事件流版本）
        
        支持发送以下事件：
        - thinking_start / thinking_end: AI 思考过程
        - tool_calls_detected: 检测到工具调用
        - tool_start: 工具开始执行
        - tool_progress: 工具执行进度
        - tool_complete: 工具执行完成
        - token_stream: AI 回复的流式文本
        
        Args:
            request: 对话请求
            
        Yields:
            SSE 格式的事件流
        """
        try:
            # 生成 thread_id
            thread_id = request.session_id
            
            logger.info(
                f"开始流式对话",
                extra={
                    "thread_id": thread_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "question": request.question,
                }
            )
            
            # 构建输入
            input_messages = [HumanMessage(content=request.question)]

            # 配置
            config = RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    "max_clarification_count": self.max_clarification_count,
                    "max_history_task_count": self.max_history_task_count,
                    "max_search_loop": self.max_search_loop,
                    "max_tool_loop": self.max_tool_loop,
                },
                # callbacks=[AgentTraceCallback(thread_id=thread_id)],
            )

            # 上下文
            context = ShopmindAssistantContext(llm=llm_service.get_llm_service().get_chat_model(),
                                               reasoning_llm=llm_service.get_llm_service().get_reasoning_model(),
                                               thread_id=thread_id,
            )
            
            # 记录是否发生过流式输出 (用于兜底拦截)
            has_streamed_answer = False
            
            # 流式输出（使用 astream_events 捕获完整事件流）
            graph = self._ensure_graph()
            async for event in graph.astream_events(
                {
                    "messages": input_messages,
                    "original_query": request.question,
                    "rewritten_query": None,
                    "current_tasks": [],
                    "sub_task_results": [],
                    "answer": None,
                },
                context=context,
                config=config,
                version="v2",
            ):
                kind = event["event"]                       # 事件类型
                name = event.get("name", "")    # 事件名称

                # 1. 拦截配置过的话术节点产生前端 UI 步骤气泡
                if name in self._NODE_NAMES:
                    if kind == "on_chain_start":
                        sse_data = {"node_name": name,  "status": "executing"}
                        yield self._format_sse_event("thinking_start", sse_data)
                    elif kind == "on_chain_end":
                        sse_data = {"node_name": name, "status": "completed"}
                        yield self._format_sse_event("thinking_end", sse_data)

                # 1.5 拦截配置过的工具产生前端 UI 工具气泡
                elif name in self._TOOL_NAMES:
                    if kind == "on_tool_start":
                        data_input = event.get("data", {}).get("input", {})
                        tool_args = {}
                        if isinstance(data_input, dict):
                            tool_args = {k: v for k, v in data_input.items() if k not in ["name", "id"]}
                        sse_data = {"tool_name": name, "tool_args": tool_args, "status": "executing"}
                        yield self._format_sse_event("tool_start", sse_data)
                    elif kind == "on_tool_end":
                        sse_data = {"tool_name": name, "status": "completed"}
                        yield self._format_sse_event("tool_complete", sse_data)

                # 2. 流式输出最终回答
                elif kind == "on_chat_model_stream" and FINAL_ANSWER_TAG in event.get("tags", []):
                    chunk = event.get("data", {}).get("chunk", "")
                    if chunk:
                        content = ""
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk

                        if content:
                            yield self._format_sse_event("token_stream", {
                                "content": content,
                                "node": name
                            })

            # 发送完成事件
            yield self._format_sse_event("complete", {"message": "对话完成"})
            
            logger.info(
                f"流式对话完成",
                extra={"thread_id": thread_id}
            )
                        
        except Exception as e:
            logger.error(f"流式对话失败: {e}", exc_info=True)
            # todo 但是无法存储到 messages 消息历史了。 预期解决办法: 前端在缺失的 ai 消息块上加一个刷新按钮，后端负责重新 chat_stream
            yield self._format_sse_event("error", {"message": "抱歉，服务暂时不可用"})

    def _format_sse_event(self, event_type: str, data: dict) -> str:
        """格式化 SSE 事件"""
        return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

    async def clear_history(self, session_id: str) -> bool:
        """
        清除对话历史

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            是否清除成功
        """
        try:
            await self.checkpointer.adelete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True
        except Exception as e:
            logger.error(f"清除对话历史失败: {e}", exc_info=True)
            return False
    
    async def get_history(self, session_id: str) -> list[dict]:
        """
        获取对话历史

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            消息历史列表
        """
        try:
            config = RunnableConfig(
                configurable={"thread_id": session_id, "checkpoint_ns": ""}
            )
            tuple_result = await self.checkpointer.aget_tuple(config)
            if not tuple_result:
                return []

            checkpoint = tuple_result.checkpoint
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])

            result = []
            for msg in messages:
                if isinstance(msg, AIMessage) and not msg.content:
                    continue
                if isinstance(msg, ToolMessage):
                    continue
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    result.append(
                        {
                            "role": "user" if msg.type == "human" else "assistant",
                            "content": msg.content,
                        }
                    )

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}", exc_info=True)
            return []
    
    # ========== 对话列表管理方法 ==========
    
    async def get_conversation_list(self, user_id: str) -> list[dict]:
        """
        获取用户的所有对话列表

        Args:
            user_id: 用户ID

        Returns:
            对话列表
        """
        logger.info(f"获取用户【user_id: {user_id}】的消息历史列表")
        return await self.conversations_manager.get_conversation_list(user_id)

    async def create_conversation(self, user_id: str, session_id: str, name: str) -> bool:
        """
        创建新对话

        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 对话名称

        Returns:
            是否创建成功
        """
        return await self.conversations_manager.create_conversation(user_id, session_id, name)

    async def update_conversation_name(self, user_id: str, session_id: str, name: str) -> bool:
        """
        更新对话名称

        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 新对话名称

        Returns:
            是否更新成功
        """
        return await self.conversations_manager.update_conversation_name(user_id, session_id, name)

    async def delete_conversation(self, user_id: str, session_id: str) -> bool:
        """
        删除对话

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        # 删除会话元数据
        deleted = await self.conversations_manager.delete_conversation(user_id, session_id)
        if deleted:
            # 同时清除 checkpoint 历史
            await self.checkpointer.adelete_thread(session_id)
        return deleted

    async def get_conversation_name(self, user_id: str, session_id: str) -> Optional[str]:
        """
        获取指定对话的名称

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            对话名称
        """
        return await self.conversations_manager.get_conversation_name(user_id, session_id)
    


def get_ai_chat_service() -> AIChatService:
    """获取 AI 对话服务单例"""
    return AIChatService.get_instance()

def init_ai_chat_service(checkpointer, conversations_manager) -> None:
    """初始化 AI 对话服务

    Args:
        checkpointer: AsyncPostgresSaver 实例
        conversations_manager: ConversationsManager 实例
    """
    AIChatService._instance = AIChatService(checkpointer, conversations_manager)
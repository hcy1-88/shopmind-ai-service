"""AI 对话服务 - 生产级实现."""

from typing import AsyncGenerator, Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.services.rag_service import get_rag_service
from langchain.agents import create_agent

from app.config.nacos_client import get_nacos_client
from app.schemas.ai_ask_schema import AIAskRequest
from app.services import llm_service
from app.checkpoints import get_redis_checkpoint_saver
from app.tools.chat_tool import platform_knowledge_search
from app.utils.logger import app_logger as logger


class AIChatService:
    """
    AI 对话服务（单例模式）
    
    架构说明：
    1. Redis Checkpoint 存储，所有用户共享同一个实例
    2. 通过 thread_id（即 session_id）来区分不同会话，实现多会话隔离
    3. thread_id = session_id（前端传递）
    4. 支持流式输出和对话历史管理
    5. 消息历史默认 2 小时过期
    """
    
    _instance: Optional["AIChatService"] = None
    
    def __init__(self):
        """初始化 AI 对话服务"""
        # Redis 短期记忆，ttl 个小时过期
        chat_config = get_nacos_client().get_chat_config()
        ttl = chat_config.get("checkpoint_expire", 2)
        self.checkpointer = get_redis_checkpoint_saver(ttl=ttl*3600)
        
        # 系统提示词
        self.system_prompt = (
            "你是一个友好且专业的 AI 购物助手。你可以帮助用户推荐商品、回答问题、提供购物建议。"
            "当用户询问平台规则、政策、流程等问题时，使用 platform_knowledge_search 工具搜索知识库获取准确信息。"
        )
        
        # 创建对话图（单例，所有请求复用）
        self.graph = self._build_graph()
        
        logger.info("AIChatService 初始化完成")
    
    @classmethod
    def get_instance(cls) -> "AIChatService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    
    def _build_graph(self):
        """
        构建对话图（仅在初始化时调用一次）
        
        Returns:
            编译后的 LangGraph 图实例（带 RAG 工具的 Agent）
        """
        # 获取 LLM
        llm = llm_service.get_llm_service().get_chat_model()
        
        # 获取 RAG 查询工具
        tools = [platform_knowledge_search]
        logger.info("RAG工具已集成")
        
        # 使用 create_react_agent 创建带工具的 Agent
        # state_modifier 可以是字符串（系统提示词）或函数
        graph = create_agent(
            llm,
            tools=tools,
            checkpointer=self.checkpointer,
            system_prompt=self.system_prompt  # 使用定义的系统提示词
        )
        
        logger.info(f"LangGraph 对话图构建完成（单例模式，集成了 {len(tools)} 个工具）")
        return graph
    
    async def chat_stream(self, request: AIAskRequest) -> AsyncGenerator[str, None]:
        """
        流式对话
        
        Args:
            request: 对话请求
            
        Yields:
            AI 回复的文本片段（增量）
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
            config = {
                "configurable": {"thread_id": thread_id}
            }
            
            # 流式输出（使用单例 graph）
            async for event in self.graph.astream_events(
                {"messages": input_messages},
                config=config,
                version="v2",
            ):
                # 只处理 LLM 生成的 token
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
            
            logger.info(
                f"流式对话完成",
                extra={"thread_id": thread_id}
            )
                        
        except Exception as e:
            logger.error(f"流式对话失败: {e}", exc_info=True)
            yield f"\n\n[系统错误: {str(e)}]"
    
    async def clear_history(self, session_id: str) -> bool:
        """
        清除对话历史
        
        Args:
            session_id: 会话ID（即 thread_id）
            
        Returns:
            是否清除成功
        """
        try:
            thread_id = session_id
            success = await self.checkpointer.clear_thread_history(thread_id)
            if success:
                logger.info(f"已清除会话历史: {thread_id}")
            return success
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
            thread_id = session_id
            messages = await self.checkpointer.get_thread_messages(thread_id)
            logger.info(f"获取会话历史: {thread_id}, 消息数量: {len(messages)}")
            return messages
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}", exc_info=True)
            return []
    


def get_ai_chat_service() -> AIChatService:
    """获取 AI 对话服务单例"""
    return AIChatService.get_instance()

def init_ai_chat_service() -> None:
    get_ai_chat_service()
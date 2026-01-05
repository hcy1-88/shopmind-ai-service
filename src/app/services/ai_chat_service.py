"""AI 对话服务 - 生产级实现."""

from typing import AsyncGenerator, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from app.schemas.ai_ask_schema import AIAskRequest
from app.services import llm_service
from app.utils.logger import app_logger as logger


class AIChatService:
    """
    AI 对话服务（单例模式）
    
    架构说明：
    1. MemorySaver 是单例的，所有用户共享同一个实例
    2. 通过 thread_id 来区分不同用户/会话，实现多用户隔离
    3. thread_id 生成规则：
       - 如果前端传递 session_id，则使用 f"{user_id}:{session_id}"
       - 如果不传 session_id，则使用 f"{user_id}:default"
    4. 支持流式输出和对话历史管理
    """
    
    _instance: Optional["AIChatService"] = None
    
    def __init__(self):
        """初始化 AI 对话服务"""
        # 内存型短期记忆，单例共享
        self.checkpointer = MemorySaver()
        
        # 创建对话提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好且专业的 AI 购物助手。你可以帮助用户推荐商品、回答问题、提供购物建议。"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
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
            编译后的 LangGraph 图实例
        """
        # 获取 LLM
        llm = llm_service.get_llm_service().get_chat_model()
        
        # 定义调用模型的函数
        def call_model(state: MessagesState):
            """调用模型生成回复"""
            # 使用提示词模板
            chain = self.prompt | llm
            response = chain.invoke(state)
            return {"messages": response}
        
        # 构建图
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        
        # 编译图，添加 checkpointer 短期记忆
        graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("LangGraph 对话图构建完成（单例模式，所有请求将复用此实例）")
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
            user_id: 用户ID
            session_id: 会话ID，如果不传则清除该用户的所有会话
            
        Returns:
            是否清除成功
        """
        try:
            # 清除指定会话
            thread_id = session_id
            # MemorySaver 没有直接的删除方法，我们通过创建空的 checkpoint 来清除
            # 注意：MemorySaver 的实现中，我们需要直接操作内部存储
            if hasattr(self.checkpointer, 'storage'):
                # 清除指定 thread_id 的所有数据
                keys_to_delete = [k for k in self.checkpointer.storage.keys() if k[0] == thread_id]
                for key in keys_to_delete:
                    del self.checkpointer.storage[key]
                logger.info(f"已清除会话历史: {thread_id}")
                return True
            else:
                logger.warning("无法清除历史：checkpointer 没有 storage 属性")
                return False
        except Exception as e:
            logger.error(f"清除对话历史失败: {e}", exc_info=True)
            return False
    


def get_ai_chat_service() -> AIChatService:
    """获取 AI 对话服务单例"""
    return AIChatService.get_instance()
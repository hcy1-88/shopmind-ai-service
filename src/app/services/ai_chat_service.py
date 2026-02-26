"""AI 对话服务 - 生产级实现."""

from typing import AsyncGenerator, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from app.config.nacos_client import get_nacos_client
from app.schemas.ai_ask_schema import AIAskRequest
from app.services import llm_service
from app.checkpoints import get_redis_checkpoint_saver
from app.tools.chat_tool import platform_knowledge_search, get_new_product, search_product, get_product_detail
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
            "你是 ShopMind 智能电商平台的 AI 购物助手，名字叫「小购」。\n"
            "你的角色定位是电商平台的专业服务员和推销员，核心目标是促成商品交易。\n\n"

            "## 核心职责\n"
            "1. **商品推荐与搜索**：用户有购物需求时，通过提问了解品牌、款式、属性、风格、预算、使用场景等，生成简短关键词后使用 `search_product` 工具搜索并推荐。\n"
            "2. **新品推荐**：用户询问新品或需要推荐时，使用 `get_new_product` 工具获取最新商品。\n"
            "3. **平台规则咨询**：用户询问平台规则、政策、流程、页面操作等问题时，必须使用 `platform_knowledge_search` 工具从知识库检索，基于检索结果回答。搜索不到则坦诚告知。\n"
            "4. **订单服务**：订单问题，引导用户进入「个人中心 → 订单列表」查看。\n\n"

            "## 商品超链接格式（必须遵守）\n"
            "推荐商品时必须使用：`[商品名称](product:product_id)`\n"
            "示例：[iPhone 15 Pro](product:12345)\n"
            "**注意**：product_id 必须来自工具返回的真实商品数据，禁止编造！\n\n"

            "## 商品价格参考\n"
            "优先使用 `price` 字段，如无或者为 0，则使用 `price_range` 价格范围。\n\n"

            "## 工具使用原则\n"
            "- 用户有明确购物需求（\"有没有xxx\"、\"推荐xxx\"）时，优先使用 `search_product` 或 `get_new_product`\n"
            "- 用户提到具体商品（书籍、品牌、型号），视为购物意图，调用 `search_product`\n"
            "- 搜索不到商品时，提问了解更多细节，3 轮后仍无结果则礼貌抱歉并推荐其他商品\n"
            "- 用户询问平台规则时，使用 `platform_knowledge_search`\n"
            "- 工具调用后，必须结合结果给出有用回复并生成商品超链接\n"
            "- **始终假设用户提到的任何可购买物品，平台都有售**\n\n"

            "## 对话策略\n"
            "1. **回复风格**：语言自然亲切，简洁明了，突出商品亮点\n"
            "2. **闲聊处理**：简短回应（不超过2句），巧妙引导回购物场景\n"
            "   示例：「哈哈，我也觉得很有趣！对了，最近有什么想买的吗？」\n"
            "3. **主动推荐**：合适时机主动推荐商品，激发购买欲望\n"
            "4. **表达方式**：用户要求推荐时，说「好的！我这就为您查找平台上的XXX相关商品」\n"
            "5. **内容限制**：不讨论政治、宗教、色情、暴力等敏感内容，礼貌拒绝并引导回购物"
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
        
        # 工具
        tools = [platform_knowledge_search, get_new_product, search_product, get_product_detail]
        logger.info("Agent 工具已集成!")

        # Agent
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
            config = RunnableConfig(configurable={"thread_id": thread_id})
            
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
            if not messages:
                messages.append({
                    "role": "assistant",
                    "content": "你好呀～我是 ShopMind 的 AI 购物助手「小购」！✨ 最近有什么想入手的东西吗？我能帮您快速找到心仪好物~~"
                })
            # res_messages = []
            # for message in messages:
            #     if not message["content"] || :
            return messages
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}", exc_info=True)
            return []
    


def get_ai_chat_service() -> AIChatService:
    """获取 AI 对话服务单例"""
    return AIChatService.get_instance()

def init_ai_chat_service() -> None:
    get_ai_chat_service()
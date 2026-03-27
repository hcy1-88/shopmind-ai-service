"""就绪节点"""

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from app.agents.v1.schema import ShoppingSubgraphState, ShopmindAssistantContext
from app.tools.chat_tool import search_product, get_product_detail
from app.utils.logger import app_logger as logger


async def ready_node(state: ShoppingSubgraphState, context: ShopmindAssistantContext):
    """槽位齐全，根据商品信息调用 LLM 生成 tool call 或最终回复"""
    task = state["task"]
    subgraph_messages: list[BaseMessage] = state.get("subgraph_messages", [])
    llm = context.llm
    thread_id = context.thread_id

    logger.info(f"[ReadyNode] thread_id: {thread_id}, task_id: {task.task_id}")

    # 如果是空消息列表，则构造消息列表
    if not subgraph_messages:
        # 构建 system prompt: role description + available tools
        sys_prompt = """你是一个电商导购助手，正在为用户搜索商品，只需搜索到 5 个商品即可。

        ## 可用工具
        - search_product: 根据自然语言 query 搜索商品，返回商品列表（包含商品 ID、名称、价格等），一页大小固定为 5.
        - get_product_detail: 根据 product_id 获取商品详细信息（款式、价格、库存等 sku 规格）

        ## 任务
        1. 如果需要搜索商品，请调用 search_product 工具（参数 query 使用商品品类和关键词构造）
        2. 当搜索到商品后，请调用 get_product_detail 工具获取商品详情，它是完整的商品信息
        3. 最终输出所有你搜索到的商品详情（注意是商品详情）.

        ## 工具使用规则
        1，使用 search_product 搜索商品时，默认从第 1 页搜索，**如果用户消息明确了页号，请务必以用户指定的页号为起始搜索页**
        2. 如果没有搜索到任何商品或者搜索结果小于 5 个，请停止调用工具。因为第1页搜不到，第2页肯定也搜不到，没必要重试。

        ## 当不需要调用工具时，请返回推荐文案。
        """

        # 构建 user prompt: product_category + keywords（不包含 filters 或 has_recommended_product_ids）
        user_prompt = f"商品品类: {task.product_category or '未指定'}"
        if task.keywords:
            user_prompt += f"\n关键词: {', '.join(task.keywords)}"
        # 换一批，则需要指定页号
        if task.is_replace_products:
            user_prompt += f"搜索商品的起始页号是第 {max(task.searched_pages) + 1} 页！"

        subgraph_messages.append(SystemMessage(content=sys_prompt))
        subgraph_messages.append(HumanMessage(content=user_prompt))

    # 绑定工具并调用 LLM
    llm_with_tools = llm.bind_tools([search_product, get_product_detail])
    response = llm_with_tools.invoke({"messages": subgraph_messages})
    return {"subgraph_messages": [response]}

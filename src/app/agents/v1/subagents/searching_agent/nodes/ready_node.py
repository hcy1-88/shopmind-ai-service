"""就绪节点"""

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from app.agents.v1.schema import SearchingSubgraphState, ShopmindAssistantContext
from app.tools.chat_tool import search_product, get_product_detail
from app.utils.logger import app_logger as logger


async def ready_node(state: SearchingSubgraphState, runtime: Runtime[ShopmindAssistantContext]):
    """槽位齐全，根据商品信息调用 LLM 生成 tool call 或最终回复"""
    task = state["task"]
    subgraph_messages: list[BaseMessage] = state.get("subgraph_messages", [])
    llm = runtime.context.llm
    thread_id = runtime.context.thread_id

    logger.info(f"[ready_node] thread_id: {thread_id}, task_id: {task.task_id}")

    # 如果是空消息列表，则构造消息列表
    if not subgraph_messages:
        # 构建 system prompt: role description + available tools
        sys_prompt = """你是一个电商导购助手，正在为用户搜索商品，只需搜索到 5 个商品即可。

        ## 可用工具
        - search_product: 根据自然语言 query 搜索商品，返回商品列表（包含商品 ID、名称、价格等），一页大小固定为 5.
        - get_product_detail: 根据 product_id 获取商品详细信息（款式、价格、库存等 sku 规格），**必须调用此工具获取商品详情**

        ## 重要：工具调用顺序
        1. 首先调用 search_product 搜索商品，获得商品 ID 列表
        2. **必须**对每一个搜索到的商品，调用 get_product_detail 获取其完整详情
        3. 等待所有 get_product_detail 返回后，工具调用结束

        ## 工具使用规则
        1. 使用 search_product 搜索商品时，默认从第 1 页搜索，**如果用户消息明确了页号，请务必以用户指定的页号为起始搜索页**
        2. 如果第1页没有搜索到任何商品或者搜索结果小于 5 个，请停止调用工具。因为第1页搜索结果个数不够，第2页肯定也不够，没必要重试。

        ## 当不需要调用工具时，请返回推荐文案。
        """
        subgraph_messages.append(SystemMessage(content=sys_prompt))

    # 构建 user prompt: product_category + keywords（不包含 filters 或 has_recommended_product_ids）
    user_prompt = f"商品品类: {task.product_category or '未指定'}"
    if task.keywords:
        user_prompt += f"\n关键词: {', '.join(task.keywords)}"

    if task.is_replace_products:
        # 换一批场景：从 filter_node 返回，需要追加用户消息指定页号
        next_page = max(task.searched_pages) + 1 if task.searched_pages else 1
        searched = ", ".join(str(p) for p in task.searched_pages) if task.searched_pages else "无"
        user_prompt += (
            f"\n【换一批】你已经在以下页码搜索过：{searched}。"
            f"本次请从第 {next_page} 页开始搜索，**不要再重复搜索已搜索过的页码**。"
        )
    user_prompt += "\n 注意: **搜索到的每一件商品，务必查询商品详情**"
    subgraph_messages.append(HumanMessage(content=user_prompt))

    # 绑定工具并调用 LLM
    llm_with_tools = llm.bind_tools([search_product, get_product_detail])
    response = llm_with_tools.invoke(subgraph_messages)

    # 检测到 tool_calls 时递增 tool_loop，返回值写入 checkpoint 持久化
    if isinstance(response, AIMessage) and response.tool_calls:
        tool_loop = state.get("tool_loop", 0)
        return {
            "subgraph_messages": [response],
            "tool_loop": tool_loop + 1,
        }
    return {"subgraph_messages": [response]}

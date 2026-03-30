"""生成节点"""

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from app.agents.v1.schema import SearchingSubgraphState, ShopmindAssistantContext, TaskStatus
from app.schemas.product_response_schema import ProductResponseDto
from app.utils.logger import app_logger as logger


async def generate_node(state: SearchingSubgraphState, runtime: Runtime[ShopmindAssistantContext]):
    """基于过滤后的商品生成最终推荐文案"""
    task = state["task"]
    subgraph_messages: list[BaseMessage] = state.get("subgraph_messages", [])
    llm = runtime.context.llm
    thread_id = runtime.context.thread_id
    logger.info(f"[generate_node] thread_id: {thread_id}, task_id: {task.task_id}")
    # 构建 generate_system_msg
    generate_system_msg = """你是一个电商导购助手，正在为用户推荐商品。

## 核心职责
根据商品详情 和 用户请求，为用户生成商品推荐文案。

## 你的思考逻辑应该如下：
1，阅读用户发送的商品预览和商品详情信息，然后根据

## 商品超链接格式（必须严格遵守）
推荐商品时必须使用：`[商品名称](product:product_id)`
示例：[iPhone 15 Pro](product:12345)
**注意**：product_id 必须来自工具返回的真实商品数据，禁止编造！

## 贴商品图片
当你给出商品图片时，请贴上商品预览图的超链接，字段是与商品 id 平级的 image 字段，它是一个图片 url


## 任务
1. 根据用户提供的商品详情，生成向用户推荐商品的文案，你无需过滤任何商品
2. 每个商品的**超链接**必须使用 `[商品名称](product:product_id)` 格式，前端会自动渲染
3. 可以简要说明推荐理由
4. 语气友好、自然，像真人超时服务员导购一样
5. 如果没有商品详情，请友好表示暂无您满意的商品，我们会尽快补货

直接返回推荐文案，不要有多余解释，前端会以 markdown 格式展示文本。"""

    # 构建 generate_user_msg: 使用 product_after_filter
    user_query = task.sub_query
    final_recommend_products: list[ProductResponseDto] = state.get("product_after_filter", [])

    # 构建商品详情文本：全量序列化过滤后的商品
    if final_recommend_products:
        product_details_list = []
        for dto in final_recommend_products:
            product_details_list.append(dto.model_dump_json(indent=2))
        product_details_text = "\n\n".join(product_details_list)
    else:
        product_details_text = "无商品详情"

    generate_user_msg = f"""
## 用户请求
{user_query}

## 商品预览和商品详情
{product_details_text}

请根据以上商品详情，为用户生成推荐文案。"""

    # 构建 generate_messages 并调用 LLM
    generate_messages = [
        SystemMessage(content=generate_system_msg),
        HumanMessage(content=generate_user_msg),
    ]

    response = await llm.ainvoke(generate_messages)

    # 追加 AIMessage 到 subgraph_messages
    subgraph_messages.append(response)

    # 设置 final_response
    task.final_response = response.content.strip()
    task.status = TaskStatus.WAITING

    # 重置"换一批"
    task.is_replace_products = False

    return {
        "subgraph_messages": [],
        "searched_res": ["__CLEAR__"],
        "searched_details": ["__CLEAR__"],
        "filtered_product_ids": [],
        "product_after_filter": [],
        "search_count_loop": 0,
        "task": task,
    }

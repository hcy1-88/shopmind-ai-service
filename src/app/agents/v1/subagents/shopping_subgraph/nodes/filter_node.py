"""过滤节点"""

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agents.v1.schema import ShoppingSubTask, ShoppingSubgraphState, ShopmindAssistantContext, FilterResult
from app.schemas.product_response_schema import ProductResponseDto
from app.utils.logger import app_logger as logger


async def filter_node(state: ShoppingSubgraphState, context: ShopmindAssistantContext):
    """对工具搜索结果进行 LLM 语义过滤，并更新 has_searched_product_id"""
    task: ShoppingSubTask = state["task"]
    searched_details: list[ProductResponseDto] = state.get("searched_details", [])
    llm = context.llm
    thread_id = context.thread_id

    logger.info(f"[FilterNode] thread_id: {thread_id}, task_id: {task.task_id}, is_replace_products={task.is_replace_products}")

    # 如果没有搜索到任何商品
    if not searched_details:
        task.final_response = "抱歉，没有搜索到符合您条件的商品"
        return {"sub_task_results": [task]}

    # 构建商品详情文本：全量序列化搜索到的商品
    product_details_list = []
    for dto in searched_details:
        product_details_list.append(dto.model_dump_json(indent=2))
    product_details_text = "\n\n".join(product_details_list)

    filters = task.filters or {}
    has_searched = task.has_searched_product_ids or []
    is_replace = task.is_replace_products

    parser = PydanticOutputParser(pydantic_object=FilterResult)
    format_instructions = parser.get_format_instructions()

    # 构建 prompt：是否排除 has_searched_product_ids 取决于 is_replace_products
    if is_replace:
        exclude_note = "注意：必须排除 has_searched_product_ids 中的商品 ID，这些商品已展示过。"
    else:
        exclude_note = "注意：不要排除 has_searched_product_ids 中的商品，即使它们之前展示过。"

    system_prompt = f"""你是一个电商导购助手，负责从搜索结果中筛选出最符合用户需求的商品。

## 你的任务
1. 仔细阅读商品详情
2. 根据用户的过滤条件（filters）筛选商品
3. 返回过滤前的所有商品 ID 列表，以及需要保留的商品 ID 列表

## 排除规则
{exclude_note}

## 输出格式
请严格按以下输出格式返回：
{format_instructions}"""

    user_prompt = f"""## 商品详情
{product_details_text}

## 用户的过滤条件
{filters}

## has_searched_product_ids：已经展示过的商品 ID（{'需排除' if is_replace else '无需排除'})
{has_searched}

请根据以上信息，筛选出需要保留的商品 ID，返回 JSON 格式。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm | parser

    result: FilterResult = await chain.ainvoke({
        "filters": filters,
    })
    logger.info(f"[FilterNode] result: {result}")

    # 填充 product_after_filter：从 searched_details 中过滤出保留的商品详情
    product_after_filter = [
        dto for dto in searched_details
        if dto.product_id in result.filtered_product_ids
    ]

    # 更新 has_searched_product_ids
    existing = set(task.has_searched_product_ids)
    task.has_searched_product_ids = list(existing | set(task.filtered_product_ids))

    # 更新状态
    return {
        "product_after_filter": product_after_filter,
        "task": task,
        "filtered_product_ids": result.filtered_product_ids
    }

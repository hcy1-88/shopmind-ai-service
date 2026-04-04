"""过滤节点"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_config
from langgraph.runtime import Runtime

from app.agents.v1.schema import ShoppingSubTask, SearchingSubgraphState, ShopmindAssistantContext, FilterResult
from app.agents.v1.config import MAX_SEARCH_LOOP
from app.schemas.product_response_schema import ProductResponseDto
from app.utils.logger import app_logger as logger


async def filter_node(state: SearchingSubgraphState, runtime: Runtime[ShopmindAssistantContext]):
    """对工具搜索结果进行 LLM 语义过滤，并更新 has_searched_product_id"""
    task: ShoppingSubTask = state["task"]
    searched_details: list[ProductResponseDto] = state.get("searched_details", [])
    llm = runtime.context.reasoning_llm
    thread_id = runtime.context.thread_id
    cfg = get_config().get("configurable", {})
    max_search_loop = cfg.get(MAX_SEARCH_LOOP, 3)

    logger.info(f"[filter_node] thread_id: {thread_id}, task_id: {task.task_id}")

    # 如果没有搜索到任何商品
    if not searched_details:
        task.final_response = "抱歉，没有搜索到符合您条件的商品"
        # 强制跳转到 generator_node：设一个必定超限的值
        return {"task": task, "search_count_loop": max_search_loop}

    # 构建商品详情文本：全量序列化搜索到的商品
    product_details_list = []
    for dto in searched_details:
        product_details_list.append(dto.model_dump_json(indent=2))
    product_details_text = "\n\n".join(product_details_list)

    filters = task.filters or {}
    has_recommended = task.has_recommended_product_ids or []
    is_replace = task.is_replace_products
    product_category = task.product_category if task.product_category else "未指定"

    parser = PydanticOutputParser(pydantic_object=FilterResult)
    format_instructions = parser.get_format_instructions()

    if is_replace:
        exclude_note = "注意：必须排除 has_recommended_product_ids 中的商品 ID，这些商品已展示过。"
    else:
        exclude_note = "注意：不要排除 has_recommended_product_ids 中的商品，即使它们之前展示过。"

    # format_instructions 已经在 f-string 里展开成普通文本，不会有 {} 模板变量冲突
    system_prompt = """你是一个电商导购助手，负责从搜索结果中筛选出最符合用户需求的商品。

## 你的任务
1. 仔细阅读商品详情 JSON，提取其中的商品 ID（id 字段）和商品品类信息
2. 根据用户的过滤条件（filters）和商品品类（product_category）筛选商品
3. 返回过滤前的所有商品 ID 列表（all_products_ids）、过滤后需要保留的商品 ID 列表（filtered_product_ids），以及过滤理由（reason）

## 品类过滤规则（product_category）
当用户指定了商品品类（product_category）时，你需要根据商品详情中的品类信息进行相关性过滤：
- 只保留与 product_category 品类相关的商品
- 排除与 product_category 完全不相关的商品（如用户要洗发水，但搜索结果中有耳机、防晒霜等）
- 如果商品详情中没有品类信息，可以根据商品名称/描述推断

## 排除规则
1. 如果 filters 过滤条件为空且 product_category 也为空，说明没有过滤条件，所有商品都应该保留在 filtered_product_ids 中
2. {exclude_note}

## 输出格式
请严格按以下输出格式返回：
{format_instructions}"""

    user_prompt = """## 商品详情
{product_details_text}

## 用户的过滤条件（filters）
{filters}

## 用户期望的商品品类（product_category）
{product_category}

## has_recommended_product_ids：
{has_recommended}

请提取商品详情中的商品 ID，筛选出需要保留的商品 ID，返回 JSON 格式。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm | parser

    result: FilterResult = chain.invoke({
        "exclude_note": exclude_note,
        "format_instructions": format_instructions,
        "product_details_text": product_details_text,
        "filters": filters,
        "product_category": product_category,
        "has_recommended": has_recommended,
    })

    logger.info(f"[过滤节点] 过滤结果: all={result.all_products_ids}, filtered={result.filtered_product_ids}, reason={result.reason}")

    # 填充 product_after_filter：从 searched_details 中过滤出保留的商品详情
    # 商品 id 可能是整数或字符串，需要统一按字符串比较
    filtered_ids_str = {str(i) for i in result.filtered_product_ids}
    product_after_filter = [
        dto for dto in searched_details
        if str(dto.id) in filtered_ids_str
    ]

    # 更新 has_recommended_product_ids
    existing = set(task.has_recommended_product_ids)
    task.has_recommended_product_ids = list(existing | set(result.filtered_product_ids))

    # 如果过滤结果为空，且未超过 search_count_loop 上限，需要"换一批"
    # search_count_loop 和 tool_loop 的更新通过返回值捎带
    if not result.filtered_product_ids:
        cfg = get_config().get("configurable", {})
        max_search_loop = cfg.get(MAX_SEARCH_LOOP, 3)
        search_count_loop = state.get("search_count_loop", 0)
        next_count = search_count_loop + 1

        if next_count < max_search_loop:
            # 未超限，设置换一批标记，通过返回值更新状态
            task.is_replace_products = True
            return {
                "task": task,
                "product_after_filter": [],
                "filtered_product_ids": [],
                "search_count_loop": next_count,
                "tool_loop": 0,
            }

    return {
        "product_after_filter": product_after_filter,
        "task": task,
        "filtered_product_ids": result.filtered_product_ids,
    }

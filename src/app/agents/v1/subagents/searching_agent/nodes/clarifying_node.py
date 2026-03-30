"""澄清节点"""

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from app.agents.v1.schema import ShoppingSubTask, SearchingSubgraphState, ShopmindAssistantContext
from app.agents.v1.utils import build_history_context
from app.utils.logger import app_logger as logger


async def clarifying_node(state: SearchingSubgraphState, runtime: Runtime[ShopmindAssistantContext]):
    """对当前购买商品进行澄清询问，生成问题"""
    task: ShoppingSubTask = state["task"]
    messages: list[BaseMessage] = state.get("messages", [])
    llm = runtime.context.llm
    thread_id = runtime.context.thread_id

    logger.info(f"[clarifying_node] thread_id: {thread_id}, task_id: {task.task_id}")

    # 构建槽位状态描述
    product_category = task.product_category or "未指定"
    keywords = task.keywords or []
    filters = task.filters or {}

    # 判断哪些槽位缺失
    missing_slots = []
    if not task.product_category:
        missing_slots.append("商品品类（product_category）")
    if not keywords:
        missing_slots.append("搜索关键词（keywords）")
    if not filters:
        missing_slots.append("过滤条件（filters，如价格区间、颜色等）")

    history_text = build_history_context(messages)

    prompt = f"""你是一个电商导购助手，正在与用户对话，帮助用户明确购物需求。

## 当前已知的商品信息
- 商品品类: {product_category}
- 关键词: {', '.join(keywords) if keywords else '暂无'}
- 过滤条件: {filters if filters else '暂无'}

## 缺失的信息
{', '.join(missing_slots) if missing_slots else '无'}

## 对话历史
{history_text}

## 任务
请根据以上信息，向用户提出一个自然、友好的问题，以获取缺失的商品信息。要求：
1. 只问 1-2 个最重要的问题
2. 问题要结合已知的商品信息，显得有上下文
3. 问题要简洁、具体，避免泛泛而问
4. 用口语化的方式表达，像真人对话一样
5. 直接输出问题即可，不要加任何解释
"""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        question = response.content.strip()
        if not question:
            question = "能再告诉我一些关于您想要的商品信息吗？"
    except Exception as e:
        logger.error(f"[ClarifyingNode] LLM 调用失败: {e}", exc_info=True)
        question = "能再告诉我一些关于您想要的商品信息吗？"

    task.final_response = question
    task.clarification_count += 1

    return {"task": task}

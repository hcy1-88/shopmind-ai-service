"""
@File       : platform_node.py
@Description: 平台规则节点 - 处理平台政策、规则、流程等查询

@Time       : 2026/3/23 2:08
@Author     : hcy18
"""
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agents.v3.schema import (
    ShopmindAssistantContext,
    PlatformSubTask,
    TaskStatus,
)
from app.agents.v3.utils import build_history_context
from app.tools.chat_tool import platform_knowledge_search
from app.utils.logger import app_logger as logger


class PlatformNodeState(BaseModel):
    sub_task: PlatformSubTask
    messages: list[BaseMessage]


async def platform_node(state: PlatformNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """
    平台规则节点 - 接收 PlatformSubTask，使用 RAG 知识库检索并生成自然语言回答

    输入（通过 Send 传入）:
        state: PlatformNodeState - 包含 sub_task: PlatformSubTask 和 messages: list[BaseMessage]

    输出:
        dict - 包含更新后的 sub_task
    """
    context = runtime.context
    llm = context.llm
    sub_task: PlatformSubTask = state.sub_task
    messages = state.messages

    thread_id = context.thread_id
    query = sub_task.original_query
    history_text = build_history_context(messages)
    logger.info(f"[PlatformNode] thread_id: {thread_id}, query: {query}, history: {history_text[:100]}...")

    try:
        # Step 1: 调用 platform_knowledge_search 工具检索知识库
        search_result = platform_knowledge_search.invoke(query)
        logger.info(f"[PlatformNode] 知识库检索结果: {search_result[:200] if len(search_result) > 200 else search_result}...")

        # Step 2: 使用 LLM 基于检索结果和历史上下文生成自然语言回答
        prompt = f"""你是一个电商平台的客服助手。请根据以下信息，用自然语言回答用户的问题。

## 历史对话上下文
{history_text}

## 检索到的知识库内容
{search_result}

## 用户问题
{query}

要求：
1. 如果知识库中有相关信息，请结合历史上下文给出清晰、完整的回答
2. 如果知识库中没有找到相关信息，请礼貌地告知用户"抱歉，未找到相关的平台政策信息"
3. 如果你认为上下文不能确定地解答用户的问题，可以请求用户补充更多的信息
4. 回答要简洁有条理，避免重复检索到的原文
5. 如果用户的问题与历史上下文中的商品相关，请结合该商品信息回答
"""
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        final_response = response.content.strip()

        # Step 3: 更新 sub_task
        sub_task.final_response = final_response
        sub_task.status = TaskStatus.COMPLETED
        logger.info(f"[PlatformNode] 完成，final_response: {final_response[:100]}...")

    except Exception as e:
        logger.error(f"[PlatformNode] 执行失败: {e}", exc_info=True)
        sub_task.status = TaskStatus.FAILED
        sub_task.final_response = f"处理平台规则查询时发生错误: {str(e)}"

    return {"sub_task_results": [sub_task]}
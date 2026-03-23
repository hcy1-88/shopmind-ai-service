"""
@File       : chitchat_node.py
@Description: 闲聊节点 - 处理闲聊、天气查询、联网搜索等

@Time       : 2026/3/23 2:08
@Author     : hcy18
"""
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agents.v3.subagents.chitchat_service import get_chitchat_service
from app.agents.v3.schema import ChitchatSubTask, ShopmindAssistantContext, TaskStatus
from app.utils.logger import app_logger as logger


class ChitChatNodeState(BaseModel):
    """闲聊节点状态"""
    sub_task: ChitchatSubTask
    messages: list[BaseMessage]


async def chitchat_node(state: ChitChatNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """
    闲聊节点 - 使用 ReAct Agent 处理闲聊、天气查询、联网搜索等

    输入（通过 Send 传入）:
        state: ChitChatNodeState - 包含 sub_task: ChitChatSubTask 和 messages: list[BaseMessage]

    输出:
        dict - 包含更新后的 sub_task
    """
    context = runtime.context
    sub_task: ChitchatSubTask = state.sub_task
    messages = state.messages

    thread_id = context.thread_id
    query = sub_task.original_query
    logger.info(f"[ChitChatNode] thread_id: {thread_id}, query: {query}")

    try:
        # 获取闲聊服务
        chitchat_service = get_chitchat_service()

        # 调用 agent 处理闲聊
        final_response = await chitchat_service.chat(query, messages)

        # 更新 sub_task
        sub_task.final_response = final_response
        sub_task.status = TaskStatus.COMPLETED
        logger.info(f"[ChitChatNode] 完成，final_response: {final_response[:100]}...")

    except Exception as e:
        logger.error(f"[ChitChatNode] 执行失败: {e}", exc_info=True)
        sub_task.status = TaskStatus.FAILED
        sub_task.final_response = f"处理闲聊时发生错误: {str(e)}"

    return {"sub_task": sub_task}

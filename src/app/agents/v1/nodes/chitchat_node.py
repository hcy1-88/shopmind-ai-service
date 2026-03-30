"""
@File       : chitchat_node.py
@Description: 闲聊节点 - 处理闲聊、天气查询、联网搜索等

@Time       : 2026/3/23 2:08
@Author     : hcy18
"""
from langgraph.runtime import Runtime
from app.agents.v1.subagents.chitchat_agent import get_chitchat_service
from app.agents.v1.schema import ChitchatSubTask, ShopmindAssistantContext, TaskStatus, ChitChatNodeState
from app.utils.logger import app_logger as logger


async def chitchat_node(state: ChitChatNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """
    闲聊节点 - 使用 ReAct Agent 处理闲聊、天气查询、联网搜索等

    输入（通过 Send 传入）:
        state: ChitChatNodeState - 包含 sub_task: ChitChatSubTask 和 messages: list[BaseMessage]

    输出:
        dict - 包含更新后的 sub_task
    """
    context = runtime.context
    logger.info(f"[chitchat_node] thread_id: {context.thread_id}")

    sub_task: ChitchatSubTask = state["sub_task"]
    messages = state["messages"]
    query = sub_task.sub_query

    try:
        chitchat_service = get_chitchat_service()
        final_response = await chitchat_service.chat(query, messages, f"chitchat_{sub_task.task_id}")
        sub_task.final_response = final_response
        sub_task.status = TaskStatus.WAITING
    except Exception as e:
        logger.error(f"[chitchat_node] 执行失败: {e}", exc_info=True)
        sub_task.status = TaskStatus.FAILED
        sub_task.final_response = f"处理闲聊时发生错误: {str(e)}"

    return {"sub_task_results": [sub_task]}

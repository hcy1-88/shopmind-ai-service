"""分发节点"""

from app.agents.v1.schema import SearchingSubgraphState, ShopmindAssistantContext
from app.utils.logger import app_logger as logger


async def dispatcher_node(state: SearchingSubgraphState, context: ShopmindAssistantContext):
    thread_id = context.thread_id
    task = state.get("task")
    logger.info(f"正在处理购物意图：thread_id: {thread_id}, task_id: {task.task_id}, status: {task.status}")

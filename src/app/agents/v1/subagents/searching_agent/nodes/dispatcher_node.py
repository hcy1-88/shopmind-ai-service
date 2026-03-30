"""分发节点"""
from langgraph.runtime import Runtime

from app.agents.v1.schema import SearchingSubgraphState, ShopmindAssistantContext
from app.utils.logger import app_logger as logger


async def dispatcher_node(state: SearchingSubgraphState,  runtime: Runtime[ShopmindAssistantContext]):
    thread_id = runtime.context.thread_id
    task = state.get("task")
    logger.info(f"[dispatcher_node] thread_id: {thread_id}, task_id: {task.task_id}, status: {task.status}")

"""
@File       : comparison_subgraph_node.py
@Description: 比较子图节点 - 调用 comparison 服务处理商品比较

@Time       : 2026/3/27
@Author     : hcy18
"""
from langgraph.runtime import Runtime

from app.agents.v1.schema import ShopmindAssistantContext, ComparisonSubTask, TaskStatus, \
    ComparisonNodeState
from app.agents.v1.subagents.comparison_agent.comparison_service import get_comparison_service
from app.utils.logger import app_logger as logger


async def comparison_subgraph_node(state: ComparisonNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """
    调用 comparison 服务处理商品比较

    ComparisonSubTask.product_ids 已包含待比较的商品ID列表
    """
    task: ComparisonSubTask = state["sub_task"]
    logger.info(f"[comparison_subgraph_node] task_id: {task.task_id}")

    product_ids = task.product_ids or []

    try:
        comparison_service = get_comparison_service()
        task.final_response = await comparison_service.compare(
            query=task.sub_query,
            product_ids=product_ids,
            thread_id=f"comparison_{task.task_id}"
        )
        task.status = TaskStatus.COMPLETED
        return {"sub_task_results": [task], "sub_tasks": [task]}
    except Exception as e:
        logger.error(f"[comparison_subgraph_node] 执行失败: {e}", exc_info=True)
        task.final_response = f"商品比较失败: {str(e)}"
        task.status = TaskStatus.FAILED
        return {"sub_task_results": [task], "sub_tasks": [task]}

"""
@File       : comparison_subgraph_node.py
@Description: 比较子图节点 - 调用 comparison 子图处理商品比较

@Time       : 2026/3/27
@Author     : hcy18
"""
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from app.agents.v1.schema import ShopmindAssistantContext, ChitChatNodeState, ComparisonSubTask, TaskStatus, \
    ComparisonNodeState
from app.agents.v1.subagents.comparison_agent.comparison_agent import ComparisonSubgraph
from app.utils.logger import app_logger as logger


async def comparison_subgraph_node(state: ComparisonNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """
    调用 comparison 子图 agent 处理商品比较

    ComparisonSubTask.product_ids 已包含待比较的商品ID列表
    """
    task: ComparisonSubTask = state["sub_task"]
    product_ids = task.product_ids or []

    logger.info(f"[ComparisonSubgraphNode] task_id: {task.task_id}, product_ids: {product_ids}")

    if not product_ids:
        logger.warning(f"[ComparisonSubgraphNode] No product_ids to compare")
        task.final_response = "没有选择要比较的商品"
        task.status = TaskStatus.COMPLETED
        return {"sub_task_results": [task]}

    # 构建子图状态
    comparison_state = {
        "task": task,
        "product_ids": product_ids,
        "product_details": [],
        "subgraph_messages": []
    }

    # 获取子图
    comparison_subgraph = ComparisonSubgraph.get_instance().get_graph()

    # 使用 task_id 作为 thread_id，支持 checkpoint
    config = RunnableConfig(configurable={"thread_id": f"comparison_{task.task_id}"})

    # 调用子图
    try:
        comparison_output = comparison_subgraph.invoke(comparison_state, context=runtime.context, config=config)
        result_task = comparison_output.get("task", task)
        return {"sub_task_results": [result_task]}
    except Exception as e:
        logger.error(f"[ComparisonSubgraphNode] Subgraph execution failed: {e}", exc_info=True)
        task.final_response = f"商品比较失败: {str(e)}"
        task.status = TaskStatus.FAILED
        return {"sub_task_results": [task]}

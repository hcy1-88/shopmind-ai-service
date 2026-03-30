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
    logger.info(f"[comparison_subgraph_node] task_id: {task.task_id}")

    product_ids = task.product_ids or []
    if not product_ids:
        task.final_response = "没有选择要比较的商品"
        task.status = TaskStatus.WAITING
        return {"sub_task_results": [task]}

    comparison_state = {
        "task": task,
        "product_ids": product_ids,
        "product_details": [],
        "subgraph_messages": []
    }

    comparison_subgraph = ComparisonSubgraph.get_instance().get_graph()
    config = RunnableConfig(configurable={"thread_id": f"comparison_{task.task_id}"})

    try:
        comparison_output = await comparison_subgraph.ainvoke(comparison_state, context=runtime.context, config=config)
        task.status = TaskStatus.COMPLETED
        result_task = comparison_output.get("task", task)
        return {"sub_task_results": [result_task]}
    except Exception as e:
        logger.error(f"[comparison_subgraph_node] 执行失败: {e}", exc_info=True)
        task.final_response = f"商品比较失败: {str(e)}"
        task.status = TaskStatus.FAILED
        return {"sub_task_results": [task]}

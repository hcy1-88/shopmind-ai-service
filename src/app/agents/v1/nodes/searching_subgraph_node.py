"""
@File       : searching_subgraph_node.py
@Description:

@Time       : 2026/3/26 17:53
@Author     : hcy18
"""
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from app.agents.v1.schema import ShopmindAssistantContext, ShoppingNodeState, TaskStatus
from app.agents.v1.subagents.searching_agent.searching_agent import SearchingSubgraph
from app.utils.logger import app_logger as logger


async def searching_subgraph_node(state: ShoppingNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """调用 shopping 子图 agent """
    task = state["sub_task"]
    logger.info(f"[searching_subgraph_node] task_id: {task.task_id}")

    shopping_state = {
        "task": task,
        "messages": state["messages"]
    }
    shopping_subgraph = SearchingSubgraph.get_instance().get_graph()
    config = RunnableConfig(configurable={"thread_id": f"search_{task.task_id}"})
    shopping_output = await shopping_subgraph.ainvoke(shopping_state, context=runtime.context, config=config)
    return {"sub_task_results": [shopping_output["task"]]}

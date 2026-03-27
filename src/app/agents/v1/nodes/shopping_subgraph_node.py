"""
@File       : shopping_subgraph_node.py
@Description:

@Time       : 2026/3/26 17:53
@Author     : hcy18
"""
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from app.agents.v1.schema import ShopmindAssistantContext, ShoppingNodeState
from app.agents.v1.subagents.shopping_agent import ShoppingSubgraph


async def shopping_subgraph_node(state: ShoppingNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """调用 shopping 子图 agent """
    task = state["sub_task"]
    shopping_state = {
        "task": task,
        "messages": state["messages"]
    }
    # 获取子图
    shopping_subgraph = ShoppingSubgraph.get_instance().get_graph()
    # 使用 task_id 作为 thread_id，让子图拥有独立的 checkpoint 记忆
    config = RunnableConfig(configurable={"thread_id": task.task_id})
    shopping_output = shopping_subgraph.invoke(shopping_state, config=config)
    return {"sub_task_results": [shopping_output["task"]]}

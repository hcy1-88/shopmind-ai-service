"""
@File       : shopping_subgraph_node.py
@Description:

@Time       : 2026/3/26 17:53
@Author     : hcy18
"""
from langgraph.runtime import Runtime

from app.agents.v1.schema import ShopmindAssistantContext, ShoppingNodeState
from app.agents.v1.subagents.shopping_agent import ShoppingSubgraph


async def shopping_subgraph_node(state: ShoppingNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """调用 shopping 子图 agent """
    shopping_state = {
        "task": state["sub_task"],
        "messages": state["messages"]
    }
    # 获取子图
    shopping_subgraph = ShoppingSubgraph.get_instance().get_graph()
    # 执行子图 todo 是否应该传入 thread_id ，让子图拥有记忆？ 父图已经拥有记忆了， ShoppingSubTask 可以被父图和子图共享修改，所以是不是没必要让子图拥有记忆？
    shopping_output = shopping_subgraph.invoke(shopping_state)
    return {"sub_task_results": [shopping_output["task"]]}

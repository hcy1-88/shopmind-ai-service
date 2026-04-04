"""
@File       : searching_subgraph_node.py
@Description:

@Time       : 2026/3/26 17:53
@Author     : hcy18
"""
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import RunnableConfig

from app.agents.v1.schema import ShopmindAssistantContext, ShoppingNodeState
from app.agents.v1.subagents.searching_agent.searching_agent import SearchingSubgraph
from app.agents.v1.config import MAX_SEARCH_LOOP, MAX_TOOL_LOOP
from app.utils.logger import app_logger as logger


async def searching_subgraph_node(state: ShoppingNodeState, runtime: Runtime[ShopmindAssistantContext]):
    """调用 shopping 子图 agent """
    task = state["sub_task"]
    logger.info(f"[searching_subgraph_node] task_id: {task.task_id}, id(task): {id(task)}, clarification_count: {getattr(task, 'clarification_count', 'N/A')}")

    # 从父图 config 读取子图配置
    cfg = get_config().get("configurable", {})
    max_search_loop = cfg.get(MAX_SEARCH_LOOP, 3)
    max_tool_loop = cfg.get(MAX_TOOL_LOOP, 3)

    shopping_state = {
        "task": task,
        "messages": state["messages"],
        "search_count_loop": 0,
        "tool_loop": 0,
    }
    shopping_subgraph = SearchingSubgraph.get_instance().get_graph()
    # max_* 配置通过 subgraph config 传递，供边函数通过 get_config() 读取
    config = RunnableConfig(
        configurable={
            "thread_id": f"search_{task.task_id}",
            MAX_SEARCH_LOOP: max_search_loop,
            MAX_TOOL_LOOP: max_tool_loop,
        }
    )
    shopping_output = await shopping_subgraph.ainvoke(shopping_state, context=runtime.context, config=config)
    task = shopping_output["task"]
    logger.info(f"[DEBUG in searching_subgraph_node] - 执行结束 - id(task): {id(task)}, clarification_count: {task.clarification_count}, task: {task}")
    # task 被子图修改，务必返回 task
    return {"sub_task_results": [task], "sub_tasks": [task]}
"""
@File       : route_to_map_edge.py
@Description: 将任务分发到 Map Node 以并行执行

@Time       : 2026/3/23 1:56
@Author     : hcy18
"""
from langgraph.runtime import Runtime
from langgraph.types import Send
from app.agents.v1.schema import ShopmindAgentState, IntentCategory, ShopmindAssistantContext
from app.utils.logger import app_logger as logger

def route_to_map_node_edge(state: ShopmindAgentState, runtime: Runtime[ShopmindAssistantContext]) -> list[Send]:
    """
    MapReduce 路由函数
    """
    thread_id = runtime.context.thread_id
    logger.info(f"[route_to_map_node_edge] thread_id: {thread_id}")

    current_tasks = state.get("current_tasks", [])
    send_list = []
    for task in current_tasks:
        if task.category == IntentCategory.SHOPPING:
            send_list.append(Send("searching_subgraph_node", {"sub_task": task, "messages": state["messages"]}))
        elif task.category == IntentCategory.PLATFORM:
            send_list.append(Send("platform_node", {"sub_task": task, "messages": state["messages"]}))
        elif task.category == IntentCategory.COMPARISON:
            send_list.append(Send("comparison_subgraph_node", {"sub_task": task, "messages": state["messages"]}))
        else:
            send_list.append(Send("chitchat_node", {"sub_task": task, "messages": state["messages"]}))
    return send_list
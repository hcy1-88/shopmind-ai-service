"""根据状态路由"""
from typing import Literal

from app.agents.v1.schema import SearchingSubgraphState, TaskStatus


async def route_by_status_edge(state: SearchingSubgraphState) -> Literal["ready_node", "clarifying_node"]:
    """根据状态（clarifying、ready）进行路由"""
    task = state.get("task")
    if task.status == TaskStatus.CLARIFYING:
        return "clarifying_node"
    else:
        return "ready_node"

"""ready节点后路由"""
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage

from app.agents.v1.schema import SearchingSubgraphState


def route_after_ready(state: SearchingSubgraphState) -> Literal["filter_node", "tool_node"]:
    """根据 ready_node 输出的 AI message 是否有 tool_calls 决定路由"""
    subgraph_messages: list[BaseMessage] = state.get("subgraph_messages", [])
    if not subgraph_messages:
        return "filter_node"
    last_message = subgraph_messages[-1]
    # 只有 AIMessage（带 tool_calls）才进 tool_node；ToolMessage 已执行完工具，直接进 filter_node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "filter_node"

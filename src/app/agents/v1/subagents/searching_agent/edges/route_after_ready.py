"""ready节点后路由"""
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.config import get_config

from app.agents.v1.schema import SearchingSubgraphState
from app.agents.v1.config import MAX_TOOL_LOOP


def route_after_ready(state: SearchingSubgraphState) -> Literal["filter_node", "tool_node"]:
    """根据 ready_node 输出的 AI message 是否有 tool_calls 决定路由

    当 tool_loop >= max_tool_loop 时，强制跳转到 filter_node，防止无限循环。
    tool_loop 的递增在 ready_node 返回时通过 Command捎带，本边只负责读取和判断。
    """
    subgraph_messages: list[BaseMessage] = state.get("subgraph_messages", [])
    if not subgraph_messages:
        return "filter_node"
    last_message = subgraph_messages[-1]
    # 只有 AIMessage（带 tool_calls）才进 tool_node；ToolMessage 已执行完工具，直接进 filter_node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_loop = state.get("tool_loop", 0)
        max_tool_loop = get_config().get("configurable", {}).get(MAX_TOOL_LOOP, 3)
        if tool_loop >= max_tool_loop:
            # 超限，强制跳转 filter_node
            return "filter_node"
        return "tool_node"
    return "filter_node"
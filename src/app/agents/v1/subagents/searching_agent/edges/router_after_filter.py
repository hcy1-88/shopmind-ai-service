"""filter节点后路由"""

from typing import Literal

from langgraph.config import get_config

from app.agents.v1.schema import SearchingSubgraphState
from app.agents.v1.config import MAX_SEARCH_LOOP


async def router_after_filter(state: SearchingSubgraphState) -> Literal["generate_node", "ready_node"]:
    """filter_node 之后的条件边"""
    task = state["task"]
    cfg = get_config().get("configurable", {})
    max_search_loop = cfg.get(MAX_SEARCH_LOOP, 3)

    # 如果没有过滤出任何商品
    if not state.get("filtered_product_ids"):
        # 递增 search_count
        current_count = state.get("search_count_loop", 0)
        new_count = current_count + 1

        if new_count >= max_search_loop:
            # 到 generator_node
            return "generate_node"
        else:
            # 设置 is_replace_products=True，换一批场景由 ready_node 处理消息构造
            task.is_replace_products = True
            state["search_count_loop"] = new_count
            # 重置 tool_loop=0，允许新一轮工具循环
            state["tool_loop"] = 0
            return "ready_node"
    return "generate_node"

"""filter节点后路由"""

from typing import Literal
from app.agents.v1.schema import SearchingSubgraphState


def router_after_filter(state: SearchingSubgraphState) -> Literal["generate_node", "ready_node"]:
    """filter_node 之后的条件边"""
    task = state["task"]
    # 如果没有过滤出任何商品
    if not state.get("filtered_product_ids"):
        # 递增 search_count
        current_count = state.get("search_count_loop", 0)
        new_count = current_count + 1

        if new_count >= task.max_search_loop:
            # 到 generator_node
            return "generate_node"
        else:
            # 设置 is_replace_products=True，换一批场景由 ready_node 处理消息构造
            task.is_replace_products = True
            state["search_count_loop"] = new_count
            return "ready_node"
    return "generate_node"

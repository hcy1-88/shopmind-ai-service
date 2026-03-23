"""filter节点后路由"""

from typing import Literal
from langchain_core.messages import HumanMessage
from app.agents.v1.schema import ShoppingSubgraphState


def router_after_filter(state: ShoppingSubgraphState) -> Literal["generate_node", "ready_node"]:
    """filer_node 之后的条件边"""
    task = state["task"]
    subgraph_messages: list = state.get("subgraph_messages", [])
    # 如果没有过滤出任何商品
    if not task.filtered_product_ids:
        # 递增 search_count
        current_count = state.get("search_count_loop", 0)
        new_count = current_count + 1

        if new_count >= task.max_search_loop:
            # 到 generator_node
            return "generate_node"
        else:
            # 回到 ready_node，追加用户消息触发下一页搜索
            next_page = max(task.searched_pages) + 1 if task.searched_pages else 1
            user_msg = f"您搜索到的商品均不符合我的条件，请从第 {next_page} 页重新搜索。"
            subgraph_messages.append(HumanMessage(content=user_msg))
            state["search_count_loop"] = new_count
            return "ready_node"
    return "generate_node"

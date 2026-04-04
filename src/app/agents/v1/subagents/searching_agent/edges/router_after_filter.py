"""filter节点后路由"""

from typing import Literal

from langgraph.config import get_config

from app.agents.v1.schema import SearchingSubgraphState
from app.agents.v1.config import MAX_SEARCH_LOOP


async def router_after_filter(state: SearchingSubgraphState) -> Literal["generate_node", "ready_node"]:
    """filter_node 之后的条件边

    search_count_loop 和 tool_loop 的更新在 filter_node 返回时捎带，本边只负责读取和判断。
    """
    cfg = get_config().get("configurable", {})
    max_search_loop = cfg.get(MAX_SEARCH_LOOP, 3)

    # 如果没有过滤出任何商品
    if not state.get("filtered_product_ids"):
        search_count_loop = state.get("search_count_loop", 0)
        if search_count_loop >= max_search_loop:
            return "generate_node"
        return "ready_node"
    return "generate_node"

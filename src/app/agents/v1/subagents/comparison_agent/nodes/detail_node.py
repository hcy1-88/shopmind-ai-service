"""
@File       : detail_node.py
@Description: 商品详情节点 - 获取待比较商品的详情

@Time       : 2026/3/27
@Author     : hcy18
"""

from app.agents.v1.schema import ComparisonSubgraphState
from app.tools.chat_tool import get_product_detail
from app.utils.logger import app_logger as logger


async def detail_node(state: ComparisonSubgraphState) -> dict:
    """
    商品详情节点 - 获取待比较商品的详情

    从 state.product_ids 获取商品ID列表，并行获取商品详情。

    Args:
        state: ComparisonSubgraphState，包含:
            - task: ComparisonSubTask
            - product_ids: list[int]
        context: 运行时上下文

    Returns:
        dict: 更新 product_details 到状态中
    """
    product_ids = state.get("product_ids", [])
    logger.info(f"[detail_node] product_ids to fetch: {product_ids}")

    if not product_ids:
        return {"product_details": []}

    # 并行获取商品详情
    product_details = []
    for product_id in product_ids:
        try:
            detail = await get_product_detail(product_id)
            if detail:
                product_details.append(detail)
        except Exception:
            pass

    return {"product_details": product_details}

"""
@File       : chat_tool.py
@Description:

@Time       : 2026/1/6 10:20
@Author     : hcy18
"""
from langchain_core.tools import tool

from app.clients.product_service import get_product_service_client
from app.schemas.product_response_schema import ProductResponseDto
from app.services.rag_service import get_rag_service


@tool
def platform_knowledge_search(query: str) -> str:
    """
    搜索平台规则和知识库，用于回答关于平台政策、规则、流程等问题。
    输入必须是一个具体、清晰的问题。

    Args:
        query: 要搜索的问题，例如 "如何申请退货？"

    Returns:
        知识库返回的答案，或错误信息。
    """
    try:
        rag_service = get_rag_service()
        index = rag_service.get_index()
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize"
        )
        response = query_engine.query(query)
        return response.response.strip() or "未在知识库中找到相关信息。"
    except Exception as e:
        return f"查询知识库时发生错误: {str(e)}"


@tool
async def get_new_product(limit: int = 3) -> list[ProductResponseDto]:
    """
    获取最新商品，调用的时候 limit 参数取值 3 ~ 5 , 不建议超过 5，不然消息太长
    Args:
        limit: 限制数，即 获取几个新品
    """
    product_client = await get_product_service_client()
    return await product_client.get_new_products(limit=limit)


@tool
async def search_product(query: str, page_number: int = 1, page_size: int = 3) -> list[ProductResponseDto]:
    """
    根据用户对商品的描述，搜索商品
    Args:
        query: 用户查询，比如 拍照好看的手机、苹果笔记本、送女朋友的礼物
        page_number: 分页的页码
        page_size: 一页的大小（不宜过大，不用超过 5，不然消息太长）
    """
    product_client = await get_product_service_client()
    return await product_client.search_products(query, page_number, page_size)
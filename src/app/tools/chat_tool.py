"""
@File       : chat_tool.py
@Description:

@Time       : 2026/1/6 10:20
@Author     : hcy18
"""
from langchain_core.tools import tool
from app.utils.logger import app_logger as logger
from app.clients.product_service import get_product_service_client
from app.schemas.product_response_schema import ProductResponseDto
from app.services.rag_service import get_rag_service


@tool
def platform_knowledge_search(query: str) -> str:
    """
    搜索平台规则和知识库，返回检索到的相关文档片段。
    用于回答关于平台政策、规则、流程等问题。
    输入必须是一个具体、清晰的问题。

    Args:
        query: 要搜索的问题，例如 "如何申请退货？"

    Returns:
        检索到的文档片段，或错误信息。
    """
    try:
        rag_service = get_rag_service()
        index = rag_service.get_index()
        logger.info(f"[知识库检索] 问题：{query}")
        retriever = index.as_retriever(similarity_top_k=3)
        docs = retriever.retrieve(query)

        # 格式化返回检索结果
        if docs:
            result_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知')
                result_parts.append(f"【文档 {i}】来源：{source}\n{doc.text}")
            result = "\n\n".join(result_parts)
            logger.info(f"[知识库检索] 检索到 {len(docs)} 个相关文档")
            logger.debug(f"[知识库检索结果] {result}")
            return result
        else:
            logger.info(f"[知识库检索] 未找到相关文档")
            return "未在知识库中找到相关信息。"
    except Exception as e:
        logger.error(f"[知识库检索] 查询失败: {e}", exc_info=True)
        return f"查询知识库时发生错误: {str(e)}"


@tool
async def get_new_product(limit: int = 3) -> list[ProductResponseDto]:
    """
    获取最新商品，调用的时候 limit 参数取值 3 ~ 5 , 不建议超过 5，不然消息太长
    Args:
        limit: 限制数，即 获取几个新品
    """
    try:
        logger.info(f"[商品查询] 获取最新商品，limit={limit}")
        product_client = await get_product_service_client()
        result = await product_client.get_new_products(limit=limit)
        logger.info(f"[商品查询] 获取到 {len(result)} 个新品")
        if result:
            product_names = [p.name for p in result]
            logger.debug(f"[商品查询结果] {product_names}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 获取新品失败: {e}", exc_info=True)
        return []


@tool
async def search_product(query: str, page_number: int = 1, page_size: int = 3) -> list[ProductResponseDto]:
    """
    根据用户对商品的自然语言描述，搜索商品。 注意，但不支持商品字段属性的过滤，仅仅是针对【商品标题】的关键词和语义搜索。
    Args:
        query: 用户查询，比如 拍照好看的手机、苹果笔记本、送女朋友的礼物
        page_number: 分页的页码
        page_size: 一页的大小（不宜过大，不用超过 5，不然消息太长）
    """
    try:
        logger.info(f"[商品查询] 搜索商品，query={query}, page={page_number}, size={page_size}")
        product_client = await get_product_service_client()
        result = await product_client.search_products(query, page_number, page_size)
        logger.info(f"[商品查询] 搜索到 {len(result)} 个商品")
        if result:
            product_names = [p.name for p in result]
            logger.debug(f"[商品查询结果] {product_names}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 搜索商品失败: {e}", exc_info=True)
        return []


@tool
async def get_product_detail(product_id: int) -> ProductResponseDto | None:
    """
    根据商品ID查询商品详情，用于获取商品的完整信息包括价格、描述、库存、款式等，如果返回为空，说明商品详情获取失败！
    Args:
        product_id: 商品ID
    """
    try:
        logger.info(f"[商品查询] 获取商品详情，product_id={product_id}")
        product_client = await get_product_service_client()
        result = await product_client.get_product_by_id(product_id)
        logger.info(f"[商品查询] 获取商品详情成功！product_name={result.name}")
        return result
    except Exception as e:
        logger.error(f"[商品查询] 获取商品详情失败: {e}", exc_info=True)
        return None
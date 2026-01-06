"""
@File       : chat_tool.py
@Description:

@Time       : 2026/1/6 10:20
@Author     : hcy18
"""
from langchain_core.tools import tool
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
"""工具定义"""

from langgraph.prebuilt import ToolNode

from app.tools.chat_tool import search_product, get_product_detail


def handle_tool_error(error: Exception) -> str:
    """处理工具执行异常，仅返回通用提示"""
    if isinstance(error, ValueError):
        return "工具参数有误，请检查输入参数是否符合要求。"
    return "工具暂时无法使用"


tool_node = ToolNode([search_product, get_product_detail], handle_tool_errors=handle_tool_error)

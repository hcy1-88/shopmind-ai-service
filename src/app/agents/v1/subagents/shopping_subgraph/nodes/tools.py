"""工具定义"""

from langgraph.prebuilt import ToolNode

from app.tools.chat_tool import search_product, get_product_detail
from app.utils.logger import app_logger as logger


def handle_tool_error(error: Exception) -> str:
    """
    处理工具执行出现异常的情况，仅返回通用提示，不暴露内部错误细节。
    :param error: 异常
    :return: 返回给 LLM 的通用错误提示
    """
    logger.error(f"[ShoppingSubgraph] 工具执行异常: {type(error).__name__}: {error}", exc_info=True)
    if isinstance(error, ValueError):
        return "工具参数有误，请检查输入参数是否符合要求。"
    else:
        return "工具暂时无法使用"


tool_node = ToolNode([search_product, get_product_detail], handle_tool_errors=handle_tool_error)

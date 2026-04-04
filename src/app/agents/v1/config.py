"""Agent 全局配置常量定义（RunnableConfig.configurable 键名）"""

from typing import TypedDict

# ======================= Config Keys =======================

MAX_CLARIFICATION_COUNT: str = "max_clarification_count"
MAX_HISTORY_TASK_COUNT: str = "max_history_task_count"
MAX_SEARCH_LOOP: str = "max_search_loop"
MAX_TOOL_LOOP: str = "max_tool_loop"


# ======================= Config TypedDict =======================

class AgentConfig(TypedDict):
    """RunnableConfig.configurable 的类型定义"""

    max_clarification_count: int
    """最大澄清轮次"""

    max_history_task_count: int
    """限制活跃的历史子任务数量"""

    max_search_loop: int
    """filter_node → ready_node 分页循环上限"""

    max_tool_loop: int
    """ready_node → tool_node 工具循环上限"""
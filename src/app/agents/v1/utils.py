"""
@File       : utils.py
@Description: agents v1 公共工具方法

@Time       : 2026/3/23
@Author     : hcy18
"""
from langchain_core.messages import BaseMessage


def build_history_context(messages: list[BaseMessage]) -> str:
    """
    构建历史对话上下文 todo 最佳方案可能是 sub_task 增量的记录任务摘要，所以上下文构建从 sub_task 摘要里取，而不是每次截取最近 5 条

    Args:
        messages: 对话消息列表

    Returns:
        格式化后的历史对话字符串
    """
    if not messages:
        return "无历史消息"

    history_parts = []
    for msg in messages[-5:]:  # 只取最近5条消息
        role = "用户" if msg.type == "human" else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # 截断过长内容
        if len(content) > 200:
            content = content[:200] + "..."
        history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)
"""Agent trace callback for LangGraph - logs node execution path."""

import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig

from app.utils.logger import app_logger as logger


# 关键 state 字段，只打印这些避免日志过大
TRACE_KEY_FIELDS = ["original_query", "rewritten_query", "answer", "sub_tasks", "sub_task_results"]


class AgentTraceCallback(AsyncCallbackHandler):
    """
    LangGraph 节点执行路径追踪回调。

    打印结构化日志：
    - [node_name] IN  → {state_summary}  thread_id=xxx
    - [node_name] OUT → {output_summary}  duration=xxxms  thread_id=xxx

    使用 run_id 作为 key，在 on_chain_end 中查找对应节点信息。
    """

    def __init__(self, thread_id: str):
        """
        Args:
            thread_id: 当前会话 ID，用于关联日志。
        """
        self._thread_id = thread_id
        # key: run_id (UUID), value: {"name": str, "start_time": float}
        self._node_info: dict[str, dict[str, Any]] = {}

    def _summarize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """只提取关键字段，避免打印完整 state"""
        return {k: v for k, v in state.items() if k in TRACE_KEY_FIELDS}

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """节点开始执行时调用"""
        name = kwargs.get("name", "unknown") if kwargs else "unknown"
        self._node_info[str(run_id)] = {
            "name": name,
            "start_time": time.perf_counter(),
        }
        state_summary = self._summarize_state(inputs) if isinstance(inputs, dict) else str(inputs)[:200]
        logger.info(
            f"[{name}] IN  → {state_summary}  thread_id={self._thread_id}",
            extra={"thread_id": self._thread_id, "node": name, "event": "chain_start"},
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """节点执行完成时调用"""
        info = self._node_info.pop(str(run_id), None)
        if info is None:
            return
        name = info["name"]
        duration_ms = round((time.perf_counter() - info["start_time"]) * 1000, 2)
        output_summary = self._summarize_state(outputs) if isinstance(outputs, dict) else str(outputs)[:200]
        logger.info(
            f"[{name}] OUT → {output_summary}  duration={duration_ms}ms  thread_id={self._thread_id}",
            extra={"thread_id": self._thread_id, "node": name, "event": "chain_end", "duration_ms": duration_ms},
        )

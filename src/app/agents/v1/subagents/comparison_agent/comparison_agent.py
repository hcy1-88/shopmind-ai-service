"""
@File       : comparison_agent.py
@Description: 比较 Agent - 处理商品比较场景

@Time       : 2026/3/27
@Author     : hcy18
"""
from typing import Optional, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.v1.schema import ComparisonSubTask, ProductResponseDto, ComparisonSubgraphState
from app.agents.v1.subagents.comparison_agent.nodes.detail_node import detail_node
from app.agents.v1.subagents.comparison_agent.nodes.compare_node import compare_node
from app.utils.logger import app_logger as logger


class ComparisonSubgraph:
    """
    处理 COMPARISON 意图的子图.

    流程:
        START -> detail_node -> compare_node -> END
    """

    _instance: Optional["ComparisonSubgraph"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.graph: CompiledStateGraph | None = None
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ComparisonSubgraph":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_subgraph(self, checkpointer):
        """构建比较子图"""
        # 使用 dict-based state 以更灵活
        comparison_subgraph = StateGraph(ComparisonSubgraphState)

        # 添加节点
        comparison_subgraph.add_node("detail_node", detail_node)
        comparison_subgraph.add_node("compare_node", compare_node)

        # 添加边
        comparison_subgraph.add_edge(START, "detail_node")
        comparison_subgraph.add_edge("detail_node", "compare_node")
        comparison_subgraph.add_edge("compare_node", END)

        # 编译
        compiled = comparison_subgraph.compile(checkpointer=checkpointer)

        # mermaid 可视化
        mermaid_graph = compiled.get_graph().draw_mermaid()
        logger.info(f"[ComparisonSubgraph] mermaid: {mermaid_graph}")

        return compiled

    def build_comparison_subgraph(self, checkpointer):
        """获取编译后的子图"""
        if not self.graph:
            self.graph = self._build_subgraph(checkpointer=checkpointer)
            logger.info("comparison 比较子图初始化成功！")

    def get_graph(self) -> CompiledStateGraph:
        return self.graph

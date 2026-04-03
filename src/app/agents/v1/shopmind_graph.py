"""
@File       : shopmind_graph.py
@Description: 总图的构建 - ShopmindAgentGraph 主图编排

@Time       : 2026/3/26 17:49
@Author     : hcy18
"""
from typing import Optional, TYPE_CHECKING

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.v1.schema import ShopmindAgentState
from app.agents.v1.nodes.query_rewrite_node import query_rewritten_node
from app.agents.v1.nodes.intent_decomposer_node import intent_decomposer_node
from app.agents.v1.nodes.searching_subgraph_node import searching_subgraph_node
from app.agents.v1.nodes.chitchat_node import chitchat_node
from app.agents.v1.nodes.platform_node import platform_node
from app.agents.v1.nodes.aggregator_node import aggregate_node
from app.agents.v1.nodes.comparison_subgraph_node import comparison_subgraph_node
from app.agents.v1.edges.route_to_map_edge import route_to_map_node_edge
from app.utils.logger import app_logger as logger

if TYPE_CHECKING:
    from app.agents.v1.subagents.searching_agent.searching_agent import SearchingSubgraph
    from app.agents.v1.subagents.chitchat_agent import ChitChatService
    from app.agents.v1.subagents.comparison_agent.comparison_service import ComparisonService


class ShopmindAgentGraph:
    """
    Shopmind 主图 - 编排所有节点的主图

    图结构:
        START -> query_rewrite_node -> intent_decomposer_node
                                              |
                              route_to_map_node_edge (Send)
                                    /      |         \\
                    searching_subgraph_node  platform_node  chitchat_node
                                    \\      |         /
                                     sub_task_results (累积)
                                              |
                                        aggregator_node
                                              |
                                             END
    """

    _instance: Optional["ShopmindAgentGraph"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._graph: CompiledStateGraph | None = None
        self.shopping_subgraph: "SearchingSubgraph | None" = None  # 持有子图引用
        self.chitchat_agent: "ChitChatService | None" = None  # 持有 chitchat 引用
        self.comparison_service: "ComparisonService | None" = None  # 持有比较服务引用
        logger.info("ShopmindAgentGraph 初始化成功!")

    def init_graph(self, checkpointer, shopping_subgraph: "SearchingSubgraph", chitchat_agent: "ChitChatService", comparison_service: "ComparisonService"):
        """初始化主图，持有子图引用并构建主图

        Args:
            checkpointer: checkpointer 实例
            shopping_subgraph: Shopping 子图实例
            chitchat_agent: ChitChatService 实例
            comparison_service: ComparisonService 实例
        """
        self.shopping_subgraph = shopping_subgraph
        self.chitchat_agent = chitchat_agent
        self.comparison_service = comparison_service
        self._graph = self._build_graph(checkpointer)

    def get_graph(self) -> CompiledStateGraph:
        """获取编译后的图

        Returns:
            编译后的图
        """
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Call init_graph() first.")
        return self._graph

    def _build_graph(self, checkpointer) -> CompiledStateGraph:
        """构建主图"""
        # 构建父图
        main_graph = StateGraph(ShopmindAgentState)

        # 添加节点
        main_graph.add_node("query_rewrite_node", query_rewritten_node)
        main_graph.add_node("intent_decomposer_node", intent_decomposer_node)
        main_graph.add_node("searching_subgraph_node", searching_subgraph_node)
        main_graph.add_node("platform_node", platform_node)
        main_graph.add_node("chitchat_node", chitchat_node)
        main_graph.add_node("comparison_subgraph_node", comparison_subgraph_node)
        main_graph.add_node("aggregator_node", aggregate_node)

        # 添加边
        main_graph.add_edge(START, "query_rewrite_node")
        main_graph.add_edge("query_rewrite_node", "intent_decomposer_node")

        # 条件边: intent_decomposer -> 并行执行 shopping/platform/chitchat/comparison
        main_graph.add_conditional_edges(
            "intent_decomposer_node",
            route_to_map_node_edge,
            ["searching_subgraph_node", "platform_node", "chitchat_node", "comparison_subgraph_node"]
        )

        # 并行节点执行完后汇聚到聚合节点
        main_graph.add_edge("searching_subgraph_node", "aggregator_node")
        main_graph.add_edge("platform_node", "aggregator_node")
        main_graph.add_edge("chitchat_node", "aggregator_node")
        main_graph.add_edge("comparison_subgraph_node", "aggregator_node")

        # 聚合节点收集结果后结束
        main_graph.add_edge("aggregator_node", END)

        # 编译
        compiled = main_graph.compile(checkpointer=checkpointer)

        # 打印 mermaid 图结构
        mermaid_graph = compiled.get_graph().draw_mermaid()
        logger.info(f"[主Agent]-ShopmindAgentGraph mermaid: {mermaid_graph}")

        return compiled

    @classmethod
    def get_instance(cls) -> "ShopmindAgentGraph":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

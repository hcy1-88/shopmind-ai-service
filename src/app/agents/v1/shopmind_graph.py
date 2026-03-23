"""
@File       : shopmind_graph.py
@Description: 总图的构建 - ShopmindAgentGraph 主图编排

@Time       : 2026/3/26 17:49
@Author     : hcy18
"""
from typing import Optional, Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.v1.schema import ShopmindAgentState
from app.agents.v1.nodes.query_rewrite_node import query_rewritten_node
from app.agents.v1.nodes.intent_decomposer_node import intent_decomposer_node
from app.agents.v1.nodes.shopping_subgraph_node import shopping_subgraph_node
from app.agents.v1.nodes.chitchat_node import chitchat_node
from app.agents.v1.nodes.platform_node import platform_node
from app.agents.v1.nodes.aggregator_node import aggregate_node
from app.agents.v1.edges.route_to_map_edge import route_to_map_node_edge
from app.utils.logger import app_logger as logger


class ShopmindAgentGraph:
    """
    Shopmind 主图 - 编排所有节点的主图

    图结构:
        START -> query_rewrite_node -> intent_decomposer_node
                                              |
                              route_to_map_node_edge (Send)
                                    /      |         \
                    shopping_subgraph_node  platform_node  chitchat_node
                                    \      |         /
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
        self.graph = self.__build_graph()
        logger.info("ShopmindAgentGraph 初始化成功!")

    def __build_graph(self) -> CompiledStateGraph:
        """构建主图"""
        # 创建 StateGraph
        main_graph = StateGraph(ShopmindAgentState)

        # 添加节点
        main_graph.add_node("query_rewrite_node", query_rewritten_node)
        main_graph.add_node("intent_decomposer_node", intent_decomposer_node)
        main_graph.add_node("shopping_subgraph_node", shopping_subgraph_node)
        main_graph.add_node("platform_node", platform_node)
        main_graph.add_node("chitchat_node", chitchat_node)
        main_graph.add_node("aggregator_node", aggregate_node)

        # 添加边
        main_graph.add_edge(START, "query_rewrite_node")
        main_graph.add_edge("query_rewrite_node", "intent_decomposer_node")

        # 条件边: intent_decomposer -> 并行执行 shopping/platform/chitchat
        main_graph.add_conditional_edges(
            "intent_decomposer_node",
            route_to_map_node_edge,
            ["shopping_subgraph_node", "platform_node", "chitchat_node"]
        )

        # 并行节点执行完后汇聚到聚合节点
        main_graph.add_edge("shopping_subgraph_node", "aggregator_node")
        main_graph.add_edge("platform_node", "aggregator_node")
        main_graph.add_edge("chitchat_node", "aggregator_node")

        # 聚合节点收集结果后结束
        main_graph.add_edge("aggregator_node", END)

        # 编译
        compiled = main_graph.compile()

        # 打印 mermaid 图结构
        mermaid_graph = compiled.get_graph().draw_mermaid()
        logger.info(f"ShopmindAgentGraph mermaid: {mermaid_graph}")

        return compiled

    @classmethod
    def get_instance(cls) -> "ShopmindAgentGraph":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_graph(self) -> CompiledStateGraph:
        """获取编译后的图"""
        return self.graph

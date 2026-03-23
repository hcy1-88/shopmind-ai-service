"""
@File       : shopping_agent.py
@Description:

@Time       : 2026/3/23 17:12
@Author     : hcy18
"""
from typing import Optional, Any
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from app.agents.v1.schema import ShoppingSubgraphState
from app.agents.v1.subagents.shopping_subgraph.nodes import (
    dispatcher_node,
    clarifying_node,
    ready_node,
    filter_node,
    generate_node,
)
from app.agents.v1.subagents.shopping_subgraph.edges import (
    route_by_status_edge,
    route_after_ready,
    router_after_filter,
)
from app.agents.v1.subagents.shopping_subgraph.nodes.tools import tool_node
from app.utils.logger import app_logger as logger


class ShoppingSubgraph:
    """
    处理 shopping 意图的子图.

    shopping_subgraph_node:
      │
      ├─── if status == CLARIFYING:
      │       └── 生成澄清问题，返回给用户
      │
      └─── if status == READY:
              └── 执行商品搜索
                      │
                      ├── search_product（关键词搜索）
                      ├── filter（按 filters 过滤）
                      └── generate（生成商品推荐文案）
    """

    _instance: Optional["ShoppingSubgraph"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.graph = self.__build_subgraph()
        logger.info("shopping 导购子图初始化成功！")

    @classmethod
    def get_instance(cls) -> "ShoppingSubgraph":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    def __build_subgraph(self):
        # 图构建
        shopping_subgraph = StateGraph(ShoppingSubgraphState)

        # 添加节点
        shopping_subgraph.add_node("dispatcher_node", dispatcher_node)
        shopping_subgraph.add_node("clarifying_node", clarifying_node)
        shopping_subgraph.add_node("ready_node", ready_node)
        shopping_subgraph.add_node("tool_node", tool_node)
        shopping_subgraph.add_node("filter_node", filter_node)
        shopping_subgraph.add_node("generate_node", generate_node)

        # 添加边
        shopping_subgraph.add_edge(START, "dispatcher_node")
        shopping_subgraph.add_conditional_edges(
            "dispatcher_node",
            route_by_status_edge,
            ["clarifying_node", "ready_node"],
        )
        shopping_subgraph.add_edge("clarifying_node", END)

        shopping_subgraph.add_conditional_edges(
            "ready_node",
            route_after_ready,
            ["tool_node", "filter_node"],
        )
        shopping_subgraph.add_edge("tool_node", "ready_node")

        shopping_subgraph.add_conditional_edges(
            "filter_node",
            router_after_filter,
            ["ready_node", "generate_node"],
        )

        shopping_subgraph.add_edge("generate_node", END)

        # 编译 todo 需要加入 checkpointer 记忆
        shopping_subgraph = shopping_subgraph.compile()

        # mermaid 可视化
        shopping_subgraph_mermaid = shopping_subgraph.get_graph().draw_mermaid()
        logger.info(f"shopping_subgraph_mermaid: {shopping_subgraph_mermaid}")

        return shopping_subgraph


    def get_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """获取编译后的子图"""
        return self.graph

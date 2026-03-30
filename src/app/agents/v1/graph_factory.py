"""
@File       : graph_factory.py
@Description: 图工厂 - 统一管理父图和所有子图的初始化

@Time       : 2026/3/26
@Author     : hcy18
"""
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.v1.shopmind_graph import ShopmindAgentGraph
from app.agents.v1.subagents.searching_agent.searching_agent import SearchingSubgraph
from app.agents.v1.subagents.chitchat_agent import ChitChatService
from app.agents.v1.subagents.comparison_agent.comparison_agent import ComparisonSubgraph
from app.utils.logger import app_logger as logger


class GraphFactory:
    """
    图工厂 - 统一管理父图和所有子图的初始化

    初始化顺序：
    1. 构建 ShoppingSubgraph
    2. 获取 ChitChatService 单例
    3. 构建 ComparisonSubgraph
    4. 初始化 ShopmindAgentGraph 并持有子图引用
    """

    @classmethod
    def build_all(cls, checkpointer: BaseCheckpointSaver) -> ShopmindAgentGraph:
        """
        一次性构建并初始化所有图

        Args:
            checkpointer: checkpointer 实例

        Returns:
            初始化完成的 ShopmindAgentGraph 单例
        """
        logger.info("GraphFactory 开始构建所有图...")

        # 1. 构建 ShoppingSubgraph
        shopping_subgraph = cls._build_shopping_subgraph(checkpointer)

        # 2. 获取 ChitChatService 单例
        chitchat_agent = cls._build_chitchat_agent()

        # 3. 构建 ComparisonSubgraph
        comparison_subgraph = cls._build_comparison_subgraph(checkpointer)

        # 4. 初始化父图
        main_graph = cls._build_main_graph(checkpointer, shopping_subgraph, chitchat_agent, comparison_subgraph)

        logger.info("GraphFactory 所有图构建完成!")
        return main_graph

    @classmethod
    def _build_shopping_subgraph(cls, checkpointer: BaseCheckpointSaver) -> SearchingSubgraph:
        """
        构建 Shopping 子图

        Args:
            checkpointer: checkpointer 实例

        Returns:
            ShoppingSubgraph 单例
        """
        shopping_subgraph = SearchingSubgraph.get_instance()
        shopping_subgraph.build_shopping_subgraph(checkpointer)
        logger.info("ShoppingSubgraph 构建完成")
        return shopping_subgraph

    @classmethod
    def _build_chitchat_agent(cls) -> ChitChatService:
        """
        构建 ChitChatService Agent

        Returns:
            ChitChatService 单例
        """
        chitchat_agent = ChitChatService.get_instance()
        chitchat_agent.build_chitchat_agent()
        logger.info("ChitChatService 构建完成")
        return chitchat_agent

    @classmethod
    def _build_comparison_subgraph(cls, checkpointer: BaseCheckpointSaver) -> ComparisonSubgraph:
        """
        构建 ComparisonSubgraph

        Args:
            checkpointer: checkpointer 实例

        Returns:
            ComparisonSubgraph 单例
        """
        comparison_subgraph = ComparisonSubgraph.get_instance()
        comparison_subgraph.build_comparison_subgraph(checkpointer)
        logger.info("ComparisonSubgraph 构建完成")
        return comparison_subgraph

    @classmethod
    def _build_main_graph(
        cls,
        checkpointer: BaseCheckpointSaver,
        shopping_subgraph: SearchingSubgraph,
        chitchat_agent: ChitChatService,
        comparison_subgraph: ComparisonSubgraph,
    ) -> ShopmindAgentGraph:
        """
        构建主图并持有子图引用

        Args:
            checkpointer: checkpointer 实例
            shopping_subgraph: Shopping 子图实例
            chitchat_agent: ChitChatService 实例
            comparison_subgraph: ComparisonSubgraph 实例

        Returns:
            ShopmindAgentGraph 单例
        """
        main_graph = ShopmindAgentGraph.get_instance()
        main_graph.init_graph(checkpointer, shopping_subgraph, chitchat_agent, comparison_subgraph)
        logger.info("ShopmindAgentGraph 构建完成")
        return main_graph

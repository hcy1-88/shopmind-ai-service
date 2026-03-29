"""
@File       : schema.py
@Description:

@Time       : 2026/3/8 17:26
@Author     : hcy18
"""
import operator
from datetime import datetime
from enum import Enum
from typing import TypedDict, Annotated

from langgraph.graph import add_messages

from app.schemas.page_result_schema import PageResult
from app.schemas.product_response_schema import ProductResponseDto

from app.utils.id_util import gen_id
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    SHOPPING = "SHOPPING"
    PLATFORM = "PLATFORM"
    CHITCHAT = "CHITCHAT"
    COMPARISON = "COMPARISON"


class TaskStatus(str, Enum):
    """子任务执行状态（状态机）"""
    NEW        = "NEW"         # 刚创建
    CLARIFYING = "CLARIFYING"  # 信息不足，等待用户澄清
    READY      = "READY"       # 槽位齐全，可执行搜索
    COMPLETED  = "COMPLETED"   # 执行完成，final_response 已生成
    FAILED     = "FAILED"      # 执行失败


class IntentItem(BaseModel):
    sub_query: str
    intent: IntentCategory
    is_new: bool
    matched_task_id: str | None = None
    extracted_slots: dict = Field(default_factory=dict, description="从用户输入提取的槽位信息")
    explicit_search: bool = Field(default=False, description="用户是否明确表示要立即搜索（如'搜一下'、'就这个了'）")
    is_replace_products: bool = Field(default=False, description="用户是否要求'换一批'（如'换一批'、'换一个'），为 true 时 filter_node 排除 has_recommended_product_ids")


class IntentResponse(BaseModel):
    intent_items: list[IntentItem]


class SubTask(BaseModel):
    """SubTask 基类 - 所有意图类型的通用父类"""
    task_id: str = Field(default_factory=gen_id)
    category: IntentCategory = Field(default=None, description="意图识别的分类")
    sub_query: str = Field(description="重写后的子问题")
    status: TaskStatus
    created_at: datetime = Field(default_factory=datetime.now)
    final_response: str | None = Field(default=None, description="最终响应内容，由各意图处理器填充")


class ShoppingSubTask(SubTask):
    """购物意图的 SubTask"""
    product_category: str | None = Field(default=None, description="商品品类（核心词，表示最小售卖单位），如手机、耳机，核心词只能是一个")
    keywords: list[str] = Field(default_factory=list, description="搜索关键词（扩展词）")
    filters: dict = Field(default_factory=dict, description="过滤条件，如价格区间、颜色等")
    clarification_count: int = Field(default=0, description="已澄清次数，最多3次")
    has_recommended_product_ids: list[int] = Field(default_factory=list, description="已经搜索过的商品，用户要求换一批时有用")
    # 已经使用过的搜索页号
    searched_pages: list[int] = Field(default=[0], description="此搜索任务已经搜索过的页号")
    is_replace_products: bool = Field(default=False, description="是否为换一批场景，为 true 时 filter_node 排除已推荐商品")
    # 一次对话内，调用搜索工具的最大循环次数（filter_node -> ready_node）
    max_search_loop: int


class PlatformSubTask(SubTask):
    """平台规则意图的 SubTask"""
    pass


class ChitchatSubTask(SubTask):
    """闲聊意图的 SubTask"""
    pass


class ComparisonSubTask(SubTask):
    """商品比较意图的 SubTask"""
    # 待比较的商品ID列表（从关联的 ShoppingSubTask.has_recommended_product_ids 获取）
    product_ids: list[int] = Field(default_factory=list, description="待比较的商品ID列表")


class ShopmindAssistantContext(BaseModel):
    llm: BaseChatModel
    # 推理模型，用于 intent_decomposer_node 和 filter_node 等需要严谨推理的节点
    reasoning_llm: BaseChatModel
    thread_id: str
    # 最大澄清轮次
    max_clarification_count: int
    # 限制活跃的历史子任务（防止意图识别时上下文太长）
    max_history_task_count: int
    # 一次对话内，调用搜索工具的循环次数（filter_node -> ready_node）
    max_search_loop: int


class ShopmindAgentState(TypedDict):
    """Agent 状态类型，使用 TypedDict 更灵活，方便后续节点添加状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    rewritten_query: str | None
    sub_tasks: list[SubTask]
    current_tasks: list[SubTask]
    # 聚合器收集任务结果
    sub_task_results: Annotated[list[SubTask], operator.add]
    # Agent 最终发送给用户的回复
    answer: str | None
    streaming_started: bool


## 平台规则节点的状态
class PlatformNodeState(BaseModel):
    sub_task: PlatformSubTask
    messages: Annotated[list[BaseMessage], add_messages]


## 闲聊节点的状态
class ChitChatNodeState(BaseModel):
    """闲聊节点状态"""
    sub_task: ChitchatSubTask
    messages: Annotated[list[BaseMessage], add_messages]


## shopping 节点的状态
class ShoppingNodeState(BaseModel):
    sub_task: ShoppingSubTask
    messages: Annotated[list[BaseMessage], add_messages]


class ComparisonNodeState(BaseModel):
    sub_task: ShoppingSubTask
    messages: Annotated[list[BaseMessage], add_messages]


## ======================= 搜索商品的子图状态 =================
class SearchingSubgraphState(TypedDict):
    # 导购任务
    task: ShoppingSubTask
    # 子图消息
    subgraph_messages: Annotated[list[BaseMessage], add_messages]  # 子图的消息
    # 每次搜索后的结果
    searched_res: Annotated[list[PageResult[list[ProductResponseDto]]], operator.add]
    # 搜到到的商品详情
    searched_details: Annotated[list[ProductResponseDto], operator.add]

    # LLM 语义过滤后需要保留的商品 ID 列表
    filtered_product_ids: list[int]
    # 需要保留的商品详情
    product_after_filter: list[ProductResponseDto]

    # 当前搜索循环次数，用于控制分页上限
    search_count_loop: int

    # 父子图的同名 key 是共享的
    messages: Annotated[list[BaseMessage], operator.add]      # 父图的消息


class FilterResult(BaseModel):
    """filter_node 输出解析器使用的 Pydantic 模型"""
    filtered_product_ids: list[int] = Field(description="过滤后需要保留的商品 ID 列表")


## ======================= 比较商品的子图（基于商品id） ================
class ComparisonSubgraphState(TypedDict):
    task: ComparisonSubTask
    product_ids: list[int]
    product_details: Annotated[list[ProductResponseDto], operator.add]
    subgraph_messages: Annotated[list[BaseMessage], add_messages]

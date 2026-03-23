"""
@File       : schema.py
@Description:

@Time       : 2026/3/8 17:26
@Author     : hcy18
"""
from datetime import datetime
from enum import Enum
from typing import TypedDict, Annotated

from langgraph.graph import add_messages

from app.utils.id_util import gen_id
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    SHOPPING = "SHOPPING"
    PLATFORM = "PLATFORM"
    CHITCHAT = "CHITCHAT"


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


class IntentResponse(BaseModel):
    intent_items: list[IntentItem]


class SubTask(BaseModel):
    """SubTask 基类 - 所有意图类型的通用父类"""
    task_id: str = Field(default_factory=gen_id)
    category: IntentCategory = Field(default=None, description="意图识别的分类")
    original_query: str = Field(description="重写后的子问题")
    status: TaskStatus
    clarification_count: int = Field(default=0, description="已澄清次数，最多3次")
    created_at: datetime = Field(default_factory=datetime.now)
    final_response: str | None = Field(default=None, description="最终响应内容，由各意图处理器填充")


class ShoppingSubTask(SubTask):
    """购物意图的 SubTask"""
    product_category: str | None = Field(default=None, description="商品品类（核心词），如手机、耳机")
    keywords: list[str] = Field(default_factory=list, description="搜索关键词（扩展词）")
    filters: dict = Field(default_factory=dict, description="过滤条件，如价格区间、颜色等")
    has_searched_product_id: list[int] = Field(default_factory=list, description="已经搜索过的商品，用户要求换一批时有用")


class PlatformSubTask(SubTask):
    """平台规则意图的 SubTask"""
    pass


class ChitchatSubTask(SubTask):
    """闲聊意图的 SubTask"""
    pass


class ShopmindAssistantContext(BaseModel):
    llm: BaseChatModel
    thread_id: str
    # 最大澄清轮次
    max_clarification_count: int
    # 限制活跃的历史子任务（防止意图识别时上下文太长）
    max_history_task_count: int


class ShopmindAgentState(TypedDict):
    """Agent 状态类型，使用 TypedDict 更灵活，方便后续节点添加状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    rewritten_query: str | None
    sub_tasks: list[SubTask]
    current_tasks: list[SubTask]


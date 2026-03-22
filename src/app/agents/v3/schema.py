"""
@File       : schema.py
@Description:

@Time       : 2026/3/8 17:26
@Author     : hcy18
"""
from datetime import datetime
from enum import Enum
from typing import Annotated
from app.utils.id_util import gen_id
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    SHOPPING = "SHOPPING"
    PLATFORM = "PLATFORM"
    CHITCHAT = "CHITCHAT"


class TaskStatus(str, Enum):
    """子任务执行状态（状态机）"""
    NEW        = "NEW"         # 刚创建，待 meta_fetcher 处理
    CLARIFYING = "CLARIFYING"  # 信息不足，等待用户澄清
    READY      = "READY"       # 槽位齐全，可执行搜索
    COMPLETED  = "COMPLETED"   # 执行完成，final_response 已生成
    FAILED     = "FAILED"      # 执行失败


class IntentItem(BaseModel):
    sub_query: str
    intent: IntentCategory
    is_new: bool
    matched_task_id: str | None = None


class IntentResponse(BaseModel):
    intent_items: list[IntentItem]


class SubTask(BaseModel):
    task_id: int = Field(default_factory=gen_id)
    category: IntentCategory = Field(default=None)
    original_query: str
    filled_slots: dict = {}
    status: TaskStatus
    created_at: datetime


class ShopmindAssistantContext(BaseModel):
    llm: BaseChatModel
    thread_id: str


class ShopmindAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    rewritten_query: str = Field(default=None, description="重写后的查询（消除指代、补充信息后的完整query）")
    sub_tasks: list[SubTask] = []


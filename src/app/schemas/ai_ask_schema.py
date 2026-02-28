from typing import Optional
from pydantic import Field
from app.schemas.base import CamelCaseModel


class AIAskRequest(CamelCaseModel):
    question: str = Field(..., description="用户的问题")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: str = Field(..., description="会话ID，用于区分不同会话，支持多会话并存")
    product_id: Optional[str] = Field(None, description="商品ID")
    order_id: Optional[str] = Field(None, description="订单ID")
    context: Optional[dict] = Field(None, description="上下文辅助信息")


class AIClearHistoryRequest(CamelCaseModel):
    """清除对话历史请求"""
    session_id: str = Field(..., description="会话ID，如果不传则清除该用户所有会话")


class AIResponse(CamelCaseModel):
    answer: str = Field(..., description="AI的回答")


class UpdateConversationNameRequest(CamelCaseModel):
    """更新对话名称请求"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    name: str = Field(..., description="新对话名称")


class DeleteConversationRequest(CamelCaseModel):
    """删除对话请求"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
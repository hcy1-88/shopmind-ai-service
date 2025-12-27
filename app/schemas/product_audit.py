"""
@File       : product_audit.py
@Description:

@Time       : 2025/12/28 1:50
@Author     : hcy18
"""
from typing import List, Optional
from datetime import datetime
from pydantic import Field
from app.schemas.base import CamelCaseModel


class CheckResult(CamelCaseModel):
    """内容检查结果子模型."""

    passed: bool = Field(
        ...,
        description="是否通过",
    )
    issue_types: List[str] = Field(
        ...,
        description="检测到的问题类型：如 sensitive_words/illegal_content/spam",
    )
    confidence: float = Field(
        ...,
        description="置信度分数 0-1",
    )
    detail: str = Field(
        ...,
        description="详细说明",
    )


class ImageCheckResult(CamelCaseModel):
    """图片审核结果子模型."""

    image_url: str = Field(
        ...,
        description="图片 URL",
    )
    passed: bool = Field(
        ...,
        description="是否通过",
    )
    issue_types: List[str] = Field(
        ...,
        description="检测到的问题类型",
    )
    confidence: float = Field(
        ...,
        description="置信度分数 0-1",
    )
    detail: str = Field(
        ...,
        description="详细说明",
    )


class ProductAuditRequest(CamelCaseModel):
    """商品审核请求模型."""

    product_id: int = Field(
        ...,
        description="商品 ID",
    )
    title: str = Field(
        ...,
        description="商品标题",
    )
    description: str = Field(
        ...,
        description="商品描述",
    )
    cover_image: str = Field(
        ...,
        description="封面图片 URL",
    )
    detail_images: List[str] = Field(
        ...,
        description="详情图片 URL 列表",
    )
    category_id: int = Field(
        ...,
        description="商品分类 ID",
    )


class ProductAuditResponse(CamelCaseModel):
    """商品审核响应模型."""

    audit_status: str = Field(
        ...,
        description="审核状态：approved/rejected",
    )
    title_check_result: CheckResult = Field(
        ...,
        description="标题审核结果",
    )
    image_check_results: List[ImageCheckResult] = Field(
        ...,
        description="图片审核结果列表",
    )
    reject_reason: Optional[str] = Field(
        default=None,
        description="拒绝原因",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="修改建议",
    )
    audit_time: datetime = Field(
        ...,
        description="审核时间",
    )
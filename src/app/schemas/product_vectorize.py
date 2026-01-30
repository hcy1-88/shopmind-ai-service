"""
@File       : product_vectorize.py
@Description:

@Time       : 2025/12/28 3:53
@Author     : hcy18
"""
from decimal import Decimal
from typing import List, Optional
from pydantic import Field

from app.schemas.base import CamelCaseModel


class VectorizeProductRequest(CamelCaseModel):
    """商品向量化请求模型."""

    product_id: int = Field(
        ...,
        description="商品 ID",
    )
    title: str = Field(
        ...,
        description="商品标题",
    )
    description: Optional[str] = Field(
        None,
        description="商品描述",
    )
    ai_summary: str = Field(
        ...,
        description="AI 摘要",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="商品标签列表",
    )



class VectorizeProductResponse(CamelCaseModel):
    """商品向量化响应模型."""

    product_id: int = Field(
        ...,
        description="商品 ID",
    )
    success: bool = Field(
        ...,
        description="向量化是否成功",
    )
    vector_id: str = Field(
        ...,
        description="向量 ID（存储在向量数据库中的 ID）",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（如果失败）",
    )


class DeleteVectorRequest(CamelCaseModel):
    """删除商品向量请求模型."""

    product_id: int = Field(
        ...,
        description="商品 ID",
    )


class DeleteVectorResponse(CamelCaseModel):
    """删除商品向量响应模型."""

    product_id: int = Field(
        ...,
        description="商品 ID",
    )
    success: bool = Field(
        ...,
        description="删除是否成功",
    )
    deleted_count: int = Field(
        ...,
        description="删除的记录数",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（如果失败）",
    )
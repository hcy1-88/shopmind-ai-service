"""
@File       : product_tag.py
@Description:

@Time       : 2025/12/28 2:39
@Author     : hcy18
"""
from typing import List, Optional
from pydantic import Field

from app.schemas.base import CamelCaseModel


class GenerateTagsRequest(CamelCaseModel):
    """生成标签请求模型."""

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
    category_id: int = Field(
        ...,
        description="商品分类 ID",
    )
    image_urls: Optional[List[str]] = Field(
        default=None,
        description="图片 URL 列表（可选，用于图像识别）",
    )


class TagInfo(CamelCaseModel):
    """标签信息子模型."""

    name: str = Field(
        ...,
        description="标签名称",
    )
    color: str = Field(
        ...,
        description="推荐的颜色（UI 显示用），是 十六进制，如 '#ff4444'",
    )


class GenerateTagsResponse(CamelCaseModel):
    """生成标签响应模型."""

    tags: List[TagInfo] = Field(
        ...,
        description="生成的标签列表",
    )
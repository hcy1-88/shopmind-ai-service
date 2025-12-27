"""Pydantic schemas module."""

from app.schemas.base import CamelCaseModel
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.product_description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.product_summary import (
    SummaryGenerateRequest,
    SummaryGenerateResponse,
)
from app.schemas.product_title_check import TitleCheckRequest, TitleCheckResponse
from app.schemas.result_context import ResultContext

__all__ = [
    # 基类
    "CamelCaseModel",
    # 统一返回类型
    "ResultContext",
    # 商品摘要
    "SummaryGenerateRequest",
    "SummaryGenerateResponse",
    # 商品描述
    "DescriptionGenerateRequest",
    "DescriptionGenerateResponse",
    # 标题检查
    "TitleCheckRequest",
    "TitleCheckResponse",
    # 图片检查
    "ImageCheckRequest",
    "ImageCheckResponse",
]

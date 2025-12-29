"""商品摘要生成相关的 Pydantic 模型."""

from decimal import Decimal
from pydantic import Field

from src.app.schemas.base import CamelCaseModel


class SummaryGenerateRequest(CamelCaseModel):
    """商品摘要生成请求模型."""

    product_id: int = Field(
            ...,
            description="商品 ID",
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="商品标题",
        examples=["高品质纯棉T恤 男女通用 透气舒适"],
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="商品描述",
        examples=["这款高品质纯棉T恤采用100%纯棉面料，透气舒适，柔软亲肤..."],
    )

    price: Decimal = Field(
        ...,
        description="商品价格",
    )
    category_id: int = Field(
        ...,
        description="商品分类 ID",
    )


class SummaryGenerateResponse(CamelCaseModel):
    """商品摘要生成响应模型."""

    summary: str = Field(
        ...,
        description="AI 生成的商品摘要（200字以内）",
    )

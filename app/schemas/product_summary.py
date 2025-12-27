"""商品摘要生成相关的 Pydantic 模型."""

from pydantic import Field

from app.schemas.base import CamelCaseModel


class SummaryGenerateRequest(CamelCaseModel):
    """商品摘要生成请求模型."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="商品标题",
        examples=["高品质纯棉T恤"],
    )
    image_urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="商品图片URL列表",
        examples=[["https://example.com/product1.jpg", "https://example.com/product2.jpg"]],
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="商品类目",
        examples=["服装/T恤"],
    )


class SummaryGenerateResponse(CamelCaseModel):
    """商品摘要生成响应模型."""

    ai_summary: str = Field(
        ...,
        description="生成的商品摘要（最多200字）",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "aiSummary": "高品质纯棉T恤，采用100%纯棉面料，透气舒适。精致剪裁，版型简约大方，适合日常穿搭。",
                },
            ],
        }
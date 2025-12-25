"""商品描述生成相关的 Pydantic 模型."""

from pydantic import BaseModel, Field


class DescriptionGenerateRequest(BaseModel):
    """商品描述生成请求模型."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="商品标题",
        examples=["高品质纯棉T恤"],
    )
    imageUrls: list[str] = Field(
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


class DescriptionGenerateResponse(BaseModel):
    """商品描述生成响应模型."""

    description: str = Field(
        ...,
        description="生成的商品描述",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "description": "这款高品质纯棉T恤采用100%纯棉面料，透气舒适，柔软亲肤。精致剪裁，版型简约大方，适合多种场合穿搭。优质做工，耐洗耐穿，是您衣橱中的百搭单品。",
                },
            ],
        }

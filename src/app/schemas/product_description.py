"""商品描述生成相关的 Pydantic 模型."""

from pydantic import Field

from src.app.schemas.base import CamelCaseModel


class DescriptionGenerateRequest(CamelCaseModel):
    """商品描述生成请求模型."""

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



class DescriptionGenerateResponse(CamelCaseModel):
    """商品描述生成响应模型."""

    description: str = Field(
        ...,
        description="生成的商品描述",
    )


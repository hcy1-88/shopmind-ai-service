"""图片检查相关的 Pydantic 模型."""

from pydantic import Field, HttpUrl

from app.schemas.base import CamelCaseModel


class ImageCheckRequest(CamelCaseModel):
    """图片检查请求模型."""

    image_url: str = Field(
        ...,
        description="图片URL（http/https）或base64编码的图片",
        examples=["https://example.com/product.jpg"],
    )


class ImageCheckResponse(CamelCaseModel):
    """图片检查响应模型."""

    valid: bool = Field(
        ...,
        description="图片是否合规",
    )
    reason: str | None = Field(
        default=None,
        description="不合规原因（如果不合规）",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "valid": False,
                    "reason": "图片包含不适当内容",
                },
                {
                    "valid": True,
                    "reason": None,
                },
            ],
        }

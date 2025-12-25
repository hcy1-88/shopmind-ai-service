"""图片检查相关的 Pydantic 模型."""

from pydantic import BaseModel, Field, HttpUrl


class ImageCheckRequest(BaseModel):
    """图片检查请求模型."""

    imageUrl: str = Field(
        ...,
        description="图片URL（http/https）或base64编码的图片",
        examples=["https://example.com/product.jpg"],
    )


class ImageCheckResponse(BaseModel):
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

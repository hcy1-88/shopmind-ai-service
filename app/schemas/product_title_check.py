"""标题检查相关的 Pydantic 模型."""

from pydantic import BaseModel, Field


class TitleCheckRequest(BaseModel):
    """标题检查请求模型."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="待检查的商品标题",
        examples=["高品质纯棉T恤 男女通用 透气舒适"],
    )


class TitleCheckResponse(BaseModel):
    """标题检查响应模型."""

    valid: bool = Field(
        ...,
        description="标题是否合规",
    )
    reason: str | None = Field(
        default=None,
        description="不合规原因（如果不合规）",
    )
    suggestions: list[str] | None = Field(
        default=None,
        description="改进建议（如果适用）",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "valid": False,
                    "reason": "包含夸大宣传词汇",
                    "suggestions": [
                        "移除'史上最好'等绝对化用语",
                        "使用更客观的描述词",
                    ],
                },
                {
                    "valid": True,
                    "reason": None,
                    "suggestions": None,
                },
            ],
        }

"""商品 AI 辅助服务."""

from typing import Optional

from app.chains.description_generation_chain import DescriptionGenerationChain
from app.chains.summary_generation_chain import SummaryGenerationChain
from app.chains.image_check_chain import ImageCheckChain
from app.chains.title_check_chain import TitleCheckChain
from app.schemas.product_description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.product_summary import SummaryGenerateRequest, SummaryGenerateResponse
from app.schemas.product_title_check import TitleCheckRequest, TitleCheckResponse
from app.utils.logger import app_logger as logger


class ProductAIService:
    """商品 AI 辅助服务."""

    _instance: Optional["ProductAIService"] = None

    def __init__(self):
        """Initialize product AI service."""
        # 使用单例 chain
        self.title_check_chain = TitleCheckChain.get_instance()
        self.image_check_chain = ImageCheckChain.get_instance()
        self.description_chain = DescriptionGenerationChain.get_instance()
        self.summary_chain = SummaryGenerationChain.get_instance()

    @classmethod
    def get_instance(cls) -> "ProductAIService":
        """
        获取 ProductAIService 单例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def check_title(self, request: TitleCheckRequest) -> TitleCheckResponse:
        """
        检测标题合规性.

        Args:
            request: 标题

        Returns:
            Title check response
        """
        try:
            logger.info(
                "Checking title compliance",
                extra={"title": request.title[:50]},
            )

            # Run chain
            result = await self.title_check_chain.check(request.title)

            # Convert to response
            response = TitleCheckResponse(
                valid=result.get("valid", False),
                reason=result.get("reason"),
                suggestions=result.get("suggestions") or [],
            )

            logger.info(
                "标题检查完毕",
                extra={
                    "title": request.title[:50],
                    "valid": response.valid,
                },
            )

            return response

        except Exception as e:
            logger.error(f"Error in title check service: {e}")
            # Return error response
            return TitleCheckResponse(
                valid=False,
                reason=f"审核服务异常: {str(e)}",
                suggestions=["请稍后重试"],
            )

    async def check_image(self, request: ImageCheckRequest) -> ImageCheckResponse:
        """
        检测图片合规性.

        Args:
            request: 图片

        Returns:
            Image check response
        """
        try:
            logger.info(
                "Checking image compliance",
                extra={"image_url": request.image_url[:100]},
            )

            # Run chain
            result = await self.image_check_chain.check(request.image_url)

            # Convert to response
            response = ImageCheckResponse(
                valid=result.get("valid", False),
                reason=result.get("reason"),
            )

            logger.info(
                "Image check completed",
                extra={
                    "image_url": request.image_url[:100],
                    "valid": response.valid,
                },
            )

            return response

        except Exception as e:
            logger.error(f"Error in image check service: {e}")
            # Return error response
            return ImageCheckResponse(
                valid=False,
                reason=f"审核服务异常: {str(e)}",
            )

    async def generate_description(
        self,
        request: DescriptionGenerateRequest,
    ) -> DescriptionGenerateResponse:
        """
        生成商品描述.

        Args:
            request: Description generation request

        Returns:
            Description generation response
        """
        try:
            logger.info(
                "生成商品描述",
                extra={
                    "title": request.title[:50],
                    "image_count": len(request.image_urls),
                },
            )

            # Run chain - 新的描述生成链只需要 title 和 image_urls
            description = await self.description_chain.generate(
                title=request.title,
                image_urls=request.image_urls,
            )

            # Create response
            response = DescriptionGenerateResponse(description=description)

            logger.info(
                "Description generation completed",
                extra={
                    "title": request.title[:50],
                    "description_length": len(description),
                },
            )

            return response

        except Exception as e:
            logger.error(f"Error in description generation service: {e}")
            # Return fallback response
            return DescriptionGenerateResponse(
                description=f"抱歉，描述生成服务暂时不可用。商品：{request.title}",
            )

    async def generate_summary(
        self,
        request: SummaryGenerateRequest,
    ) -> SummaryGenerateResponse:
        """
        生成商品摘要.

        Args:
            request: 包含 title 和 description 的请求

        Returns:
            包含 summary 的响应
        """
        try:
            logger.info(
                "生成商品摘要",
                extra={
                    "title": request.title[:50],
                    "description_length": len(request.description),
                },
            )

            # Run chain - 只需要 title 和 description，返回字符串
            summary = await self.summary_chain.generate(
                title=request.title,
                description=request.description,
            )

            # Create response
            response = SummaryGenerateResponse(summary=summary)

            logger.info(
                "商品摘要生成完成",
                extra={
                    "title": request.title[:50],
                    "summary_length": len(summary),
                },
            )

            return response

        except Exception as e:
            logger.error(f"Error in summary generation service: {e}", exc_info=True)
            # Return fallback response
            return SummaryGenerateResponse(
                summary=f"抱歉，摘要生成服务暂时不可用。商品：{request.title}",
            )

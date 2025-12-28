"""商品 AI 辅助服务."""

from app.chains.product.description_generation_chain import DescriptionGenerationChain
from app.chains.product.summary_generation_chain import SummaryGenerationChain
from app.chains.product.image_check_chain import ImageCheckChain
from app.chains.product.tag_generator_chain import ProductTagGenChain
from app.chains.product.title_check_chain import TitleCheckChain
from app.schemas.product_description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.product_summary import SummaryGenerateRequest, SummaryGenerateResponse
from app.schemas.product_tag import GenerateTagsRequest, GenerateTagsResponse
from app.schemas.product_title_check import TitleCheckRequest, TitleCheckResponse
from app.utils.logger import app_logger as logger


class ProductAIService:
    """商品 AI 辅助服务."""

    @staticmethod
    async def check_title(request: TitleCheckRequest) -> TitleCheckResponse:
        """
        检测标题合规性.

        Args:
            request: 标题

        Returns:
            Title check response
        """
        logger.info(
            "Checking title compliance",
            extra={"title": request.title[:50]},
        )

        # Run chain
        result = await TitleCheckChain.get_instance().check(request.title)

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

    @staticmethod
    async def check_image(request: ImageCheckRequest) -> ImageCheckResponse:
        """
        检测图片合规性.

        Args:
            request: 图片

        Returns:
            Image check response
        """
        logger.info(
            "Checking image compliance",
            extra={"image_url": request.image_url[:100]},
        )

        # Run chain
        result = await ImageCheckChain.get_instance().check(request.image_url)

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

    @staticmethod
    async def generate_description(
        request: DescriptionGenerateRequest,
    ) -> DescriptionGenerateResponse:
        """
        生成商品描述.

        Args:
            request: Description generation request

        Returns:
            Description generation response
        """
        logger.info(
            "生成商品描述",
            extra={
                "title": request.title[:50],
                "image_count": len(request.image_urls),
            },
        )

        # Run chain - 新的描述生成链只需要 title 和 image_urls
        description = await DescriptionGenerationChain.get_instance().generate(
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

    @staticmethod
    async def generate_summary(
        request: SummaryGenerateRequest,
    ) -> SummaryGenerateResponse:
        """
        生成商品摘要.

        Args:
            request: 包含 title 和 description 的请求

        Returns:
            包含 summary 的响应
        """
        logger.info(
            "生成商品摘要",
            extra={
                "title": request.title[:50],
                "description_length": len(request.description),
            },
        )

        # Run chain - 只需要 title 和 description，返回字符串
        summary = await SummaryGenerationChain.get_instance().generate(
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

    @staticmethod
    async def generate_product_tags(
        request: GenerateTagsRequest,
    ) -> GenerateTagsResponse:
        """
        生成商品标签
        """
        res = await ProductTagGenChain.get_instance().generate(request)
        logger.info("商品标签生成完毕，商品 id：%s", request.product_id)
        return res
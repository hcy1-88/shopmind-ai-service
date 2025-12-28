"""AI service routers for FastAPI."""

from fastapi import APIRouter, Depends, status

from app.schemas.product_description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.product_tag import GenerateTagsResponse, GenerateTagsRequest
from app.schemas.result_context import ResultContext
from app.schemas.product_summary import SummaryGenerateRequest, SummaryGenerateResponse
from app.schemas.product_title_check import TitleCheckRequest, TitleCheckResponse
from app.services.product_ai_service import ProductAIService
from app.utils.logger import app_logger as logger

# Create router
router = APIRouter(
    prefix="/ai",
    tags=["AI Services"],
)


def get_product_ai_service() -> ProductAIService:
    """获取 ProductAIService 单例实例（依赖注入）."""
    return ProductAIService.get_instance()


@router.get(
    "/health",
    response_model=ResultContext[dict],
    summary="Health check",
    description="Check if AI service is healthy",
)
async def health_check() -> ResultContext[dict]:
    """
    Health check endpoint.

    Returns:
        ResultContext containing health status
    """
    return ResultContext.ok(
        data={
            "status": "healthy",
            "service": "ai-service",
        },
        message="服务运行正常",
    )


@router.post(
    "/title-check",
    response_model=ResultContext[TitleCheckResponse],
    summary="检查商品标题是否合规",
    description="检查商品标题是否合规，不能有违规字词",
)
async def check_title(
    request: TitleCheckRequest,
    service: ProductAIService = Depends(get_product_ai_service),
) -> ResultContext[TitleCheckResponse]:
    """
    检查商品标题是否合规.

    Args:
        request: 商品标题检查请求
        service: ProductAIService 实例（依赖注入）

    Returns:
        检查结果
    """
    logger.info(
        "Received title check request",
        extra={"title": request.title[:50]},
    )

    response = await service.check_title(request)

    return ResultContext.ok(data=response, message="标题检查完成")


@router.post(
    "/image-check",
    response_model=ResultContext[ImageCheckResponse],
    summary="检查图片合规性",
    description="检查图片是否符合平台标准",
)
async def check_image(
    request: ImageCheckRequest,
    service: ProductAIService = Depends(get_product_ai_service),
) -> ResultContext[ImageCheckResponse]:
    """
    检查图片合规性

    Args:
        request: 请求体
        service: ProductAIService 实例（依赖注入）

    Returns:
        检查结果
    """
    logger.info(
        "Received image check request",
        extra={"image_url": request.image_url[:100]},
    )

    response = await service.check_image(request)

    return ResultContext.ok(data=response, message="图片检查完成")


@router.post(
    "/description-generate",
    response_model=ResultContext[DescriptionGenerateResponse],
    summary="生成商品描述",
    description="根据商品标题和图片生成详细的营销性商品描述",
)
async def generate_description(
    request: DescriptionGenerateRequest,
    service: ProductAIService = Depends(get_product_ai_service),
) -> ResultContext[DescriptionGenerateResponse]:
    """
    生成一段描述.

    Args:
        request: 生成描述的请求体
        service: ProductAIService 实例（依赖注入）

    Returns:
        检查结果
    """
    logger.info(
        "Received description generation request",
        extra={
            "title": request.title[:50],
            "image_count": len(request.image_urls),
        },
    )

    response = await service.generate_description(request)

    return ResultContext.ok(
        data=response, message="商品描述生成完成"
    )


@router.post(
    "/generate/product-summary",
    response_model=ResultContext[SummaryGenerateResponse],
    summary="生成商品摘要",
    description="根据完整的商品信息（标题、描述）生成简洁摘要",
)
async def generate_summary(
    request: SummaryGenerateRequest,
    service: ProductAIService = Depends(get_product_ai_service),
) -> ResultContext[SummaryGenerateResponse]:
    """
    生成商品摘要

    Args:
        request: 包含完整商品信息的请求体
        service: ProductAIService 实例（依赖注入）

    Returns:
        商品摘要
    """
    logger.info(
        "摘要生成请求：",
        extra={
            "product_id": request.product_id,
            "title": request.title[:50],
            "description": request.description[:50],
        },
    )

    response = await service.generate_summary(request)

    return ResultContext.ok(
        data=response, message="商品摘要生成完成"
    )




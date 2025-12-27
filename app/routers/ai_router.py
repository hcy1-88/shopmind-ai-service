"""AI service routers for FastAPI."""

from fastapi import APIRouter, status

from app.schemas.description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.result_context import ResultContext
from app.schemas.summary import SummaryGenerateRequest, SummaryGenerateResponse
from app.schemas.title_check import TitleCheckRequest, TitleCheckResponse
from app.services.product_ai_service import ProductAIService
from app.utils.logger import app_logger as logger

# Create router
router = APIRouter(
    prefix="/ai",
    tags=["AI Services"],
)

# 延迟初始化：不在模块级别初始化，避免在 Nacos 连接之前触发
# product_ai_service 将在第一次调用时通过 get_product_ai_service() 获取
def get_product_ai_service() -> ProductAIService:
    """获取 ProductAIService 单例实例（延迟初始化）."""
    return ProductAIService.get_instance()


@router.post(
    "/title-check",
    response_model=ResultContext[TitleCheckResponse],
    summary="检查商品标题是否合规",
    description="检查商品标题是否合规，不能有违规字词",
    status_code=status.HTTP_200_OK,
)
async def check_title(
    request: TitleCheckRequest,
) -> ResultContext[TitleCheckResponse]:
    """
    检查商品标题是否合规.

    Args:
        request: 商品标题检查请求

    Returns:
        检查结果
    """
    try:
        logger.info(
            "Received title check request",
            extra={"title": request.title[:50]},
        )

        service = get_product_ai_service()
        response = await service.check_title(request)

        return ResultContext.ok(data=response, message="标题检查完成")

    except Exception as e:
        logger.error(f"Error in title check endpoint: {e}")
        return ResultContext.fail(
            message=f"标题检查服务错误: {str(e)}",
            code="SYS9999",
        )


@router.post(
    "/image-check",
    response_model=ResultContext[ImageCheckResponse],
    summary="检查图片合规性",
    description="检查图片是否符合平台标准",
    status_code=status.HTTP_200_OK,
)
async def check_image(
    request: ImageCheckRequest,
) -> ResultContext[ImageCheckResponse]:
    """
    检查图片合规性

    Args:
        request: 请求体

    Returns:
        检查结果
    """
    try:
        logger.info(
            "Received image check request",
            extra={"image_url": request.imageUrl[:100]},
        )

        service = get_product_ai_service()
        response = await service.check_image(request)

        return ResultContext.ok(data=response, message="图片检查完成")

    except Exception as e:
        logger.error(f"Error in image check endpoint: {e}")
        return ResultContext.fail(
            message=f"图片检查服务错误: {str(e)}",
            code="SYS9999",
        )


@router.post(
    "/description-generate",
    response_model=ResultContext[DescriptionGenerateResponse],
    summary="生成商品描述",
    description="根据商品的标题、图片、分类 生成描述，图片可选",
    status_code=status.HTTP_200_OK,
)
async def generate_description(
    request: DescriptionGenerateRequest,
) -> ResultContext[DescriptionGenerateResponse]:
    """
    生成一段描述.

    Args:
        request: 生成描述的请求体

    Returns:
        检查结果
    """
    try:
        logger.info(
            "Received description generation request",
            extra={
                "title": request.title[:50],
                "category": request.category,
                "image_count": len(request.imageUrls),
            },
        )

        service = get_product_ai_service()
        response = await service.generate_description(request)

        return ResultContext.ok(
            data=response, message="商品描述生成完成"
        )

    except Exception as e:
        logger.error(f"Error in description generation endpoint: {e}")
        return ResultContext.fail(
            message=f"商品描述生成服务错误: {str(e)}",
            code="SYS9999",
        )


@router.post(
    "/summary-generate",
    response_model=ResultContext[SummaryGenerateResponse],
    summary="生成商品摘要",
    description="根据商品的标题、图片、分类 生成简洁摘要（最多200字），图片可选",
    status_code=status.HTTP_200_OK,
)
async def generate_summary(
    request: SummaryGenerateRequest,
) -> ResultContext[SummaryGenerateResponse]:
    """
    生成商品摘要.

    Args:
        request: 生成摘要的请求体

    Returns:
        生成结果
    """
    try:
        logger.info(
            "Received summary generation request",
            extra={
                "title": request.title[:50],
                "category": request.category,
                "image_count": len(request.imageUrls),
            },
        )

        service = get_product_ai_service()
        response = await service.generate_summary(request)

        return ResultContext.ok(
            data=response, message="商品摘要生成完成"
        )

    except Exception as e:
        logger.error(f"Error in summary generation endpoint: {e}")
        return ResultContext.fail(
            message=f"商品摘要生成服务错误: {str(e)}",
            code="SYS9999",
        )


@router.get(
    "/health",
    response_model=ResultContext[dict],
    summary="Health check",
    description="Check if AI service is healthy",
    status_code=status.HTTP_200_OK,
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

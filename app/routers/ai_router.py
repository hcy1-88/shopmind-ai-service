"""AI service routers for FastAPI."""

from fastapi import APIRouter, status

from app.schemas.description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.result_context import ResultContext
from app.schemas.title_check import TitleCheckRequest, TitleCheckResponse
from app.services.product_ai_service import ProductAIService
from app.utils.logger import app_logger as logger

# Create router
router = APIRouter(
    prefix="/ai",
    tags=["AI Services"],
)

# Initialize unified product AI service (使用单例模式)
product_ai_service = ProductAIService.get_instance()


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

        response = await product_ai_service.check_title(request)

        return ResultContext.success(data=response, message="标题检查完成")

    except Exception as e:
        logger.error(f"Error in title check endpoint: {e}")
        return ResultContext.fail(
            message=f"标题检查服务错误: {str(e)}",
            code="SYS9999",
        )


@router.post(
    "/image-check",
    response_model=ResultContext[ImageCheckResponse],
    summary="Check image compliance",
    description="Check if product image complies with platform standards",
    status_code=status.HTTP_200_OK,
)
async def check_image(
    request: ImageCheckRequest,
) -> ResultContext[ImageCheckResponse]:
    """
    Check product image compliance.

    Args:
        request: Image check request containing the image URL

    Returns:
        ResultContext containing ImageCheckResponse with validation result
    """
    try:
        logger.info(
            "Received image check request",
            extra={"image_url": request.imageUrl[:100]},
        )

        response = await product_ai_service.check_image(request)

        return ResultContext.success(data=response, message="图片检查完成")

    except Exception as e:
        logger.error(f"Error in image check endpoint: {e}")
        return ResultContext.fail(
            message=f"图片检查服务错误: {str(e)}",
            code="SYS9999",
        )


@router.post(
    "/description-generate",
    response_model=ResultContext[DescriptionGenerateResponse],
    summary="Generate product description",
    description="Generate attractive product description based on title, images, and category",
    status_code=status.HTTP_200_OK,
)
async def generate_description(
    request: DescriptionGenerateRequest,
) -> ResultContext[DescriptionGenerateResponse]:
    """
    Generate product description.

    Args:
        request: Description generation request

    Returns:
        ResultContext containing DescriptionGenerateResponse with generated description
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

        response = await product_ai_service.generate_description(request)

        return ResultContext.success(
            data=response, message="商品描述生成完成"
        )

    except Exception as e:
        logger.error(f"Error in description generation endpoint: {e}")
        return ResultContext.fail(
            message=f"商品描述生成服务错误: {str(e)}",
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
    return ResultContext.success(
        data={
            "status": "healthy",
            "service": "ai-service",
        },
        message="服务运行正常",
    )

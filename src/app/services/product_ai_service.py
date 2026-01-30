"""商品 AI 辅助服务."""
from app.chains.product.audit_chain import ProductAuditChain
from app.chains.product.description_generator_chain import DescriptionGenerationChain
from app.chains.product.search_keyword_chain import SearchKeywordEnhanceChain
from app.chains.product.summary_generator_chain import SummaryGenerationChain
from app.chains.product.image_check_chain import ImageCheckChain
from app.chains.product.tag_generator_chain import ProductTagGenChain
from app.chains.product.title_check_chain import TitleCheckChain
from app.schemas.product_audit import ProductAuditRequest, ProductAuditResponse
from app.schemas.product_description import (
    DescriptionGenerateRequest,
    DescriptionGenerateResponse,
)
from app.schemas.image_check import ImageCheckRequest, ImageCheckResponse
from app.schemas.product_summary import SummaryGenerateRequest, SummaryGenerateResponse
from app.schemas.product_tag import GenerateTagsRequest, GenerateTagsResponse
from app.schemas.product_title_check import TitleCheckRequest, TitleCheckResponse
from app.schemas.search_schema import SearchKeyWordEnhanceRequest, SearchKeywordEnhanceResponse
from app.schemas.product_vectorize import DeleteVectorRequest, DeleteVectorResponse
from app.vector_store.product_collection import get_product_collection
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
        response = await TitleCheckChain.get_instance().generate(request)

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
        response = await ImageCheckChain.get_instance().generate(request)

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
        resp = await DescriptionGenerationChain.get_instance().generate(request)

        logger.info(
            "Description generation completed",
            extra={
                "title": request.title[:50],
                "description_length": len(resp.description),
            },
        )

        return resp

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
        response = await SummaryGenerationChain.get_instance().generate(request)

        logger.info(
            "商品摘要生成完成",
            extra={
                "title": request.title[:50],
                "summary_length": len(response.summary),
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

    @staticmethod
    async def audit_product(request: ProductAuditRequest) -> ProductAuditResponse:
        """审核商品"""
        res = await ProductAuditChain.get_instance().generate(request)
        logger.info(f"商品 id：{request.product_id} 审核完毕，审核状态为：{res.audit_status}")
        return res


    @staticmethod
    async def enhance_keyword(request: SearchKeyWordEnhanceRequest) -> SearchKeywordEnhanceResponse:
        """增强搜索词"""
        res = await SearchKeywordEnhanceChain.get_instance().generate(request)
        logger.info(f"用户搜索词: {request.keyword}， 增强后：{res.core_words} and {res.expand_words}")
        return res

    @staticmethod
    async def delete_product_vector(request: DeleteVectorRequest) -> DeleteVectorResponse:
        """
        删除商品向量

        Args:
            request: 包含商品 ID 的请求

        Returns:
            删除结果
        """
        try:
            logger.info(f"开始删除商品向量，product_id: {request.product_id}")

            collection = get_product_collection()

            # 根据 product_id 删除记录
            delete_result = collection.delete(f"product_id == {request.product_id}")

            # 刷新 collection 以确保数据持久化
            collection.flush()

            deleted_count = delete_result.delete_count

            logger.info(f"商品 {request.product_id} 向量删除成功，删除记录数: {deleted_count}")

            return DeleteVectorResponse(
                product_id=request.product_id,
                success=True,
                deleted_count=deleted_count,
                error_message=None,
            )

        except Exception as e:
            error_msg = f"删除商品向量失败: {str(e)}"
            logger.error(
                f"商品 {request.product_id} 向量删除失败: {e}",
                exc_info=True
            )

            return DeleteVectorResponse(
                product_id=request.product_id,
                success=False,
                deleted_count=0,
                error_message=error_msg,
            )
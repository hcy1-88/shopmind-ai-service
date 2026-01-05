"""商品向量化服务."""

from typing import Optional

from app.schemas.product_vectorize import VectorizeProductRequest, VectorizeProductResponse
from app.services.embedding_service import get_embedding_service
from app.vector_store.product_collection import get_product_collection
from app.utils.logger import app_logger as logger


class ProductVectorizeService:
    """商品向量化服务."""

    _instance: Optional["ProductVectorizeService"] = None

    @staticmethod
    async def vectorize_product(request: VectorizeProductRequest) -> VectorizeProductResponse:
        """
        将商品向量化并存储到 Milvus.

        Args:
            request: 商品向量化请求

        Returns:
            向量化响应
        """
        try:
            logger.info(f"开始向量化商品，product_id: {request.product_id}")

            # 1. 构建完整文本：标题 + 描述 + 摘要 + 标签
            text_parts = [
                f"商品标题：{request.title}",
                f"商品描述：{request.description}",
                f"AI 摘要：{request.ai_summary}",
            ]

            # 添加标签
            if request.tags:
                tags_text = "，".join(request.tags)
                text_parts.append(f"商品标签：{tags_text}")

            full_text = "\n".join(text_parts)

            logger.debug(
                f"商品 {request.product_id} 组合文本长度: {len(full_text)} 字符"
            )

            # 2. 使用 embedding service 进行文本嵌入
            embedding_service = get_embedding_service()
            embedding_vector = await embedding_service.embed_text(full_text)

            logger.info(
                f"商品 {request.product_id} 嵌入向量生成成功，维度: {len(embedding_vector)}"
            )

            # 3. 准备插入数据
            entities = [{
                "product_id": request.product_id,
                "embedding": embedding_vector,
            }]

            # 4. 插入到 Milvus
            collection = get_product_collection()
            insert_result = collection.upsert(entities)

            # 5. 刷新 collection 以确保数据持久化
            collection.flush()

            # 6. 获取插入的 ID
            vector_id = str(insert_result.primary_keys[0])

            logger.info(
                f"商品 {request.product_id} 向量化成功，vector_id: {vector_id}"
            )

            return VectorizeProductResponse(
                product_id=request.product_id,
                success=True,
                vector_id=vector_id,
                error_message=None,
            )

        except Exception as e:
            error_msg = f"商品向量化失败: {str(e)}"
            logger.error(
                f"商品 {request.product_id} 向量化失败: {e}",
                exc_info=True
            )

            return VectorizeProductResponse(
                product_id=request.product_id,
                success=False,
                vector_id="",
                error_message=error_msg,
            )

    @classmethod
    def get_instance(cls) -> "ProductVectorizeService":
        """
        获取 ProductVectorizeService 单例.

        Returns:
            ProductVectorizeService 实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_product_vectorize_service() -> ProductVectorizeService:
    """获取商品向量化服务单例（便捷函数）."""
    return ProductVectorizeService.get_instance()


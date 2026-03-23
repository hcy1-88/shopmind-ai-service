"""
@File       : product_service.py
@Description:

@Time       : 2026/1/6 12:53
@Author     : hcy18
"""
from typing import Optional

import httpx

from app.clients.service_discovery import get_product_service_url
from app.schemas import ResultContext
from app.schemas.page_result_schema import PageResult
from app.schemas.product_response_schema import ProductResponseDto
from app.utils.logger import app_logger as logger
from app.utils.trace_context import get_trace_id, TRACE_ID_HEADER


class ProductServiceClient:
    """商品服务客户端."""

    _instance: Optional["ProductServiceClient"] = None

    def __init__(self):
        self._base_url: Optional[str] = None
        self.timeout = 10.0  # 请求超时时间（秒）

    async def _get_base_url(self) -> str:
        """获取商品服务的基础 URL（带缓存）."""
        if not self._base_url:
            self._base_url = await get_product_service_url()
        return self._base_url

    def _get_headers(self) -> dict[str, str]:
        """获取带 Trace ID 的请求头."""
        trace_id = get_trace_id()
        return {
            TRACE_ID_HEADER: trace_id,
            "Content-Type": "application/json"
        }


    async def get_new_products(self, limit: int = 10) -> list[ProductResponseDto]:
        """
        获取最新商品，支持随机，也就是多次调用可以得到不同的商品。

        Args:
            limit: 限制数，限制返回 limit 个新品

        Returns:
            商品 list 列表
        """
        try:
            base_url = await self._get_base_url()
            url = f"{base_url}/products/new"
            logger.info(f"获取新商品, url={url}, limit={limit}")

            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    params={"limit": limit},
                    headers=headers
                )
                response.raise_for_status()

                products_result_context = ResultContext[list[ProductResponseDto]](**response.json())

                if products_result_context.success:
                    products = products_result_context.data
                    logger.info("获取新商品成功！")
                    return products
                else:
                    logger.error(f"请求商品商品失败！url: {url}, request: limit={limit}")
                    raise httpx.HTTPError("商品服务异常！")
        except httpx.TimeoutException:
            logger.error(f"新品获取超时！")
            raise
        except Exception as e:
            logger.error(
                f"获取新品异常: error={str(e)}",
                exc_info=True
            )
            raise


    async def get_product_by_id(self, product_id: int) -> ProductResponseDto:
        """
        根据商品ID查询商品详情

        Args:
            product_id: 商品ID

        Returns:
            商品详情
        """
        try:
            base_url = await self._get_base_url()
            url = f"{base_url}/products/detail/{product_id}"
            logger.info(f"获取商品详情, url={url}, product_id={product_id}")

            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    headers=headers
                )
                response.raise_for_status()

                product_result_context = ResultContext[ProductResponseDto](**response.json())

                if product_result_context.success:
                    product = product_result_context.data
                    logger.info(f"获取商品详情成功！product_name={product.name}")
                    return product
                else:
                    logger.error(f"获取商品详情失败！url: {url}, product_id={product_id}")
                    raise httpx.HTTPError("商品服务异常！")
        except httpx.TimeoutException:
            logger.error(f"获取商品详情超时！")
            raise
        except Exception as e:
            logger.error(
                f"获取商品详情异常: error={str(e)}",
                exc_info=True
            )
            raise

    async def search_products(self, query: str, page_number: int = 1, page_size: int = 10) -> PageResult[list[ProductResponseDto]]:
        """
        根据输入的查询，搜索商品
        Args:
            query: 用户输入的商品查询。也就是商品搜索框里的输入
            page_number: 分页页码
            page_size: 一页的大小
        Returns: 与 query 输入相关的商品列表
        """
        try:
            base_url = await self._get_base_url()
            url = f"{base_url}/products/search"
            logger.info(f"获取新商品, url={url}, page_number={page_number}, page_size={page_size}")

            headers = self._get_headers()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    params={"keyword": query, "pageNumber": page_number, "pageSize": page_size},
                    headers=headers
                )
                response.raise_for_status()

                products_result_context = ResultContext[PageResult[list[ProductResponseDto]]](**response.json())

                if products_result_context.success:
                    products = products_result_context.data
                    logger.info("搜索新商品成功！")
                    return products
                else:
                    logger.error(f"搜索商品商品失败！")
                    raise httpx.HTTPError("商品服务异常！")
        except httpx.TimeoutException:
            logger.error(f"搜索商品超时！")
            raise
        except Exception as e:
            logger.error(
                f"搜索商品异常: error={str(e)}",
                exc_info=True
            )
            raise

    @classmethod
    async def get_instance(cls) -> "ProductServiceClient":
        """统一单例"""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._get_base_url()
        return cls._instance


async def get_product_service_client() -> ProductServiceClient:
    return await ProductServiceClient.get_instance()


async def init_product_service_client():
    await get_product_service_client()

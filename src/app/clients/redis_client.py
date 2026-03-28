"""Redis client for caching user vectors."""

import json
from typing import Optional
import redis.asyncio as aioredis

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger


class RedisClient:
    """Redis client wrapper for user vector caching."""

    _instance: Optional["RedisClient"] = None

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.prefix = "user_vector:"
        self.ttl = 3600  # 默认 1 小时过期

    @classmethod
    def get_instance(cls) -> "RedisClient":
        """获取 Redis 客户端单例."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self):
        """连接 Redis."""
        try:
            nacos_client = get_nacos_client()
            chat_config = nacos_client.get_chat_config()
            redis_config = chat_config["checkpointer"]["redis"]
            
            # 构建连接参数
            connect_params = {
                "password": redis_config.get("password"),
                "encoding": "utf-8",
                "decode_responses": True,
                "max_connections": redis_config.get("max_connections", 10),
            }
            
            self.redis = await aioredis.from_url(
                redis_config["url"],
                **connect_params
            )

            # 测试连接
            await self.redis.ping()
            logger.info("Redis 连接成功")

        except Exception as e:
            logger.error(f"Redis 连接失败: {e}", exc_info=True)
            raise

    async def close(self):
        """关闭 Redis 连接."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis 连接已关闭")


def get_redis_client() -> RedisClient:
    """获取 Redis 客户端单例."""
    return RedisClient.get_instance()


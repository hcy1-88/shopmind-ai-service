"""Redis Checkpoint Saver for LangGraph."""

from __future__ import annotations

import pickle
import redis
import redis.asyncio as aioredis
from typing import Optional, Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple, CheckpointMetadata

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    基于 Redis 的 Checkpoint 存储
    
    特点：
    - 支持消息历史持久化
    - 支持过期时间（TTL）
    - 多会话隔离（通过 thread_id）
    
    注意：使用同步 Redis 客户端，因为 BaseCheckpointSaver 要求同步方法
    """
    
    def __init__(self, ttl: int = 7200):
        """
        初始化 Redis Checkpoint Saver
        
        Args:
            ttl: 过期时间（秒），默认 2 小时 = 7200 秒
        """
        super().__init__()
        self.ttl = ttl
        self.prefix = "checkpoint:"
        
        # 创建同步 Redis 连接（用于 checkpoint 操作）
        nacos_client = get_nacos_client()
        redis_config = nacos_client.get_redis_config()
        
        self.redis = redis.from_url(
            redis_config["url"],
            password=redis_config.get("password"),
            encoding="utf-8",
            decode_responses=False,  # checkpoint 需要存储二进制数据
            max_connections=redis_config.get("max_connections", 10),
        )
        
        # 同时创建异步 Redis 连接（用于异步方法）
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 在异步上下文中，创建异步客户端
            self._async_redis_future = asyncio.create_task(self._create_async_redis(redis_config))
        except RuntimeError:
            # 不在异步上下文中，稍后创建
            self._async_redis_future = None
            self._async_redis = None
        
        logger.info(f"RedisCheckpointSaver 初始化完成，TTL={ttl}秒")
    
    async def _create_async_redis(self, redis_config):
        """创建异步 Redis 连接"""
        if not hasattr(self, '_async_redis') or self._async_redis is None:
            self._async_redis = await aioredis.from_url(
                redis_config["url"],
                password=redis_config.get("password"),
                encoding="utf-8",
                decode_responses=False,
                max_connections=redis_config.get("max_connections", 10),
            )
        return self._async_redis
    
    async def _get_async_redis(self):
        """获取异步 Redis 连接"""
        if not hasattr(self, '_async_redis') or self._async_redis is None:
            nacos_client = get_nacos_client()
            redis_config = nacos_client.get_redis_config()
            self._async_redis = await self._create_async_redis(redis_config)
        return self._async_redis
    
    def _make_key(self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: Optional[str] = None) -> str:
        """生成 Redis key"""
        if checkpoint_id:
            return f"{self.prefix}{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        return f"{self.prefix}{thread_id}:{checkpoint_ns}:latest"
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        """保存 checkpoint 到 Redis"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        
        key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
        
        # 序列化数据
        data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
        }
        
        try:
            # 使用 pickle 序列化
            serialized = pickle.dumps(data)
            
            # 保存到 Redis，设置过期时间
            self.redis.setex(key, self.ttl, serialized)
            
            # 同时保存一个 latest 指针
            latest_key = self._make_key(thread_id, checkpoint_ns, None)
            self.redis.setex(latest_key, self.ttl, checkpoint_id.encode('utf-8'))
            
            logger.debug(f"Checkpoint 已保存: {key}")
        except Exception as e:
            logger.error(f"保存 checkpoint 失败: {e}", exc_info=True)
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """从 Redis 获取 checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        try:
            # 如果没有指定 checkpoint_id，获取最新的
            if not checkpoint_id:
                latest_key = self._make_key(thread_id, checkpoint_ns, None)
                checkpoint_id_bytes = self.redis.get(latest_key)
                if not checkpoint_id_bytes:
                    return None
                checkpoint_id = checkpoint_id_bytes.decode('utf-8') if isinstance(checkpoint_id_bytes, bytes) else checkpoint_id_bytes
            
            # 获取 checkpoint 数据
            key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
            serialized = self.redis.get(key)
            
            if not serialized:
                return None
            
            # 反序列化
            data = pickle.loads(serialized)
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=data["checkpoint"],
                metadata=data.get("metadata", {}),
                parent_config=data.get("parent_config"),
            )
            
        except Exception as e:
            logger.error(f"获取 checkpoint 失败: {e}", exc_info=True)
            return None
    
    def list(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """列出 checkpoints（简化实现，只返回最新的）"""
        tuple_result = self.get_tuple(config)
        if tuple_result:
            yield tuple_result
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """异步获取 checkpoint（LangGraph 异步流需要）"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        try:
            redis_async = await self._get_async_redis()
            
            # 如果没有指定 checkpoint_id，获取最新的
            if not checkpoint_id:
                latest_key = self._make_key(thread_id, checkpoint_ns, None)
                checkpoint_id_bytes = await redis_async.get(latest_key)
                if not checkpoint_id_bytes:
                    return None
                checkpoint_id = checkpoint_id_bytes.decode('utf-8') if isinstance(checkpoint_id_bytes, bytes) else checkpoint_id_bytes
            
            # 获取 checkpoint 数据
            key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
            serialized = await redis_async.get(key)
            
            if not serialized:
                return None
            
            # 反序列化
            data = pickle.loads(serialized)
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=data["checkpoint"],
                metadata=data.get("metadata", {}),
                parent_config=data.get("parent_config"),
            )
            
        except Exception as e:
            logger.error(f"异步获取 checkpoint 失败: {e}", exc_info=True)
            return None
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        """异步保存 checkpoint"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        
        key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
        
        # 序列化数据
        data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_config": config.get("parent_config"),
        }
        
        try:
            # 使用 pickle 序列化
            serialized = pickle.dumps(data)
            
            # 异步保存到 Redis
            redis_async = await self._get_async_redis()
            await redis_async.setex(key, self.ttl, serialized)
            
            # 同时保存一个 latest 指针
            latest_key = self._make_key(thread_id, checkpoint_ns, None)
            await redis_async.setex(latest_key, self.ttl, checkpoint_id.encode('utf-8'))
            
            logger.debug(f"Checkpoint 已异步保存: {key}")
        except Exception as e:
            logger.error(f"异步保存 checkpoint 失败: {e}", exc_info=True)
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """异步写入 writes（LangGraph 需要）"""
        # 简化实现：不单独存储 writes，直接通过 checkpoint 管理
        # 如果需要更精细的控制，可以单独存储 writes
        pass
    
    async def clear_thread_history(self, thread_id: str) -> bool:
        """
        清除指定 thread_id 的所有历史
        
        Args:
            thread_id: 会话ID
            
        Returns:
            是否清除成功
        """
        try:
            # 查找所有相关的 keys
            pattern = f"{self.prefix}{thread_id}:*"
            keys = []
            for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            # 删除所有 keys
            if keys:
                self.redis.delete(*keys)
                logger.info(f"已清除 thread_id={thread_id} 的历史，删除了 {len(keys)} 个 key")
            
            return True
        except Exception as e:
            logger.error(f"清除历史失败: {e}", exc_info=True)
            return False
    
    async def get_thread_messages(self, thread_id: str) -> list[dict]:
        """
        获取指定 thread_id 的消息历史
        
        Args:
            thread_id: 会话ID
            
        Returns:
            消息列表
        """
        try:
            config = RunnableConfig(configurable={
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                })
            
            tuple_result = self.get_tuple(config)
            if not tuple_result:
                return []
            
            # 从 checkpoint 中提取消息
            checkpoint = tuple_result.checkpoint
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            
            # 转换为前端需要的格式
            result = []
            for msg in messages:
                if not msg.content and isinstance(msg, AIMessage) or isinstance(msg, ToolMessage):
                    continue
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    result.append({
                        "role": "user" if msg.type == "human" else "assistant",
                        "content": msg.content
                    })
            
            return result
        except Exception as e:
            logger.error(f"获取消息历史失败: {e}", exc_info=True)
            return []


def get_redis_checkpoint_saver(ttl: int = 7200) -> RedisCheckpointSaver:
    """获取 Redis Checkpoint Saver"""
    return RedisCheckpointSaver(ttl=ttl)


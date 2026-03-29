"""Redis Checkpoint Saver for LangGraph."""

from __future__ import annotations

import pickle
from warnings import deprecated

import redis
import redis.asyncio as aioredis
from typing import Optional, Any, Iterator

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple, CheckpointMetadata

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger


@deprecated("redis checkpoint 已经废弃，考虑到持久化问题，已迁移至 postgres checkpoint")
class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    基于 Redis 的 Checkpoint 存储
    
    特点：
    - 支持消息历史持久化
    - 支持过期时间（TTL）
    - 多会话隔离（通过 thread_id）
    
    注意：使用同步 Redis 客户端，因为 BaseCheckpointSaver 要求同步方法
    """
    
    # 对话列表的 Redis key 前缀
    CONVERSATION_LIST_PREFIX = "conversation_list:"
    
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
        chat_config = nacos_client.get_chat_config()
        redis_config = chat_config["checkpointer"]["redis"]
        
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
            chat_config = nacos_client.get_chat_config()
            redis_config = chat_config["checkpointer"]["redis"]
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
    
    # ========== 对话列表管理方法 ==========
    
    def _get_conversation_list_key(self, user_id: str) -> str:
        """获取对话列表的 Redis key"""
        return f"{self.CONVERSATION_LIST_PREFIX}{user_id}"
    
    async def get_conversation_list(self, user_id: str) -> list[dict]:
        """
        获取用户的所有对话列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            对话列表，每个元素包含 session_id 和 name
        """
        try:
            redis_async = await self._get_async_redis()
            key = self._get_conversation_list_key(user_id)
            
            # 使用 Hash 结构存储 session_id -> name 的映射
            conversations = await redis_async.hgetall(key)
            
            result = []
            for session_id, name in conversations.items():
                # 解码 bytes
                session_id = session_id.decode('utf-8') if isinstance(session_id, bytes) else session_id
                name = name.decode('utf-8') if isinstance(name, bytes) else name
                result.append({
                    "session_id": session_id,
                    "name": name
                })
            
            # 按更新时间排序（最新的在前）
            # 由于没有存储更新时间，暂按 session_id 字符串排序（或保持原顺序）
            return result
        except Exception as e:
            logger.error(f"获取对话列表失败: {e}", exc_info=True)
            return []
    
    async def create_conversation(self, user_id: str, session_id: str, name: str) -> bool:
        """
        创建新对话
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 对话名称
            
        Returns:
            是否创建成功
        """
        try:
            redis_async = await self._get_async_redis()
            key = self._get_conversation_list_key(user_id)
            
            # 保存到 Hash
            await redis_async.hset(key, session_id, name)
            
            # 设置过期时间（与消息历史一致）
            await redis_async.expire(key, self.ttl)
            
            logger.info(f"创建对话成功: user_id={user_id}, session_id={session_id}, name={name}")
            return True
        except Exception as e:
            logger.error(f"创建对话失败: {e}", exc_info=True)
            return False
    
    async def update_conversation_name(self, user_id: str, session_id: str, name: str) -> bool:
        """
        更新对话名称
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 新对话名称
            
        Returns:
            是否更新成功
        """
        try:
            redis_async = await self._get_async_redis()
            key = self._get_conversation_list_key(user_id)
            
            # 如果对话不存在，则创建
            exists = await redis_async.hexists(key, session_id)
            if not exists:
                logger.info(f"对话不存在，将创建: session_id={session_id}")
            
            # 保存或更新对话名称
            await redis_async.hset(key, session_id, name)
            
            # 设置过期时间（与消息历史一致）
            await redis_async.expire(key, self.ttl)
            
            logger.info(f"更新对话名称成功: session_id={session_id}, name={name}")
            return True
        except Exception as e:
            logger.error(f"更新对话名称失败: {e}", exc_info=True)
            return False
    
    async def delete_conversation(self, user_id: str, session_id: str) -> bool:
        """
        删除对话
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        try:
            redis_async = await self._get_async_redis()
            key = self._get_conversation_list_key(user_id)
            
            # 从对话列表中删除
            await redis_async.hdel(key, session_id)
            
            # 清除该会话的历史消息
            await self.clear_thread_history(session_id)
            
            logger.info(f"删除对话成功: session_id={session_id}")
            return True
        except Exception as e:
            logger.error(f"删除对话失败: {e}", exc_info=True)
            return False
    
    async def get_conversation_name(self, user_id: str, session_id: str) -> Optional[str]:
        """
        获取指定对话的名称
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            
        Returns:
            对话名称，如果不存在返回 None
        """
        try:
            redis_async = await self._get_async_redis()
            key = self._get_conversation_list_key(user_id)
            
            name = await redis_async.hget(key, session_id)
            if name:
                return name.decode('utf-8') if isinstance(name, bytes) else name
            return None
        except Exception as e:
            logger.error(f"获取对话名称失败: {e}", exc_info=True)
            return None


def get_redis_checkpoint_saver(ttl: int = 7200) -> RedisCheckpointSaver:
    """获取 Redis Checkpoint Saver"""
    return RedisCheckpointSaver(ttl=ttl)


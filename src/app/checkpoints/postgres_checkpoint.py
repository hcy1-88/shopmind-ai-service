"""
@File       : postgres_checkpoint.py
@Description: 基于 AsyncPostgresSaver 的 checkpoint 实现（组合模式），支持对话列表管理

@Time       : 2026/3/26 20:23
@Author     : hcy18
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.utils.id_util import gen_id
from app.utils.logger import app_logger as logger


# conversations 表 DDL
CONVERSATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    conversation_name VARCHAR(512) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, session_id)
)
"""

CONVERSATIONS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)
"""


class PostgresCheckpoint:
    """
    基于 AsyncPostgresSaver 的 checkpointer（组合模式），支持对话列表管理功能。

    内部持有 AsyncPostgresSaver 实例，新增对话列表管理方法以对齐 RedisCheckpointSaver API。
    支持懒加载初始化，线程安全。
    """

    def __init__(self, db_uri: str):
        """
        初始化 PostgresCheckpoint（延迟初始化，异步资源在首次使用时创建）

        Args:
            db_uri: PostgreSQL 连接 URI
        """
        self._db_uri = db_uri
        self._pool: Optional[AsyncConnectionPool] = None
        self._saver: Optional[AsyncPostgresSaver] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> AsyncPostgresSaver:
        """确保异步资源已初始化（懒加载，线程安全）"""
        if self._initialized:
            return self._saver

        async with self._init_lock:
            # 双重检查
            if self._initialized:
                return self._saver

            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
            }
            self._pool = AsyncConnectionPool(
                conninfo=self._db_uri,
                max_size=20,
                kwargs=connection_kwargs,
            )
            await self._pool.open()

            self._saver = AsyncPostgresSaver(self._pool)

            # 创建标准 checkpoint 表和 conversations 表
            await self._saver.setup()
            await self._setup_conversations_table()

            self._initialized = True
            logger.info("PostgresCheckpoint 初始化完成")
            return self._saver

    async def _setup_conversations_table(self) -> None:
        """创建 conversations 表（如果不存在）"""
        async with self._pool.cursor() as cur:
            await cur.execute(CONVERSATIONS_TABLE_DDL)
            await cur.execute(CONVERSATIONS_INDEX_DDL)

    # ========== 委托给内部 saver 的 Checkpoint 方法 ==========

    async def aget_tuple(self, config: RunnableConfig) -> Any:
        """异步获取 checkpoint tuple"""
        saver = await self._ensure_initialized()
        return await saver.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """异步列出 checkpoints"""
        saver = await self._ensure_initialized()
        return await saver.alist(config, filter=filter, before=before, limit=limit)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: dict,
    ) -> RunnableConfig:
        """异步保存 checkpoint"""
        saver = await self._ensure_initialized()
        return await saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """异步写入 writes"""
        saver = await self._ensure_initialized()
        return await saver.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """删除线程所有 checkpoints"""
        saver = await self._ensure_initialized()
        return await saver.adelete_thread(thread_id)

    # ========== 对话列表管理方法 ==========

    async def get_conversation_list(self, user_id: str) -> list[dict]:
        """
        获取用户的所有对话列表

        Args:
            user_id: 用户ID

        Returns:
            对话列表，每个元素包含 session_id 和 name
        """
        await self._ensure_initialized()
        try:
            async with self._pool.cursor() as cur:
                await cur.execute(
                    """
                    SELECT session_id, conversation_name as name
                    FROM conversations
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
                return [
                    {"session_id": row["session_id"], "name": row["name"]}
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"获取对话列表失败: {e}", exc_info=True)
            return []

    async def create_conversation(
        self, user_id: str, session_id: str, name: str
    ) -> bool:
        """
        创建新对话

        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 对话名称

        Returns:
            是否创建成功
        """
        await self._ensure_initialized()
        try:
            conversation_id = gen_id()
            async with self._pool.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO conversations (id, user_id, session_id, conversation_name)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET conversation_name = EXCLUDED.conversation_name,
                                 updated_at = CURRENT_TIMESTAMP
                    """,
                    (conversation_id, user_id, session_id, name),
                )
            logger.info(
                f"创建对话成功: user_id={user_id}, session_id={session_id}, name={name}"
            )
            return True
        except Exception as e:
            logger.error(f"创建对话失败: {e}", exc_info=True)
            return False

    async def update_conversation_name(
        self, user_id: str, session_id: str, name: str
    ) -> bool:
        """
        更新对话名称

        Args:
            user_id: 用户ID
            session_id: 会话ID
            name: 新对话名称

        Returns:
            是否更新成功
        """
        await self._ensure_initialized()
        try:
            async with self._pool.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO conversations (id, user_id, session_id, conversation_name)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET conversation_name = EXCLUDED.conversation_name,
                                 updated_at = CURRENT_TIMESTAMP
                    """,
                    (gen_id(), user_id, session_id, name),
                )
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
        await self._ensure_initialized()
        try:
            async with self._pool.cursor() as cur:
                await cur.execute(
                    "DELETE FROM conversations WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
            await self.clear_thread_history(session_id)
            logger.info(f"删除对话成功: session_id={session_id}")
            return True
        except Exception as e:
            logger.error(f"删除对话失败: {e}", exc_info=True)
            return False

    async def get_conversation_name(
        self, user_id: str, session_id: str
    ) -> Optional[str]:
        """
        获取指定对话的名称

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            对话名称，如果不存在返回 None
        """
        await self._ensure_initialized()
        try:
            async with self._pool.cursor() as cur:
                await cur.execute(
                    """
                    SELECT conversation_name FROM conversations
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (user_id, session_id),
                )
                row = await cur.fetchone()
                if row:
                    return row["conversation_name"]
                return None
        except Exception as e:
            logger.error(f"获取对话名称失败: {e}", exc_info=True)
            return None

    async def clear_thread_history(self, thread_id: str) -> bool:
        """
        清除指定 thread_id 的所有历史

        Args:
            thread_id: 会话ID

        Returns:
            是否清除成功
        """
        try:
            await self.adelete_thread(thread_id)
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
            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )
            tuple_result = await self.aget_tuple(config)
            if not tuple_result:
                return []

            checkpoint = tuple_result.checkpoint
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])

            result = []
            for msg in messages:
                if isinstance(msg, AIMessage) and not msg.content:
                    continue
                if isinstance(msg, ToolMessage):
                    continue
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    result.append(
                        {
                            "role": "user" if msg.type == "human" else "assistant",
                            "content": msg.content,
                        }
                    )

            return result
        except Exception as e:
            logger.error(f"获取消息历史失败: {e}", exc_info=True)
            return []


def get_postgres_checkpoint(db_uri: str) -> PostgresCheckpoint:
    """
    创建 PostgresCheckpoint 实例（同步工厂函数）

    Args:
        db_uri: PostgreSQL 连接 URI

    Returns:
        PostgresCheckpoint 实例（懒加载初始化）
    """
    return PostgresCheckpoint(db_uri)
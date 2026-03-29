"""
@File       : conversations_manager.py
@Description: PostgreSQL 会话元数据管理器（独立职责）

@Time       : 2026/3/29
@Author     : refactored from postgres_checkpoint.py
"""
from __future__ import annotations

import asyncio
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

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


class ConversationsManager:
    """
    会话元数据管理器（与 LangGraph checkpointer 职责分离）

    管理 conversations 表（id, user_id, session_id, conversation_name, created_at, updated_at）
    不涉及 checkpoint 状态存储。
    """

    def __init__(self, pool: AsyncConnectionPool):
        """
        Args:
            pool: AsyncConnectionPool 实例（与 AsyncPostgresSaver 共用同一 pool）
        """
        self._pool = pool
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def ensure_initialized(self) -> None:
        """创建 conversations 表（如果不存在）"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(CONVERSATIONS_TABLE_DDL)
                    await cur.execute(CONVERSATIONS_INDEX_DDL)

            self._initialized = True
            logger.info("ConversationsManager 初始化完成")

    async def get_conversation_list(self, user_id: str) -> list[dict]:
        """
        获取用户的所有对话列表

        Args:
            user_id: 用户ID

        Returns:
            对话列表，每个元素包含 session_id 和 name
        """
        await self.ensure_initialized()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
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
        await self.ensure_initialized()
        try:
            conversation_id = gen_id()
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
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
        await self.ensure_initialized()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
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
        删除对话（不清理 checkpoint 历史，由调用方负责）

        Args:
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        await self.ensure_initialized()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "DELETE FROM conversations WHERE user_id = %s AND session_id = %s",
                        (user_id, session_id),
                    )
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
        await self.ensure_initialized()
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
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

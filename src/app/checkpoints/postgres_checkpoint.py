"""
@File       : postgres_checkpoint.py
@Description: 基于 AsyncPostgresSaver 的 checkpoint 实现，支持对话列表管理

@Time       : 2026/3/26 20:23
@Author     : hcy18
"""
from __future__ import annotations

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


class PostgresCheckpoint(AsyncPostgresSaver):
    """
    基于 AsyncPostgresSaver 的 checkpointer，支持对话列表管理功能。

    继承自 AsyncPostgresSaver，拥有完整的 checkpoint 存储能力，
    同时新增对话列表管理方法以对齐 RedisCheckpointSaver API。
    """

    @classmethod
    async def get_async_checkpoint(cls, db_uri: str) -> "PostgresCheckpoint":
        """
        创建异步 Postgres checkpointer

        Args:
            db_uri: PostgreSQL 连接 URI

        Returns:
            PostgresCheckpoint 实例
        """
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        async with AsyncConnectionPool(
            conninfo=db_uri,
            max_size=20,
            kwargs=connection_kwargs,
        ) as pool:
            checkpointer = cls(pool)
            # NOTE: 需要调用 .setup() 第一次使用时初始化表结构
            await checkpointer.setup()
            return checkpointer

    async def setup(self) -> None:
        """设置 checkpoint 数据库，包括 conversations 表"""
        # 先调用父类的 setup 创建标准的 checkpoint 表
        await super().setup()

        # 创建 conversations 表（如果不存在）
        async with self._cursor() as cur:
            await cur.execute(CONVERSATIONS_TABLE_DDL)
            await cur.execute(CONVERSATIONS_INDEX_DDL)

    # ========== 对话列表管理方法 ==========

    async def get_conversation_list(self, user_id: str) -> list[dict]:
        """
        获取用户的所有对话列表

        Args:
            user_id: 用户ID

        Returns:
            对话列表，每个元素包含 session_id 和 name
        """
        try:
            async with self._cursor() as cur:
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
        try:
            conversation_id = gen_id()
            async with self._cursor() as cur:
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
        try:
            async with self._cursor() as cur:
                # 如果对话不存在，则创建
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
        try:
            async with self._cursor() as cur:
                await cur.execute(
                    "DELETE FROM conversations WHERE user_id = %s AND session_id = %s",
                    (user_id, session_id),
                )
            # 清除该会话的历史消息
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
        try:
            async with self._cursor() as cur:
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

            # 从 checkpoint 中提取消息
            checkpoint = tuple_result.checkpoint
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])

            # 转换为前端需要的格式
            result = []
            for msg in messages:
                # 跳过空的 AIMessage 和所有 ToolMessage
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
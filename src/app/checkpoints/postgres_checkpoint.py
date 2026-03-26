"""
@File       : postgres_checkpoint.py
@Description:

@Time       : 2026/3/26 20:23
@Author     : hcy18
"""
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool, ConnectionPool


class PostgresCheckpoint:
    """
    基于 postgresql 的 checkpointer
    """
    async def get_async_checkpoint(self, db_uri: str):
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        # 使用异步数据库连接池
        async with AsyncConnectionPool(
                # Example configuration
                conninfo=db_uri,
                max_size=20,
                kwargs=connection_kwargs,
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)

            # NOTE: you need to call .setup() the first time you're using your checkpointer
            await checkpointer.setup()
            return checkpointer

    def get_checkpoint(self, db_uri: str):
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        with ConnectionPool(
                # Example configuration
                conninfo=db_uri,
                max_size=20,
                kwargs=connection_kwargs,
        ) as pool:
            checkpointer = PostgresSaver(pool)

            # NOTE: you need to call .setup() the first time you're using your checkpointer
            checkpointer.setup()
            return checkpointer
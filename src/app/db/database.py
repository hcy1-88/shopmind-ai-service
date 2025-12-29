"""PostgreSQL database connection and session management."""

from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from src.app.config.nacos_client import get_nacos_client
from src.app.utils.logger import app_logger as logger

# Base class for ORM models
Base = declarative_base()

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


async def init_db() -> None:
    """Initialize database connection with configuration from Nacos."""
    global _engine, _async_session_maker

    try:
        # 从 nacos 获取配置
        nacos_client = get_nacos_client()
        db_config = nacos_client.get_postgres_config()

        # 构建数据库 url
        db_url = (
            f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # 创建 async engine
        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20),
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        # 创建 session maker
        _async_session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        logger.info(
            "Database initialized",
            extra={
                "host": db_config["host"],
                "database": db_config["database"],
            },
        )

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """Close database connection."""
    global _engine

    if _engine:
        await _engine.dispose()
        logger.info("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session.

    Yields:
        AsyncSession instance

    Example:
        ```python
        async with get_db() as session:
            result = await session.execute(select(User))
        ```
    """
    if not _async_session_maker:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_engine() -> AsyncEngine:
    """
    Get database engine.

    Returns:
        AsyncEngine instance
    """
    if not _engine:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine

"""FastAPI application main entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.config.nacos_client import get_nacos_client
from app.db import close_db, init_db
from app.middleware.trace_middleware import TraceIDMiddleware
from app.mq import close_rocketmq, init_rocketmq
from app.routers import ai_router
from app.schemas.result_context import ResultContext
from app.utils.logger import app_logger as logger
from app.utils.logger import setup_logger
from app.vector_store import close_milvus, init_milvus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI service...")

    try:
        # Get settings
        settings = get_settings()

        # Setup logging
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        setup_logger(log_level)

        # nacos 初始化
        nacos_client = get_nacos_client(settings)
        await nacos_client.connect()

        # 初始化 LLM 服务
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()
        logger.info("LLM service initialized")

        # # 初始化 database
        # await init_db()
        # logger.info("Database initialized")
        #
        # # 初始化 Milvus
        # await init_milvus()
        # logger.info("Milvus initialized")
        #
        # # 初始化 RocketMQ
        # await init_rocketmq()
        # logger.info("RocketMQ initialized")

        logger.info("AI service started successfully")

    except Exception as e:
        logger.error(f"Failed to start AI service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AI service...")

    try:
        # 注销 from Nacos
        nacos_client = get_nacos_client()
        await nacos_client.deregister_service()
        logger.info("Nacos service deregistered")

        # # Close RocketMQ
        # await close_rocketmq()
        # logger.info("RocketMQ closed")
        #
        # # Close Milvus
        # await close_milvus()
        # logger.info("Milvus closed")
        #
        # # Close database
        # await close_db()
        # logger.info("Database closed")

        logger.info("AI service shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="ShopMind AI Service",
    description="智能电商平台",
    version="0.1.0",
    lifespan=lifespan,
)

# Add TraceID middleware (必须在 CORS 之前，以便尽早设置 traceId)
app.add_middleware(TraceIDMiddleware)

# Add CORS middleware 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Request validation error",
        extra={
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )
    result = ResultContext.fail(
        message="请求参数验证失败",
        code="VALIDATION_ERROR",
        data={"errors": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=result.model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "error": str(exc),
        },
        exc_info=True,
    )
    result = ResultContext.fail(
        message=f"内部服务器错误: {str(exc)}",
        code="SYS9999",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=result.model_dump(),
    )


# Include routers
app.include_router(ai_router.router)


# Root endpoint
@app.get("/", tags=["Root"], response_model=ResultContext[dict])
async def root() -> ResultContext[dict]:
    """Root endpoint."""
    return ResultContext.success(
        data={
            "service": "shopmind-ai-service",
            "version": "0.1.0",
            "status": "running",
        },
        message="服务运行中",
    )


# Health check endpoint
@app.get("/health", tags=["Health"], response_model=ResultContext[dict])
async def health() -> ResultContext[dict]:
    """Health check endpoint."""
    return ResultContext.success(
        data={
            "status": "healthy",
            "service": "shopmind-ai-service",
        },
        message="服务健康",
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.service_ip,
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

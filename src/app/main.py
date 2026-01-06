"""FastAPI application main entry point."""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
# from fastapi.middleware.cors import CORSMiddleware  # CORS 由 Gateway 处理
from fastapi.responses import JSONResponse
from app.clients.redis_client import get_redis_client
from app.config import get_settings
from app.config.nacos_client import get_nacos_client, init_nacos
from app.middleware.trace_middleware import TraceIDMiddleware
from app.routers import ai_ask_router, ai_product_router, rag_router
from app.schemas.result_context import ResultContext
from app.services.ai_chat_service import init_ai_chat_service
from app.services.embedding_service import init_embedding_service
from app.services.llm_service import init_llm_service
from app.services.rag_service import init_rag_service
from app.utils.logger import app_logger as logger
from app.utils.logger import setup_logger
from app.vector_store import init_milvus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    try:
        # Get settings first
        settings = get_settings()

        # Setup logging (必须在记录日志之前配置)
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        setup_logger(level=log_level)  # 使用关键字参数确保正确传递 level
        
        logger.info("正在启动 Shopmind AI service...")

        # nacos 初始化
        await init_nacos(settings)

        # 初始化 LLM 服务
        await init_llm_service()

        # 初始化 Embedding 服务
        init_embedding_service()

        # 初始化 Milvus
        init_milvus()

         # 初始化 Redis
        redis_client = get_redis_client()
        await redis_client.connect()
        logger.info("Redis 初始化完成")

        logger.info("Shopmind AI service 启动成功！")

        # 初始化对话
        init_ai_chat_service()
        logger.info("AI 对话初始化成功！")

        # 初始化 RAG 服务
        init_rag_service()
        
        # ===== 新增：打印服务启动 Banner =====
        service_name = "ShopMind AI Service"
        host = settings.service_ip
        port = settings.service_port
        url = f"http://{host}:{port}"

        banner = f"""
        {'=' * 60}
        🚀 {service_name} 已启动！
        🔗 访问地址: {url}
        🌐 Host: {host}
        🚪 Port: {port}
        {'=' * 60}
        """
        print(banner, file=sys.stderr)

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

        # Close Milvus
        from app.vector_store.milvus_client import MilvusClient
        milvus_client = MilvusClient.get_instance()
        await milvus_client.close()
        logger.info("Milvus closed")

        # 关闭 Redis
        redis_client = get_redis_client()
        await redis_client.close()
        logger.info("Redis 连接已关闭")

        logger.info("Shopmind AI service 已关闭...")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="ShopMind AI Service",
    description="智能电商平台",
    version="0.1.0",
    lifespan=lifespan,
)

# Add TraceID middleware
app.add_middleware(TraceIDMiddleware)

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
app.include_router(ai_product_router.router)
app.include_router(ai_ask_router.router)
app.include_router(rag_router.router)

# Root endpoint
@app.get("/", tags=["Root"], response_model=ResultContext[dict])
async def root() -> ResultContext[dict]:
    """Root endpoint."""
    return ResultContext.ok(
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
    return ResultContext.ok(
        data={
            "status": "healthy",
            "service": "shopmind-ai-service",
        },
        message="服务健康",
    )


from fastapi import Request

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.service_ip,
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        http="h11"
    )

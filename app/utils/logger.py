"""Logging configuration for the application."""

import logging
import sys

from pythonjsonlogger import json

from app.utils.trace_context import get_trace_id

APP_LOGGER_NAME = "shopmind_ai_service"


class TraceIDFilter(logging.Filter):
    """日志过滤器：自动注入 traceId 到日志记录中."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        为日志记录添加 traceId.
        
        Args:
            record: 日志记录对象
            
        Returns:
            True，表示不过滤该记录
        """
        record.traceId = get_trace_id()
        return True


def setup_logger(name: str = APP_LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured JSON logger.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 添加 traceId 过滤器
    handler.addFilter(TraceIDFilter())

    # JSON formatter
    # pythonjsonlogger 会自动将所有 record 属性包含在 JSON 中，包括 traceId
    formatter = json.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        timestamp=True,
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Application logger
app_logger = setup_logger(APP_LOGGER_NAME)

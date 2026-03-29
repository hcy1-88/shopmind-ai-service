"""Logging configuration for the application."""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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


def setup_logger(name: str = APP_LOGGER_NAME, level: int = logging.INFO, log_dir: str = "") -> logging.Logger:
    """
    Setup structured JSON logger.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for agent trace log files. If empty, uses project root.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with JSON formatter
    # 使用 sys.stderr 而不是 sys.stdout，避免被 uvicorn 重定向
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.addFilter(TraceIDFilter())

    # JSON formatter
    # pythonjsonlogger 会自动将所有 record 属性包含在 JSON 中，包括 traceId
    formatter = json.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        timestamp=True,
        json_ensure_ascii=False,
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for agent trace logs (TimedRotatingFileHandler, daily rotation, 7 days retention)
    if log_dir:
        log_dir_path = Path(log_dir)
    else:
        # 默认使用项目根目录
        log_dir_path = Path(__file__).parent.parent.parent.parent

    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "shopmind_agent.log"

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.addFilter(TraceIDFilter())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# Application logger
app_logger = setup_logger(APP_LOGGER_NAME)

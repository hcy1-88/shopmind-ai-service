"""Database models (ORM)."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.database import Base


class AIRequestLog(Base):
    """AI request log model for tracking API calls."""

    __tablename__ = "ai_request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Request type: title_check, image_check, description_generate",
    )
    request_data: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="Request payload (JSON)",
    )
    response_data: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="Response payload (JSON)",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="success",
        comment="Request status: success, error",
    )
    error_message: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if failed",
    )
    processing_time: Mapped[float] = mapped_column(
        Integer,
        nullable=True,
        comment="Processing time in milliseconds",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<AIRequestLog(id={self.id}, type={self.request_type}, status={self.status})>"

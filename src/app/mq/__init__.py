"""Message queue module."""

from src.app.mq.rocketmq_client import get_rocketmq_producer, init_rocketmq

__all__ = ["get_rocketmq_producer", "init_rocketmq"]

"""RocketMQ client for message queue operations."""

import json
from typing import Any, Optional

from rocketmq.client import Message, Producer, PushConsumer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger


class RocketMQProducer:
    """RocketMQ producer wrapper."""

    def __init__(self):
        """Initialize RocketMQ producer."""
        self.producer: Optional[Producer] = None
        self._started = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def start(self) -> None:
        """Start RocketMQ producer with retry logic."""
        try:
            # Get RocketMQ configuration from Nacos
            nacos_client = get_nacos_client()
            config = nacos_client.get_rocketmq_config()

            # Initialize producer
            self.producer = Producer(config["group_id"])
            self.producer.set_name_server_address(config["namesrv_addr"])

            # Set credentials if provided
            if config.get("access_key") and config.get("secret_key"):
                self.producer.set_session_credentials(
                    config["access_key"],
                    config["secret_key"],
                    "",
                )

            # Start producer
            self.producer.start()
            self._started = True

            logger.info(
                "RocketMQ producer started",
                extra={
                    "namesrv_addr": config["namesrv_addr"],
                    "group_id": config["group_id"],
                },
            )

        except Exception as e:
            logger.error(f"Failed to start RocketMQ producer: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown RocketMQ producer."""
        try:
            if self.producer and self._started:
                self.producer.shutdown()
                self._started = False
                logger.info("RocketMQ producer shutdown")
        except Exception as e:
            logger.error(f"Error shutting down RocketMQ producer: {e}")

    def send_message(
        self,
        topic: str,
        body: dict[str, Any],
        tags: str = "",
        keys: str = "",
    ) -> str:
        """
        Send message to RocketMQ topic.

        Args:
            topic: Topic name
            body: Message body (will be JSON serialized)
            tags: Message tags
            keys: Message keys

        Returns:
            Message ID
        """
        if not self.producer or not self._started:
            raise RuntimeError("RocketMQ producer not started")

        try:
            # Create message
            msg_body = json.dumps(body, ensure_ascii=False)
            msg = Message(topic)
            msg.set_body(msg_body.encode("utf-8"))

            if tags:
                msg.set_tags(tags)
            if keys:
                msg.set_keys(keys)

            # Send message
            result = self.producer.send_sync(msg)

            logger.info(
                "Message sent to RocketMQ",
                extra={
                    "topic": topic,
                    "msg_id": result.msg_id,
                    "status": result.status,
                },
            )

            return result.msg_id

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    def send_async(
        self,
        topic: str,
        body: dict[str, Any],
        callback: callable,
        tags: str = "",
        keys: str = "",
    ) -> None:
        """
        Send message asynchronously.

        Args:
            topic: Topic name
            body: Message body
            callback: Callback function
            tags: Message tags
            keys: Message keys
        """
        if not self.producer or not self._started:
            raise RuntimeError("RocketMQ producer not started")

        try:
            msg_body = json.dumps(body, ensure_ascii=False)
            msg = Message(topic)
            msg.set_body(msg_body.encode("utf-8"))

            if tags:
                msg.set_tags(tags)
            if keys:
                msg.set_keys(keys)

            self.producer.send_async(msg, callback)

            logger.info(
                "Async message sent to RocketMQ",
                extra={"topic": topic},
            )

        except Exception as e:
            logger.error(f"Failed to send async message: {e}")
            raise


class RocketMQConsumer:
    """RocketMQ consumer wrapper."""

    def __init__(self, group_id: str):
        """
        Initialize RocketMQ consumer.

        Args:
            group_id: Consumer group ID
        """
        self.consumer: Optional[PushConsumer] = None
        self.group_id = group_id
        self._started = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def start(self, callback: callable) -> None:
        """
        Start RocketMQ consumer.

        Args:
            callback: Message callback function
        """
        try:
            # Get RocketMQ configuration from Nacos
            nacos_client = get_nacos_client()
            config = nacos_client.get_rocketmq_config()

            # Initialize consumer
            self.consumer = PushConsumer(self.group_id)
            self.consumer.set_name_server_address(config["namesrv_addr"])

            # Set credentials if provided
            if config.get("access_key") and config.get("secret_key"):
                self.consumer.set_session_credentials(
                    config["access_key"],
                    config["secret_key"],
                    "",
                )

            # Register callback
            def message_callback(msg):
                try:
                    body = json.loads(msg.body.decode("utf-8"))
                    callback(body)
                    return True
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    return False

            self.consumer.subscribe(message_callback)

            # Start consumer
            self.consumer.start()
            self._started = True

            logger.info(
                "RocketMQ consumer started",
                extra={
                    "namesrv_addr": config["namesrv_addr"],
                    "group_id": self.group_id,
                },
            )

        except Exception as e:
            logger.error(f"Failed to start RocketMQ consumer: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown RocketMQ consumer."""
        try:
            if self.consumer and self._started:
                self.consumer.shutdown()
                self._started = False
                logger.info("RocketMQ consumer shutdown")
        except Exception as e:
            logger.error(f"Error shutting down RocketMQ consumer: {e}")


# Global RocketMQ producer instance
_rocketmq_producer: Optional[RocketMQProducer] = None


async def init_rocketmq() -> None:
    """Initialize RocketMQ producer."""
    global _rocketmq_producer

    if _rocketmq_producer is None:
        _rocketmq_producer = RocketMQProducer()
        _rocketmq_producer.start()


async def close_rocketmq() -> None:
    """Close RocketMQ producer."""
    global _rocketmq_producer

    if _rocketmq_producer:
        _rocketmq_producer.shutdown()
        _rocketmq_producer = None


def get_rocketmq_producer() -> RocketMQProducer:
    """
    Get RocketMQ producer instance.

    Returns:
        RocketMQProducer instance
    """
    if _rocketmq_producer is None:
        raise RuntimeError(
            "RocketMQ producer not initialized. Call init_rocketmq() first.",
        )
    return _rocketmq_producer

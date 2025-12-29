"""Milvus vector store client for embedding storage and retrieval."""

from typing import Optional

from pymilvus import MilvusClient as PyMilvusClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.app.config.nacos_client import get_nacos_client
from src.app.utils.logger import app_logger as logger


class MilvusClient:
    """Milvus vector database client wrapper."""

    def __init__(self):
        """Initialize Milvus client."""
        self.client: Optional[PyMilvusClient] = None
        self._connected = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def connect(self) -> None:
        """Connect to Milvus server with retry logic."""
        try:
            # Get Milvus configuration from Nacos
            nacos_client = get_nacos_client()
            config = nacos_client.get_milvus_config()

            # Build connection URI
            uri = f"http://{config['host']}:{config['port']}"

            # Initialize Milvus client
            self.client = PyMilvusClient(
                uri=uri,
                user=config.get("user"),
                password=config.get("password"),
                db_name=config.get("db_name", "default"),
            )

            self._connected = True
            logger.info(
                "Connected to Milvus",
                extra={
                    "host": config["host"],
                    "port": config["port"],
                    "db_name": config.get("db_name", "default"),
                },
            )

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            if self.client:
                self.client.close()
                self._connected = False
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")

    def is_connected(self) -> bool:
        """Check if connected to Milvus."""
        return self._connected

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        description: str = "",
    ) -> None:
        """
        Create a new collection in Milvus.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            description: Collection description
        """
        if not self.client:
            raise RuntimeError("Milvus client not connected")

        try:
            # Check if collection exists
            if self.client.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return

            # Create collection with schema
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )

            # Add fields
            schema.add_field(
                field_name="id",
                datatype="INT64",
                is_primary=True,
            )
            schema.add_field(
                field_name="vector",
                datatype="FLOAT_VECTOR",
                dim=dimension,
            )
            schema.add_field(
                field_name="text",
                datatype="VARCHAR",
                max_length=65535,
            )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                description=description,
            )

            logger.info(
                f"Collection {collection_name} created",
                extra={"dimension": dimension},
            )

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def insert_vectors(
        self,
        collection_name: str,
        vectors: list,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> list:
        """
        Insert vectors into collection.

        Args:
            collection_name: Collection name
            vectors: List of vectors
            texts: List of text content
            metadata: Optional metadata for each vector

        Returns:
            List of inserted IDs
        """
        if not self.client:
            raise RuntimeError("Milvus client not connected")

        try:
            data = [
                {
                    "vector": vector,
                    "text": text,
                    **(metadata[i] if metadata else {}),
                }
                for i, (vector, text) in enumerate(zip(vectors, texts))
            ]

            result = self.client.insert(
                collection_name=collection_name,
                data=data,
            )

            logger.info(
                f"Inserted {len(vectors)} vectors into {collection_name}",
            )
            return result

        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vectors: list,
        limit: int = 10,
        output_fields: Optional[list[str]] = None,
    ) -> list:
        """
        Search similar vectors in collection.

        Args:
            collection_name: Collection name
            query_vectors: Query vectors
            limit: Number of results to return
            output_fields: Fields to return in results

        Returns:
            Search results
        """
        if not self.client:
            raise RuntimeError("Milvus client not connected")

        try:
            results = self.client.search(
                collection_name=collection_name,
                data=query_vectors,
                limit=limit,
                output_fields=output_fields or ["text"],
            )

            logger.info(
                f"Searched {collection_name}",
                extra={"limit": limit, "results": len(results)},
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise


# Global Milvus client instance
_milvus_client: Optional[MilvusClient] = None


async def init_milvus() -> None:
    """Initialize Milvus client."""
    global _milvus_client

    if _milvus_client is None:
        _milvus_client = MilvusClient()
        _milvus_client.connect()


async def close_milvus() -> None:
    """Close Milvus client."""
    global _milvus_client

    if _milvus_client:
        _milvus_client.disconnect()
        _milvus_client = None


def get_milvus_client() -> MilvusClient:
    """
    Get Milvus client instance.

    Returns:
        MilvusClient instance
    """
    if _milvus_client is None:
        raise RuntimeError("Milvus client not initialized. Call init_milvus() first.")
    return _milvus_client

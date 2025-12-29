"""Milvus vector store client for embedding storage and retrieval."""

from typing import Optional
from pymilvus import MilvusClient as PyMilvusClient
from app.config.nacos_client import get_nacos_client


class MilvusClient:
    """Milvus vector database client wrapper."""

    _instance: Optional["MilvusClient"] = None

    def __init__(self):
        """Initialize Milvus client."""
        self.client: Optional[PyMilvusClient] = None
        self._connected = False
        self._config = None


    def _initialize(self) -> None:
        """Initialize Milvus client."""
        try:
            nacos_client = get_nacos_client()
            self._config = nacos_client.get_milvus_config()
            self.client = PyMilvusClient(uri=str(self._config["uri"]), token=str(self._config["token"]))
            self._connected = True
            database_name = str(self._config["db_name"])
            if database_name not in self.client.list_databases():
                self.client.create_database(database_name)

            else:
                print(f"Database '{database_name}' already exists.")
        except Exception as e:
            print(e)


from typing import Optional
import tempfile
import os
from pathlib import Path

from app.config.nacos_client import get_nacos_client
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from app.services.embedding_service import get_embedding_service
from app.utils.logger import app_logger as logger


class RAGService:

    _instance: Optional["RAGService"] = None

    def __init__(self):
        """
        初始化 RAG 服务参数，从 nacos 取，配置形如：
        ```yaml
        # RAG 存储配置 - PostgreSQL DocStore & IndexStore
        rag:
          storage:
            postgres:
              # 文档存储 (DocStore)
              docstore:
                host: localhost
                port: 15432
                database: shopmind_rag
                user: postgres
                password: hcy991002
                table_name: platform_rule_docs
                schema_name: public
                perform_setup: true   # 首次启动时自动建表
                debug: false
                use_jsonb: true       # 强烈建议开启：使用 JSONB 类型提升查询性能

              # 索引元数据存储 (IndexStore) - 虽 VectorStoreIndex 不强制需要，但保留以备扩展
              indexstore:
                host: localhost
                port: 15432
                database: shopmind_rag
                user: postgres
                password: hcy991002
                table_name: platform_rule_indices
                schema_name: public
                perform_setup: true
                debug: false
                use_jsonb: true

              # 向量存储 (Milvus) - 注意，维度由 embedding service 配置项进行提供，不可自定义
              milvus:
                host: localhost
                port: 19530
                token: root:Milvus
                # 不存在，则创建
                db_name: platform_rules
                collection_name: platform_rules_collection
                similarity_metric: COSINE
                index_type: HNSW
                HNSW:
                  params_M: 8
                  params_efConstruction: 100
        ```
        """
        self.rag_config = get_nacos_client().get_rag_config()
    
    def init_docstore(self) -> PostgresDocumentStore:
        """
        初始化文档存储
        
        Returns:
            PostgresDocumentStore 实例
        """
        docstore_cfg = self.rag_config["storage"]["postgres"]["docstore"]
        return PostgresDocumentStore.from_params(
            host=docstore_cfg["host"],
            port=str(docstore_cfg["port"]),
            database=docstore_cfg["database"],
            user=docstore_cfg["user"],
            password=docstore_cfg["password"],
            table_name=docstore_cfg["table_name"],
            schema_name=docstore_cfg["schema_name"],
            perform_setup=docstore_cfg["perform_setup"],
            debug=docstore_cfg["debug"],
            use_jsonb=docstore_cfg["use_jsonb"],
        )

    def init_index_store(self) -> PostgresIndexStore:
        """
        初始化索引存储
        
        Returns:
            PostgresIndexStore 实例
        """
        indexstore_cfg = self.rag_config["storage"]["postgres"]["indexstore"]
        return PostgresIndexStore.from_params(
            host=indexstore_cfg["host"],
            port=str(indexstore_cfg["port"]),
            database=indexstore_cfg["database"],
            user=indexstore_cfg["user"],
            password=indexstore_cfg["password"],
            table_name=indexstore_cfg["table_name"],
            schema_name=indexstore_cfg["schema_name"],
            perform_setup=indexstore_cfg["perform_setup"],
            debug=indexstore_cfg["debug"],
            use_jsonb=indexstore_cfg["use_jsonb"],
        )

    def init_vector_store(self) -> MilvusVectorStore:
        """
        初始化向量存储（Milvus）
        
        Returns:
            MilvusVectorStore 实例
        """
        vector_store_cfg = self.rag_config["storage"]["milvus"]
        
        # 获取 embedding 维度
        embedding_service = get_embedding_service()
        dim = embedding_service.text_model_dim
        
        # 构建 Milvus URI
        uri = f"http://{vector_store_cfg['host']}:{vector_store_cfg['port']}"
        
        return MilvusVectorStore(
            uri=uri,
            token=vector_store_cfg["token"],
            db_name=vector_store_cfg["db_name"],
            dim=dim,
            collection_name=vector_store_cfg["collection_name"],
            embedding_field="embedding",
            doc_id_field="doc_id",
            similarity_metric=vector_store_cfg["similarity_metric"],
            consistency_level="Strong",
            overwrite=False,
            index_config={
                "index_type": vector_store_cfg["index_type"],
                "metric_type": vector_store_cfg["similarity_metric"],
                "params": {
                    "M": vector_store_cfg["HNSW"]["params_M"],
                    "efConstruction": vector_store_cfg["HNSW"]["params_efConstruction"]
                }
            }
        )
    

    def _initialize(self) -> None:
        """
        初始化 RAG 服务
        """
        # 初始化存储组件
        self.docstore = self.init_docstore()
        self.index_store = self.init_index_store()
        self.vector_store = self.init_vector_store()
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore, index_store=self.index_store, vector_store=self.vector_store)  # pyright: ignore[reportUndefinedVariable]
        
        # 直接使用 EmbeddingService 中的 LlamaIndex Embedding（已经是 DashScopeEmbedding）
        self.embed_model = get_embedding_service().provider.text_embeddings
        logger.info(f"RAG 服务使用统一的 Embedding 模型: {get_embedding_service().text_model}")


    def get_index(self) -> VectorStoreIndex:
        """获取索引，用于增删改查"""
        # 检查索引是否存在
        index_structs = list(self.index_store.index_structs())
        
        if index_structs:
            # 索引已存在，从存储加载
            logger.info("索引已存在，从存储加载")
            index = load_index_from_storage(
                self.storage_context, 
                embed_model=self.embed_model
            )
        else:
            # 索引不存在，创建新的空索引
            logger.info("索引不存在，创建新的空索引")
            index = VectorStoreIndex.from_documents(
                [],  # 空文档列表
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
        
        return index
    
    async def upload_document(self, file_content: bytes, filename: str, category: str = "platform_rule") -> dict:
        """
        上传并索引文档
        
        Args:
            file_content: 文件内容
            filename: 文件名
            category: 分类
            
        Returns:
            上传结果
        """
        try:
            # 创建临时文件
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / filename
                file_path.write_bytes(file_content)
                
                # 读取文档
                documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                
                # 添加元数据
                for doc in documents:
                    doc.metadata["filename"] = filename
                    doc.metadata["category"] = category
                
                logger.info(f"读取到 {len(documents)} 个 Document，准备插入索引")
                
                # 获取索引（如果不存在会自动创建空索引）
                index = self.get_index()
                
                # 手动分割文档为节点
                from llama_index.core.node_parser import SentenceSplitter
                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                nodes = splitter.get_nodes_from_documents(documents)
                
                logger.info(f"文档已分割为 {len(nodes)} 个节点")
                
                # 先保存节点到 DocStore（关键步骤！）
                self.docstore.add_documents(nodes, allow_update=True, store_text=True)
                logger.info(f"节点已保存到 DocStore，当前总节点数: {len(self.docstore.docs)}")
                
                # 再插入到向量索引
                index.insert_nodes(nodes)
                logger.info(f"节点已插入到向量索引")
                
                # 收集文档 ID
                doc_ids = list(set([node.ref_doc_id for node in nodes if node.ref_doc_id]))
                
                logger.info(f"文档上传成功: {filename}, 生成了 {len(nodes)} 个节点, 引用文档 ID: {doc_ids}")
                
                return {
                    "success": True,
                    "filename": filename,
                    "doc_ids": doc_ids,
                    "node_count": len(nodes),
                    "doc_count": len(documents)
                }
        except Exception as e:
            logger.error(f"文档上传失败: {e}", exc_info=True)
            raise
    
    async def list_documents(self) -> list[dict]:
        """
        列出所有文档（按文件名去重，一个文件只显示一次）
        
        Returns:
            文档列表
        """
        try:
            # 从索引获取 docstore（确保获取最新数据）
            index = self.get_index()

            docstore = index.docstore

            # 获取所有文档块
            docs = docstore.docs
            
            logger.info(f"从 DocStore 获取到 {len(docs)} 个文档块")
            
            # 按 filename 去重
            file_map = {}
            for doc_id, doc in docs.items():
                filename = doc.metadata.get("filename", "未知")
                
                # 如果文件名还没记录，则添加
                if filename not in file_map:
                    file_map[filename] = {
                        "filename": filename,
                        "category": doc.metadata.get("category", "未分类"),
                        "text_preview": doc.text[:100] if doc.text else "",
                    }
            
            result = list(file_map.values())
            
            logger.info(f"查询文档列表: {len(result)} 个文件（已去重）")
            return result
        except Exception as e:
            logger.error(f"查询文档列表失败: {e}", exc_info=True)
            raise
    
    async def delete_document(self, filename: str) -> bool:
        """
        删除文档（删除该文件的所有块）
        
        Args:
            filename: 文件名
            
        Returns:
            是否删除成功
        """
        try:
            # 获取最新索引和 docstore
            index = self.get_index()
            docstore = index.docstore
            
            # 找到该文件的所有节点 ID 和 ref_doc_id（一个文件只有一个 ref_doc_id）
            node_ids_to_delete = []
            ref_doc_id = None
            
            for node_id, node in docstore.docs.items():
                if node.metadata.get("filename") == filename:
                    node_ids_to_delete.append(node_id)
                    if node.ref_doc_id and ref_doc_id is None:
                        ref_doc_id = node.ref_doc_id
            
            if not node_ids_to_delete or ref_doc_id is None:
                logger.warning(f"未找到文件: {filename}")
                return False
            
            logger.info(f"准备删除文件 {filename}: ref_doc_id={ref_doc_id}, {len(node_ids_to_delete)} 个节点")
            
            # 1. 从 VectorStore 和 IndexStore 删除（使用 ref_doc_id）
            try:
                index.delete_ref_doc(ref_doc_id, delete_from_docstore=False)
                logger.info(f"已从 VectorStore 和 IndexStore 删除 ref_doc_id: {ref_doc_id}")
            except Exception as e:
                logger.warning(f"从 VectorStore/IndexStore 删除失败: {e}")
            
            # 2. 手动从 DocStore 删除所有节点
            for node_id in node_ids_to_delete:
                try:
                    docstore.delete_document(node_id, raise_error=False)
                    logger.info(f"已从 DocStore 删除 node_id: {node_id}")
                except Exception as e:
                    logger.warning(f"从 DocStore 删除 {node_id} 失败: {e}")
            
            logger.info(f"文档删除成功: {filename}, 删除了 ref_doc_id={ref_doc_id}, {len(node_ids_to_delete)} 个节点")
            return True
        except Exception as e:
            logger.error(f"文档删除失败: {e}", exc_info=True)
            return False
    
    @classmethod
    def get_instance(cls) -> "RAGService":
        """
        获取 RAGService 单例.

        Returns:
            RAGService 实例
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

def get_rag_service() -> RAGService:
    """获取 RAGService 单例（便捷函数）."""
    return RAGService.get_instance()

def init_rag_service() -> None:
    """初始化 RAGService"""
    get_rag_service()
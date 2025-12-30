"""
@File       : product_collection.py
@Description: Product collection 管理模块

@Time       : 2025/12/29 20:52
@Author     : hcy18
"""
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from app.services.embedding_service import get_embedding_service
from app.utils.logger import app_logger as logger

COLLECTION_NAME = "product_collection"


def create_product_collection_schema() -> CollectionSchema:
    """
    创建 product collection 的 schema.

    Returns:
        CollectionSchema
    """
    # 获取嵌入维度
    embedding_service = get_embedding_service()
    vector_dim = embedding_service.text_model_dim

    logger.info(f"创建 product collection schema，向量维度: {vector_dim}")

    # 定义字段
    fields = [
        # 主键字段：自动生成的唯一 ID（使用雪花算法）
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=True,  # 自动生成 ID
            description="主键 ID，自动生成"
        ),
        
        # 商品 ID：业务主键
        FieldSchema(
            name="product_id",
            dtype=DataType.INT64,
            description="商品 ID（业务主键）"
        ),
        
        # 价格字段：使用 DOUBLE 类型存储价格
        FieldSchema(
            name="price",
            dtype=DataType.DOUBLE,
            description="商品价格"
        ),
        
        # 密集向量字段：存储商品的嵌入向量
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=vector_dim,
            description="商品嵌入向量（由标题+描述+摘要+标签组成）"
        ),
    ]

    # 创建 schema
    schema = CollectionSchema(
        fields=fields,
        description="商品向量数据库 Collection",
        enable_dynamic_field=False,  # 不允许动态字段
    )

    return schema


def init_product_collection() -> Collection:
    """
    初始化 product collection（创建或加载）.

    Returns:
        Collection 实例
    """
    try:
        # 检查 collection 是否已存在
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' 已存在，加载现有 collection")
            collection = Collection(name=COLLECTION_NAME)
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' 不存在，创建新的 collection")
            
            # 创建 schema
            schema = create_product_collection_schema()
            
            # 创建 collection
            collection = Collection(
                name=COLLECTION_NAME,
                schema=schema,
                using='default',
                shards_num=2,  # 分片数量
            )
            
            logger.info(f"Collection '{COLLECTION_NAME}' 创建成功")

        # 创建索引（如果尚未创建）
        _create_indexes(collection)

        # 加载 collection 到内存
        collection.load()
        logger.info(f"Collection '{COLLECTION_NAME}' 已加载到内存")

        return collection

    except Exception as e:
        logger.error(f"初始化 product collection 失败: {e}", exc_info=True)
        raise


def _create_indexes(collection: Collection) -> None:
    """
    为 collection 创建索引.

    Args:
        collection: Collection 实例
    """
    try:
        # 如果没有索引，创建 HNSW 索引
        if not collection.indexes:
            logger.info("创建向量索引（HNSW）")
            
            # HNSW 索引参数
            index_params = {
                "index_type": "HNSW",  # HNSW 索引类型
                "metric_type": "COSINE",  # 余弦相似度
                "params": {
                    "M": 16,  # 每个节点的最大连接数
                    "efConstruction": 200,  # 构建索引时的搜索范围
                },
            }
            
            # 创建向量索引
            collection.create_index(
                field_name="embedding",
                index_params=index_params,
                index_name="embedding_index"
            )
            
            logger.info("向量索引创建成功")
        else:
            logger.info("向量索引已存在，跳过创建")

        # 主键索引由 Milvus 自动创建，无需手动创建

    except Exception as e:
        logger.error(f"创建索引失败: {e}", exc_info=True)
        raise


def get_product_collection() -> Collection:
    """
    获取 product collection 实例（便捷函数）.

    Returns:
        Collection 实例
    """
    if not utility.has_collection(COLLECTION_NAME):
        return init_product_collection()
    return Collection(name=COLLECTION_NAME)

# AI Service - 智能电商平台 AI 微服务

这是一个专业的 AI 微服务项目，为智能电商平台提供 AI 能力支持，包括商品标题审核、图片审核和商品描述生成等功能。

## 🚀 技术栈

- **Web 框架**: FastAPI 0.115+
- **AI 框架**: LangChain 0.3+, LangGraph 0.2+, LlamaIndex 0.12+
- **数据库**: PostgreSQL 18 (使用 SQLAlchemy 2.0 + asyncpg)
- **向量数据库**: Milvus 2.4+
- **消息队列**: RocketMQ 5.1.4
- **服务注册与配置**: Nacos
- **包管理器**: uv
- **LLM**: OpenAI GPT-4o / GPT-4o-mini (支持扩展通义千问等)

## 📁 项目结构

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 应用入口
│   ├── config/                      # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py              # 配置类定义
│   │   └── nacos_client.py          # Nacos 客户端封装
│   ├── db/                          # 数据库
│   │   ├── __init__.py
│   │   ├── database.py              # PostgreSQL 连接管理
│   │   └── models.py                # 数据模型
│   ├── vector_store/                # 向量数据库
│   │   ├── __init__.py
│   │   └── milvus_client.py         # Milvus 客户端封装
│   ├── mq/                          # 消息队列
│   │   ├── __init__.py
│   │   └── rocketmq_client.py       # RocketMQ 客户端封装
│   ├── services/                    # 业务服务层
│   │   ├── __init__.py
│   │   ├── llm_service.py           # LLM 服务抽象层
│   │   ├── title_check_service.py   # 标题审核服务
│   │   ├── image_check_service.py   # 图片审核服务
│   │   └── description_service.py   # 描述生成服务
│   ├── chains/                      # LangChain 链
│   │   ├── __init__.py
│   │   ├── title_check_chain.py     # 标题审核链
│   │   ├── image_check_chain.py     # 图片审核链
│   │   └── description_chain.py     # 描述生成链
│   ├── schemas/                     # Pydantic 数据模型
│   │   ├── __init__.py
│   │   ├── title_check.py           # 标题审核模型
│   │   ├── image_check.py           # 图片审核模型
│   │   └── description.py           # 描述生成模型
│   ├── routers/                     # API 路由
│   │   ├── __init__.py
│   │   └── ai_router.py             # AI 相关路由
│   └── utils/                       # 工具函数
│       ├── __init__.py
│       └── logger.py                # 日志配置
├── .env.example                     # 环境变量示例
├── .gitignore                       # Git 忽略文件
├── pyproject.toml                   # 项目配置和依赖
└── README.md                        # 项目说明文档
```

## ⚙️ 安装和配置

### 1. 环境要求

- Python 3.12+
- uv (推荐) 或 pip
- PostgreSQL 18
- Milvus 2.4+
- Nacos 2.x
- RocketMQ 5.1.4

### 2. 安装依赖

使用 uv 安装依赖（推荐）:

```bash
# 安装 uv (如果还没有安装)
pip install uv

# 同步依赖
uv sync
```

或使用 pip:

```bash
pip install -e .
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置:

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的参数：

```env
# Nacos 配置 (必须)
NACOS_SERVER_ADDR=127.0.0.1:8848
NACOS_NAMESPACE=public
NACOS_GROUP=DEFAULT_GROUP
NACOS_DATA_ID=ai-service.yaml

# OpenAI API Key (必须)
OPENAI_API_KEY=your-openai-api-key-here

# 其他配置可以从 Nacos 读取
```

### 4. Nacos 配置中心

在 Nacos 中创建配置 `ai-service.yaml`，内容示例：

```yaml
# PostgreSQL 配置
postgresql:
  host: localhost
  port: 5432
  user: postgres
  password: your-password
  database: shopmind
  pool_size: 10
  max_overflow: 20

# Milvus 配置
milvus:
  host: localhost
  port: 19530
  user: null
  password: null
  db_name: default

# RocketMQ 配置
rocketmq:
  namesrv_addr: 127.0.0.1:9876
  access_key: null
  secret_key: null
  group_id: ai-service-group

# LLM 配置
llm:
  provider: openai
  openai:
    api_key: your-openai-api-key-here
    api_base: null
    model: gpt-4o-mini
    vision_model: gpt-4o
  temperature: 0.7
  max_tokens: 2000
  timeout: 60
```

## 🚀 启动服务

### 开发模式

```bash
# 使用 uvicorn 启动 (带热重载)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或使用 uv run
uv run uvicorn app.main:app --reload
```

### 生产模式

```bash
# 使用 Python 直接运行
python -m app.main

# 或使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

服务启动后，访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## 📡 API 接口

### 1. 标题审核

**接口**: `POST /ai/title-check`

检查商品标题是否符合平台规范。

**请求示例**:
```json
{
  "title": "高品质纯棉T恤 男女通用 透气舒适"
}
```

**响应示例**:
```json
{
  "valid": true,
  "reason": null,
  "suggestions": null
}
```

不合规示例:
```json
{
  "valid": false,
  "reason": "包含夸大宣传词汇",
  "suggestions": [
    "移除'史上最好'等绝对化用语",
    "使用更客观的描述词"
  ]
}
```

### 2. 图片审核

**接口**: `POST /ai/image-check`

检查商品图片是否符合平台规范。

**请求示例**:
```json
{
  "imageUrl": "https://example.com/product.jpg"
}
```

**响应示例**:
```json
{
  "valid": true,
  "reason": null
}
```

### 3. 商品描述生成

**接口**: `POST /ai/description-generate`

根据商品标题、图片和分类生成营销描述。

**请求示例**:
```json
{
  "title": "高品质纯棉T恤",
  "imageUrls": [
    "https://example.com/product1.jpg",
    "https://example.com/product2.jpg"
  ],
  "category": "服装/T恤"
}
```

**响应示例**:
```json
{
  "description": "这款高品质纯棉T恤采用100%纯棉面料，透气舒适，柔软亲肤。精致剪裁，版型简约大方，适合多种场合穿搭。优质做工，耐洗耐穿，是您衣橱中的百搭单品。"
}
```

## 🏗️ 架构设计

### 服务架构

```
┌─────────────────┐
│   Nacos         │  服务注册与配置中心
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────────┐
│        AI Service (FastAPI)         │
│  ┌──────────────────────────────┐   │
│  │   API Layer (Routers)        │   │
│  └──────────┬───────────────────┘   │
│             ↓                        │
│  ┌──────────────────────────────┐   │
│  │   Service Layer              │   │
│  │  - TitleCheckService         │   │
│  │  - ImageCheckService         │   │
│  │  - DescriptionService        │   │
│  └──────────┬───────────────────┘   │
│             ↓                        │
│  ┌──────────────────────────────┐   │
│  │   Chain Layer (LangChain)    │   │
│  │  - TitleCheckChain           │   │
│  │  - ImageCheckChain           │   │
│  │  - DescriptionChain          │   │
│  └──────────┬───────────────────┘   │
│             ↓                        │
│  ┌──────────────────────────────┐   │
│  │   LLM Service (抽象层)        │   │
│  │  - OpenAI Provider           │   │
│  │  - Tongyi Provider (扩展)    │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
         │            │            │
         ↓            ↓            ↓
   PostgreSQL      Milvus      RocketMQ
```

### 关键设计

1. **配置管理**: 优先从 Nacos 读取配置，本地 `.env` 作为 fallback
2. **服务抽象**: LLM 服务支持多 provider，易于切换和扩展
3. **异步设计**: 所有 IO 操作使用 `async/await`，提高并发性能
4. **错误处理**: 完善的异常捕获和日志记录
5. **结构化日志**: 使用 JSON 格式日志，便于日志收集和分析

## 🔧 开发指南

### 添加新的 LLM Provider

1. 在 `app/services/llm_service.py` 中创建新的 Provider 类
2. 实现 `get_chat_model()` 和 `get_vision_model()` 方法
3. 在 `LLMService._initialize()` 中注册新 provider

### 添加新的 AI 功能

1. 在 `app/schemas/` 创建请求/响应模型
2. 在 `app/chains/` 创建 LangChain 链
3. 在 `app/services/` 创建服务类
4. 在 `app/routers/ai_router.py` 添加路由

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解 (Type Hints)
- 函数和类添加文档字符串
- 使用 `black` 格式化代码
- 使用 `ruff` 进行代码检查

## 📊 监控和日志

### 日志格式

服务使用结构化 JSON 日志：

```json
{
  "asctime": "2025-12-23T10:00:00",
  "name": "ai_service",
  "levelname": "INFO",
  "message": "Title check completed",
  "title": "商品标题",
  "valid": true
}
```

### 健康检查

- 服务健康检查: `GET /health`
- AI 服务健康检查: `GET /ai/health`

## 🐛 故障排查

### 常见问题

1. **无法连接 Nacos**
   - 检查 `NACOS_SERVER_ADDR` 配置
   - 确认 Nacos 服务已启动

2. **数据库连接失败**
   - 检查 PostgreSQL 服务状态
   - 验证数据库配置 (host, port, credentials)

3. **Milvus 连接失败**
   - 确认 Milvus 服务已启动
   - 检查 Milvus 端口 (默认 19530)

4. **LLM 调用失败**
   - 验证 OpenAI API Key 是否正确
   - 检查网络连接和代理设置
   - 查看日志中的详细错误信息

## 📝 待扩展功能

- [ ] RAG (检索增强生成) 功能
- [ ] 商品推荐系统
- [ ] 更多 LLM Provider 支持 (通义千问、文心一言)
- [ ] 请求日志持久化到数据库
- [ ] 性能监控和统计
- [ ] 接口限流和熔断
- [ ] 多模型负载均衡

## 📄 License

MIT License

## 👥 贡献者

- 由 Claude Code 协助搭建

## 📧 联系方式

如有问题，请提交 Issue 或联系开发团队。

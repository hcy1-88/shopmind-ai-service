# AI Service - 智能电商平台 AI 微服务

这是一个专业的 AI 微服务项目，为智能电商平台提供 AI 能力支持，包括商品标题审核、图片审核和商品描述生成等功能。

## 🚀 技术栈

- **Web 框架**: FastAPI 0.115+
- **AI 框架**: LangChain 0.3+, LangGraph 0.2+, LlamaIndex 0.12+
- **向量数据库**: Milvus 2.4+
- **服务注册与配置**: Nacos
- **包管理器**: uv
- **LLM**: OpenAI GPT-4o / GPT-4o-mini (支持扩展通义千问等)

## 🎯 核心功能

### LangChain 应用

#### 智能审核
- **商品标题审核**：自动检测商品标题中的违规内容，包括敏感词、虚假宣传等
- **商品图片审核**：基于视觉模型识别商品图片中的违规内容

#### 内容生成
- **商品描述生成**：根据商品信息自动生成吸引人的商品描述文案
- 扩展支持：评价摘要、营销文案等

### LangGraph 应用

#### 智能对话 Agent
- **多轮对话管理**：基于 Redis Checkpoint 的会话状态管理
- **工具调用**：支持商品搜索、新品推荐、知识库检索等工具
- **流式输出**：支持实时流式对话响应
- **会话历史**：自动保存和获取对话历史，支持上下文记忆

#### 商品工具
- `search_product`：根据用户需求智能搜索商品
- `get_new_product`：获取平台最新上架商品
- 支持分页查询和结果限制

### LlamaIndex RAG 应用

#### 知识库检索
- **向量检索**：基于 Milvus 的语义向量检索
- **文档索引**：支持平台规则、政策文档的索引管理
- **相似度搜索**：快速检索相关知识片段

#### 平台知识搜索工具
- `platform_knowledge_search`：从知识库检索平台规则、流程等信息
- 返回检索到的文档片段，由 Agent 统一生成回复

## ⚙️ 安装和配置

### 1. 环境要求

- Python 3.12+
- uv (推荐) 或 pip
- Milvus 2.4+
- Nacos 2.x

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

# Application Settings 必须
APP_NAME=shopmind-ai-service
APP_VERSION=0.1.0
DEBUG=true
LOG_LEVEL=INFO

# Service Configuration 必须
SERVICE_NAME=shopmind-ai-service
SERVICE_PORT=8085
SERVICE_CLUSTER=DEFAULT

# Nacos Configuration 必须
NACOS_SERVER_ADDR=127.0.0.1:8848
NACOS_NAMESPACE=shopmind-dev
NACOS_GROUP=DEFAULT_GROUP
NACOS_DATA_ID=shopmind-ai-service.yaml
NACOS_USERNAME=nacos
NACOS_PASSWORD=nacos


# 其他配置可以从 Nacos 读取
```

### 4. Nacos 配置中心

在 Nacos 中创建配置 `ai-service.yaml`，内容示例：

```yaml

# Milvus 配置
milvus:
  host: localhost
  port: 19530
  user: null
  password: null
  db_name: default


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



## 📧 联系方式

如有问题，请提交 Issue 或联系负责人 19011289503。

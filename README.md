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
- **流式输出**：支持实时流式对话响应（SSE）
- **会话历史**：自动保存和获取对话历史，支持上下文记忆

##### 核心能力

- **多意图识别与多任务并行**：支持将用户复杂query拆分为多个独立子任务（SHOPPING、PLATFORM、CHITCHAT、COMPARISON），并行执行后聚合结果
- **多轮澄清**：槽位不足时主动向用户提问澄清（最多3轮），支持品类、关键词、过滤条件等槽位提取
- **商品过滤**：基于LLM语义理解对搜索结果进行智能过滤，排除不符合条件的商品
- **换一批**：支持用户浏览更多商品，自动翻页搜索并排除已展示商品
- **状态持久化**：通过 Redis Checkpoint 实现多会话状态恢复

##### 状态机

购物子任务采用状态机驱动：
- `NEW` → `CLARIFYING` → `READY` → `COMPLETED`
- 槽位齐全触发 `READY`，否则进入 `CLARIFYING` 等待用户澄清

#### 商品工具
- `search_product`：根据自然语言query搜索商品，支持分页
- `get_product_detail`：获取商品详细信息（款式、价格、库存等SKU规格）
- `get_new_product`：获取平台最新上架商品
- `platform_knowledge_search`：RAG知识库检索
- `tavily_search`：网页搜索
- `get_current_weather` / `get_forecast_weather`：天气查询

### LlamaIndex RAG 应用

#### 知识库检索
- **向量检索**：基于 Milvus 的语义向量检索
- **文档索引**：支持平台规则、政策文档的索引管理
- **相似度搜索**：快速检索相关知识片段

#### 平台知识搜索工具
- `platform_knowledge_search`：从知识库检索平台规则、流程等信息
- 返回检索到的文档片段，由 Agent 统一生成回复

## 🏗️ 导购 Agent 架构设计

### 主图结构（ShopmindAgentGraph）

```
START
  │
  ▼
query_rewrite_node  (重写用户query)
  │
  ▼
intent_decomposer_node  (意图分解：SHOPPING / PLATFORM / CHITCHAT / COMPARISON)
  │
  ▼
route_to_map_node_edge  (条件路由，并行分发)
  │
      ┌─────────────┬──────────────┬──────────────┐
      ▼             ▼              ▼              ▼
shopping_     platform_      chitchat_     comparison_
subgraph_node  node          node         subgraph_node
  │             │              │              │
  └─────────────┴──────────────┴──────────────┘
              │
              ▼
        aggregator_node  (聚合各子任务结果)
              │
              ▼
             END
```

![Agent 架构图](./docs/agent-architecture.drawio.png)

### 导购子图结构（ShoppingSubgraph）

```
dispatcher_node
      │
      ▼
route_by_status_edge
      │
  ┌───┴───┐
  ▼       ▼
clarifying_  ready_node
node       (槽位齐全)
  │           │
  │       ┌──┴──┐
  │       ▼     ▼
  │   tool_node filter_node
  │   (调用工具)  │
  │       │     │
  │       └─┬───┘
  │         ▼
  │   router_after_filter
  │         │
  │    ┌────┴────┐
  │    ▼         ▼
  │ generate_  ready_node
  │ node     (翻页搜索)
  │    │
  │    └──────────┘
  │         │
  └─────────┘
```

### 商品比较子图结构（ComparisonSubgraph）

当用户使用代词（"这几个"、"那些"）指代已推荐的商品时，触发比较子图：

```
START
  │
  ▼
detail_node  (获取商品详情)
  │
  ▼
compare_node  (LLM生成对比文案和购买建议)
  │
  ▼
  END
```

**触发条件**：
- 用户说了比较类词汇（"比较"、"区别"、"选哪个"）
- 用户使用了代词（"这几个"、"那些"）
- 历史的 ShoppingSubTask 中 `has_recommended_product_ids` 非空

**路由规则**：
- 有代词 + 有推荐商品 → COMPARISON → comparison_subgraph（商品级比较）
- 无代词或无推荐商品 → CHITCHAT（品牌级讨论/泛泛而谈）

### 核心组件

| 组件 | 职责 |
|------|------|
| `query_rewrite_node` | 重写用户query，结合历史上下文优化表达 |
| `intent_decomposer_node` | 意图分解、槽位提取、判断新旧意图 |
| `searching_subgraph_node` | 购物意图处理，内含完整的状态机子图 |
| `platform_node` | 平台规则/政策问答 |
| `chitchat_node` | 闲聊问答 |
| `comparison_subgraph_node` | 商品比较处理，调用商品详情工具生成对比文案 |
| `aggregator_node` | 聚合多任务结果，生成最终回复 |

### 导购子图节点

| 节点 | 职责 |
|------|------|
| `dispatcher_node` | 任务分发，日志记录 |
| `clarifying_node` | 槽位不足时生成澄清问题 |
| `ready_node` | 槽位齐全，调用search_product/get_product_detail工具 |
| `tool_node` | 执行工具调用，收集搜索结果 |
| `filter_node` | LLM语义过滤，排除不符合条件的商品 |
| `generate_node` | 生成最终商品推荐文案 |

### 比较子图节点

| 节点 | 职责 |
|------|------|
| `detail_node` | 从 `has_recommended_product_ids` 获取商品ID，并行调用 `get_product_detail` 获取详情 |
| `compare_node` | 将商品详情传给 LLM 生成对比文案和购买建议 |

### 关键设计

1. **父子图状态共享**：通过 LangGraph 的 `operator.add` 注解实现 `messages`、`sub_task_results` 在父子图间自动合并
2. **换一批实现**：通过 `is_replace_products` 标记控制 `filter_node` 排除已推荐商品，`ready_node` 自动翻页搜索
3. **最大搜索循环**：通过 `max_search_loop` 控制分页上限，避免无限翻页
4. **澄清次数限制**：通过 `clarification_count` 控制澄清轮次上限（默认3轮）
5. **商品比较路由**：通过 `has_recommended_product_ids` 判断是否为已推荐商品的比较，有则走 COMPARISON 子图

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

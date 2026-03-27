# ShopMind Agent 设计方案

## 1. 整体架构

ShopMind Agent 采用 **LangGraph** 构建的**状态机编排**架构，主图（父图）负责顶层流程编排，子图（Subgraph）负责特定领域的任务执行。

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         ShopmindAgentGraph (主图)                        │
│                                                                            │
│  START ──► query_rewrite ──► intent_decomposer                            │
│                                          │                                │
│                              route_to_map_node_edge                       │
│                                 (Send 并行分发)                           │
│                              ┌──────┴──────┬───────────┬───────────┐   │
│                              ▼             ▼           ▼            ▼   │
│                    searching_      platform    chitchat    comparison_    │
│                    subgraph_node   _node       _node     subgraph_node  │
│                         (子图)                  (子图)                    │
│                              │             │           │              │
│                              └─────────────┴───────────┴──────────────┘  │
│                                      │                                   │
│                                 aggregator                               │
│                                   _node                                 │
│                                      │                                   │
│                                     END                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1.1 设计原则

- **层级编排**：主图负责流程调度，子图负责具体执行
- **并行处理**：多意图任务通过 `Send` 并行分发到不同节点
- **状态驱动**：通过 `TaskStatus` 状态机控制任务生命周期
- **单例模式**：所有 Graph 和 Agent 均采用单例模式，避免重复创建开销

---

## 2. 主图设计 (ShopmindAgentGraph)

### 2.1 节点 (Nodes)

| 节点名称 | 功能说明 |
|---------|---------|
| `query_rewrite_node` | 对用户原始 query 进行改写，增强表达能力 |
| `intent_decomposer_node` | 意图分解，将改写后的 query 拆分为多个子任务 |
| `searching_subgraph_node` | 调用搜索子图，处理 SHOPPING 意图 |
| `platform_node` | 处理 PLATFORM 意图（平台规则/政策查询） |
| `chitchat_node` | 处理 CHITCHAT 意图（闲聊、天气等） |
| `comparison_subgraph_node` | 调用比较子图，处理 COMPARISON 意图（商品比较） |
| `aggregator_node` | 聚合所有子任务的执行结果，生成最终回复 |

### 2.2 边 (Edges)

```
START
  │
  ▼
query_rewrite_node
  │
  ▼
intent_decomposer_node
  │
  ▼ [conditional_edges: route_to_map_node_edge]
  │
  ├───── Send("searching_subgraph_node") ───▶ searching_subgraph_node
  ├───── Send("platform_node") ──────────────▶ platform_node
  ├───── Send("chitchat_node") ─────────────▶ chitchat_node
  └───── Send("comparison_subgraph_node") ───▶ comparison_subgraph_node

searching_subgraph_node ───┐
                           │
platform_node ─────────────┼──▶ aggregator_node ──▶ END
                           │
chitchat_node ─────────────┤
                           │
comparison_subgraph_node ──┘
```

### 2.3 条件路由 (route_to_map_node_edge)

根据 `IntentCategory` 将子任务分发到对应的处理器：

```python
if task.category == IntentCategory.SHOPPING:
    → searching_subgraph_node
elif task.category == IntentCategory.PLATFORM:
    → platform_node
elif task.category == IntentCategory.COMPARISON:
    → comparison_subgraph_node
else:
    → chitchat_node
```

---

## 3. 购物子图设计 (ShoppingSubgraph)

购物子图是一个**状态机**，用于处理商品搜索和推荐。

### 3.1 节点 (Nodes)

| 节点名称 | 功能说明 |
|---------|---------|
| `dispatcher_node` | 任务分发器，根据状态分发到澄清或就绪路径 |
| `clarifying_node` | 生成澄清问题，等待用户补充信息 |
| `ready_node` | 槽位齐全，执行商品搜索准备 |
| `tool_node` | 调用搜索工具，执行商品检索 |
| `filter_node` | 对搜索结果进行 LLM 语义过滤 |
| `generate_node` | 生成商品推荐文案 |

### 3.2 状态流转 (TaskStatus)

```
                    ┌─────────────┐
                    │     NEW     │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐            ┌──────────┐
        │CLARIFYING │            │   READY   │
        └─────┬─────┘            └────┬─────┘
              │                      │
              │                      ├──────▶ tool_node ──▶ ready_node (循环)
              │                      │              ▲
              │                      ▼              │
              │                 ┌──────────┐         │
              │                 │  FILTER  │─────────┘
              │                 └────┬─────┘
              │                      │
              │           ┌─────────┴─────────┐
              │           ▼                   ▼
              │    ┌───────────┐        ┌──────────┐
              └───▶│  COMPLETED│        │   READY  │ (继续过滤)
                   └─────┬─────┘        └──────────┘
                         │
                         ▼
                    generate_node
                         │
                         ▼
                        END
```

### 3.3 边 (Edges)

| 边 | 路由条件 |
|----|---------|
| `route_by_status_edge` | `NEW` → `clarifying_node` / `ready_node` |
| `route_after_ready_edge` | `READY` → `tool_node` / `filter_node` |
| `router_after_filter_edge` | `filter_node` → `ready_node` (再过滤) / `generate_node` (完成) |

### 3.4 搜索循环

`tool_node` → `ready_node` → `filter_node` 形成**搜索-过滤循环**：

- 用户可要求"换一批"（`is_replace_products=true`），重新执行搜索
- 最大循环次数由 `max_search_loop` 限制

---

## 4. 闲聊服务设计 (ChitChatService)

采用 **ReAct Agent** 架构，内置工具：

| 工具 | 功能 |
|-----|------|
| `tavily_search` | 联网搜索，获取最新信息 |
| `get_current_weather` | 查询实时天气 |
| `get_forecast_weather` | 查询天气预报 |

### 4.1 特点

- 单例模式管理
- 不使用 checkpointer，对话历史通过输入上下文注入
- 支持对话历史上下文理解

---

## 5. 状态设计

### 5.1 主图状态 (ShopmindAgentState)

```python
{
    "messages": [...],              # 对话消息历史
    "original_query": str,          # 原始用户 query
    "rewritten_query": str,         # 改写后的 query
    "sub_tasks": [SubTask, ...],   # 所有子任务列表
    "current_tasks": [SubTask],    # 当前待处理的子任务
    "sub_task_results": [SubTask], # 已完成的子任务结果
    "answer": str                   # 最终回复
}
```

### 5.2 子图状态 (ShoppingSubgraphState)

```python
{
    "task": ShoppingSubTask,           # 购物子任务
    "subgraph_messages": [...],         # 子图内部消息
    "searched_res": [...],              # 搜索结果
    "searched_details": [...],          # 商品详情
    "filtered_product_ids": [int, ...],# 过滤后的商品 ID
    "product_after_filter": [...],      # 过滤后的商品
    "search_count_loop": int,            # 当前搜索循环次数
    "messages": [...]                   # 父图消息（共享）
}
```

### 5.3 意图分类 (IntentCategory)

| 分类 | 说明 |
|-----|------|
| `SHOPPING` | 购物意图（商品搜索、推荐） |
| `PLATFORM` | 平台规则意图（政策、规则查询） |
| `CHITCHAT` | 闲聊意图（天气、闲聊、泛泛的品牌讨论等） |
| `COMPARISON` | 商品比较意图（代词指代已推荐商品的比较） |

### 5.4 任务状态 (TaskStatus)

| 状态 | 说明 |
|-----|------|
| `NEW` | 刚创建 |
| `CLARIFYING` | 信息不足，等待用户澄清 |
| `READY` | 槽位齐全，可执行 |
| `COMPLETED` | 执行完成 |
| `FAILED` | 执行失败 |

---

## 6. 商品比较子图设计 (ComparisonSubgraph)

当用户使用代词（"这几个"、"那些"）指代已推荐的商品时，触发比较子图。

### 6.1 节点 (Nodes)

| 节点名称 | 功能说明 |
|---------|---------|
| `detail_node` | 从 `has_recommended_product_ids` 获取商品ID，并行调用 `get_product_detail` 获取详情 |
| `compare_node` | 将商品详情传给 LLM 生成对比文案和购买建议 |

### 6.2 流程

```
START
  │
  ▼
detail_node (获取商品详情)
  │
  ▼
compare_node (生成对比文案)
  │
  ▼
  END
```

### 6.3 触发条件

当用户 query 满足以下条件时，路由到 COMPARISON 意图：

1. 包含比较类词汇（"比较"、"区别"、"选哪个"）
2. 使用了代词（"这几个"、"那些"、"那个"）
3. 历史的 ShoppingSubTask 中 `has_recommended_product_ids` 非空

### 6.4 路由规则

| 用户输入 | 代词 | has_recommended_product_ids | 路由 |
|---------|------|---------------------------|------|
| "华为和iPhone选哪个" | 无 | 可能非空 | CHITCHAT（品牌讨论） |
| "休闲裤还是工装裤" | 无 | 空 | CHITCHAT（泛泛讨论） |
| "这几个有什么区别" | 有 | 非空 | COMPARISON（商品级比较） |
| "那几个呢" | 有 | 空 | CHITCHAT（无锚点） |

---

## 7. 初始化流程 (GraphFactory)

```
GraphFactory.build_all()
    │
    ├─ 1. SearchingSubgraph.build_searching_subgraph()
    │       └─ SearchingSubgraph.get_instance()
    │
    ├─ 2. ChitChatService.build_chitchat_agent()
    │       └─ ChitChatService.get_instance()
    │
    ├─ 3. ComparisonSubgraph.build_comparison_subgraph()
    │       └─ ComparisonSubgraph.get_instance()
    │
    └─ 4. ShopmindAgentGraph.init_graph()
            └─ ShopmindAgentGraph.get_instance()
```

所有 Graph 和 Agent 均采用**单例模式**，`GraphFactory` 统一管理初始化顺序。

---

## 8. 技术栈

| 组件 | 技术 |
|-----|------|
| 图编排框架 | LangGraph 0.2+ |
| 状态管理 | TypedDict + Annotated |
| 检查点持久化 | Redis / PostgreSQL |
| LLM | OpenAI GPT-4o / DashScope (Qwen) |
| Agent 类型 | ReAct Agent (Chitchat) |

---

## 9. 设计亮点

1. **MapReduce 模式**：主图通过 `Send` 实现多任务并行处理
2. **状态机驱动**：购物子图通过 `TaskStatus` 控制搜索-澄清-过滤流程
3. **父子图状态共享**：子图可访问父图消息，实现上下文一致
4. **单例+工厂模式**：统一管理图的生命周期，避免重复创建开销
5. **可扩展性**：新增意图类型只需添加对应 Node 和 Edge，遵循开闭原则
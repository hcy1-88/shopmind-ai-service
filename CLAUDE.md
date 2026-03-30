# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShopMind AI Service is an AI microservice for an intelligent e-commerce platform. It handles AI interactions including intelligent copywriting, content moderation, and shopping guidance agents. The service communicates with large language models (LLMs) to provide AI capabilities.

## Technology Stack

- **Web Framework**: FastAPI 0.115+
- **AI Framework**: LangChain 0.3+, LangGraph 0.2+, LlamaIndex 0.12+
- **Vector Database**: Milvus 2.4+
- **LLM**: OpenAI GPT-4o/GPT-4o-mini, DashScope (Alibaba Qwen)
- **Session Storage**: Redis, PostgreSQL
- **Service Registry/Config**: Nacos
- **Package Manager**: uv (Python 3.12+)

## Key Features

### 1. AI Agents (LangGraph-based)

Multi-agent system with state machine orchestration:

- **Main Graph**: `ShopmindAgentGraph` with nodes for query rewrite, intent decomposition, shopping subgraph, platform rules, chitchat, and result aggregation
- **Shopping Subgraph**: Handles product search/recommendation with states: NEW → CLARIFYING → READY → COMPLETED
- **Intent Categories**: SHOPPING, PLATFORM (rules/policy), CHITCHAT

### 2. Product AI Services

- Title Check / Image Check - Content moderation
- Description Generate / Summary Generate / Tag Generate - Content creation
- Product Vectorization - Product embedding for search

### 3. RAG (Retrieval Augmented Generation)

- Milvus vector search for semantic retrieval
- PostgreSQL docstore for document indexing
- Platform knowledge base for rules/policy queries

### 4. Tools for Agents

- `search_product` - Natural language product search
- `get_product_detail` - Get product by ID
- `get_new_product` - Latest products
- `platform_knowledge_search` - RAG-based knowledge lookup
- `tavily_search` - Web search
- `get_current_weather` / `get_forecast_weather` - Weather queries

## Common Commands

```bash
# Install dependencies
uv sync

# Development (hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
python -m app.main

# Run tests
pytest

# Linting
ruff check .
ruff format .
```

## Architecture

### Service Initialization (lifespan in main.py)
1. Nacos connection and service registration
2. LLM service initialization (from Nacos config)
3. Embedding service initialization
4. Milvus/Redis connections
5. RAG service initialization

### Configuration
- Local: `.env` file for app settings
- Centralized: Nacos for Milvus, LLM, Redis, chat config
- Pydantic settings for type-safe config

### State Management
- LangGraph state for agent conversations
- Redis/PostgreSQL checkpoints for session persistence
- `session_id` (thread_id) for multi-session isolation

### Streaming Response
- SSE (Server-Sent Events) via `/ai/chat` endpoint
- Event types: `thinking_start/end`, `tool_calls`, `tool_start/end`, `token_stream`

## Directory Structure

```
src/app/
├── agents/v1/          # LangGraph agents (nodes, edges, subgraphs)
├── chains/             # LangChain chains
├── checkpoint/         # Redis/PostgreSQL checkpoint persistence
├── clients/            # External service clients (Redis, Product Service, Nacos)
├── config/             # Settings and Nacos client
├── providers/          # LLM providers (OpenAI, DashScope)
├── routers/            # FastAPI routes (ai_ask, ai_product, rag)
├── schemas/            # Pydantic request/response schemas
├── services/           # Business services (ai_chat, llm, rag, product_ai)
├── tools/              # Agent tools definitions
├── vector_store/       # Milvus client and collections
└── main.py             # Entry point with lifespan management
```

## Agent
if user asks questions about agent, you can read files bellow to understand the design of graph-agent :
1. design-markdown - E:\work_directory\ShopMind\shopmind-python\shopmind-ai-service\docs\Agent设计方案.md
2. agent code - E:\work_directory\ShopMind\shopmind-python\shopmind-ai-service\src\app\agents\v1
3. entry to chat - E:\work_directory\ShopMind\shopmind-python\shopmind-ai-service\src\app\services\ai_chat_service.py
4. project's README.md
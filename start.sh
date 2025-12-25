#!/bin/bash

# AI Service 快速启动脚本

echo "================================"
echo "  AI Service 启动脚本"
echo "================================"

# 检查是否安装了 uv
if ! command -v uv &> /dev/null
then
    echo "❌ uv 未安装，请先安装 uv:"
    echo "   pip install uv"
    exit 1
fi

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  .env 文件不存在，从 .env.example 复制..."
    cp .env.example .env
    echo "✅ 已创建 .env 文件，请编辑并配置必要的参数"
    echo "   特别是 OPENAI_API_KEY 和 NACOS_SERVER_ADDR"
    exit 0
fi

echo "📦 同步依赖..."
uv sync

echo "🚀 启动服务 (开发模式)..."
echo "   API 文档: http://localhost:8000/docs"
echo "   健康检查: http://localhost:8000/health"
echo ""

uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

@echo off
REM AI Service 快速启动脚本 (Windows)

echo ================================
echo   AI Service 启动脚本
echo ================================
echo.

REM 检查是否安装了 uv
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ uv 未安装，请先安装 uv:
    echo    pip install uv
    exit /b 1
)

REM 检查 .env 文件
if not exist .env (
    echo ⚠️  .env 文件不存在，从 .env.example 复制...
    copy .env.example .env
    echo ✅ 已创建 .env 文件，请编辑并配置必要的参数
    echo    特别是 OPENAI_API_KEY 和 NACOS_SERVER_ADDR
    exit /b 0
)

echo 📦 同步依赖...
uv sync

echo 🚀 启动服务 (开发模式)...
echo    API 文档: http://localhost:8000/docs
echo    健康检查: http://localhost:8000/health
echo.

uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

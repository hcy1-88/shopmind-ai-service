"""项目启动脚本 - 在项目根目录运行: python run.py 或 uv run python run.py"""

import os
import sys
from pathlib import Path

# 设置 UTF-8 编码，解决 Windows 中文乱码问题
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # 设置控制台输出编码为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.service_ip,
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

# 启动命令：在项目根目录，输入 uv run python run.py
# swagger API 文档：http://ip:8085/docs
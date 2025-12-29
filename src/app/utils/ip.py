"""
@File       : ip.py
@Description:

@Time       : 2025/12/24 17:17
@Author     : hcy18
"""
import socket
from src.app.utils.logger import app_logger as logger

def get_local_ip():
    """获取本机内网 IP（非 127.0.0.1）"""
    try:
        # 连接一个外部地址（不会真正发包）
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        logger.error("获取 ip 异常！")
        return "127.0.0.1"
"""
@File       : image_util.py
@Description:

@Time       : 2025/12/26 9:00
@Author     : hcy18
"""
import base64
import mimetypes
import re
from urllib.parse import urlparse

import httpx


def is_base64_image(image_data: str) -> bool:
    """
    检查图片是否已经是 base64 编码

    Returns:
        True if base64, False otherwise
    """

    # 正则：匹配 data:image/xxx;base64,<base64>
    pattern = r'^data:image/([a-zA-Z0-9+.-]+);base64,([A-Za-z0-9+/=]+)$'
    if re.fullmatch(pattern, image_data):
        return True
    else:
        return False


async def load_image_from_url(url: str) -> str:
    """
    将图片 url 转换为 base64.

    Args:
        url: Image URL

    Returns:
        Base64 encoded image, 含 data 前缀！
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        # 1. 优先用响应头
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type.startswith("image/"):
            mime = content_type
        else:
            # 2. 用 URL 扩展名猜测
            parsed = urlparse(url)
            ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else ""
            mime, _ = mimetypes.guess_type(f"dummy.{ext}")
            mime = mime or "image/jpeg"  # 3. 默认

        base64_data = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime};base64,{base64_data}"
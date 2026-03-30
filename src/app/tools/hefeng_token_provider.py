"""
@File       : hefeng_token_provider.py
@Description: 和风天气 JWT Token 提供者 - 负责 JWT 生成、缓存和自动刷新

@Time       : 2026/3/29
@Author     : hcy18
"""
import asyncio
import base64
import time
from typing import Optional

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.config.nacos_client import get_nacos_client
from app.utils.logger import app_logger as logger


class HefengWeatherTokenProvider:
    """
    和风天气 JWT Token 提供者（进程级单例）

    负责：
    1. JWT Token 生成（Ed25519 算法）
    2. Token 缓存与自动刷新（提前 60 秒刷新）
    3. 401 自动重试机制
    """

    _instance: Optional["HefengWeatherTokenProvider"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0  # Unix 时间戳

    @classmethod
    def get_instance(cls) -> "HefengWeatherTokenProvider":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ========== 配置读取 ==========

    def _get_hefeng_config(self) -> dict:
        """从 Nacos 读取和风天气配置"""
        chat_config = get_nacos_client().get_chat_config()
        hefeng_cfg = chat_config.get("hefeng_weather", {})
        return {
            "api_host": hefeng_cfg.get("api_host", ""),
            "kid": hefeng_cfg.get("kid", ""),
            "project_id": hefeng_cfg.get("project_id", ""),
            "private_key": hefeng_cfg.get("private_key", ""),
        }

    # ========== Token 生成 ==========

    def generate_token(self, expires_in: int = 900) -> str:
        """
        生成和风天气 JWT 令牌

        Args:
            expires_in: 有效期（秒），最大 86400，默认 15 分钟

        Returns:
            JWT 令牌字符串
        """
        config = self._get_hefeng_config()
        kid = config["kid"]
        project_id = config["project_id"]
        private_key_b64 = config["private_key"]

        now = int(time.time())

        payload = {
            "sub": project_id,
            "iat": now - 30,  # 提前 30 秒，防止时间误差
            "exp": now + expires_in
        }

        headers = {
            "alg": "EdDSA",
            "kid": kid
        }

        # 从 Base64 解码私钥
        private_key_bytes = base64.b64decode(private_key_b64)
        # Ed25519 私钥：跳过 PKCS8 头 16 字节，取后 32 字节
        ed25519_private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes[16:])

        token = jwt.encode(payload, ed25519_private_key, algorithm="EdDSA", headers=headers)
        return token

    # ========== Token 缓存管理 ==========

    def _is_token_expired(self) -> bool:
        """检查 Token 是否已过期或即将过期（不足 60 秒）"""
        if self._cached_token is None:
            return True
        return time.time() >= (self._token_expires_at - 60)

    async def get_token(self) -> str:
        """
        获取有效 Token，必要时自动刷新（线程安全）

        Returns:
            JWT Token 字符串
        """
        if not self._is_token_expired():
            return self._cached_token

        async with self._lock:
            # 双重检查
            if not self._is_token_expired():
                return self._cached_token

            self._cached_token = self.generate_token()
            self._token_expires_at = time.time() + 900  # 15 分钟后过期
            logger.info(f"[HefengToken] 生成新 JWT Token，过期时间: {self._token_expires_at}")
            return self._cached_token

    def _invalidate_token(self) -> None:
        """强制失效 Token，下次调用时会重新生成"""
        self._cached_token = None
        self._token_expires_at = 0
        logger.info("[HefengToken] Token 已失效")

    # ========== 带 401 重试的请求 ==========

    async def request_with_auth(self, url: str, **kwargs) -> httpx.Response:
        """
        发起带 JWT 认证的请求，首次 401 自动重试

        Args:
            url: 请求 URL
            **kwargs: 传递给 httpx 请求的其他参数

        Returns:
            httpx.Response 对象

        Raises:
            httpx.HTTPStatusError: 重试后仍失败时抛出
        """
        token = await self.get_token()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept-Encoding"] = "gzip"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, **kwargs)

            if response.status_code == 401:
                # 首次 401，强制刷新 Token 并重试
                logger.warning("[HefengToken] 收到 401，开始重试...")
                self._invalidate_token()
                token = await self.get_token()
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers, **kwargs)

            return response


def get_hefeng_token_provider() -> HefengWeatherTokenProvider:
    """获取和风天气 Token 提供者单例"""
    return HefengWeatherTokenProvider.get_instance()

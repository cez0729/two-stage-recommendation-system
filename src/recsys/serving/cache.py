"""Optional Redis cache with automatic failure bypass."""

from __future__ import annotations

import json
from typing import Any


class RedisCache:
    """Best-effort cache; all Redis errors degrade to a cache miss."""

    def __init__(self, url: str, ttl_seconds: int = 300) -> None:
        import redis

        self.client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.05,
            socket_timeout=0.05,
            retry_on_timeout=False,
        )
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any | None:
        try:
            value = self.client.get(key)
            return None if value is None else json.loads(value)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> bool:
        try:
            self.client.setex(key, self.ttl_seconds, json.dumps(value, ensure_ascii=False))
            return True
        except Exception:
            return False

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

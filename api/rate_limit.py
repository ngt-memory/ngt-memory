"""
Простой in-process rate limiter (token bucket) для NGT Memory API.

Ограничивает частоту запросов на ключ (API-key, либо client IP при открытом
доступе). Реализация in-memory — для одного воркера точная, для multi-worker
лимит делится между процессами (каждый воркер считает свою долю); для строгого
глобального лимита используйте Redis-бэкенд (вне скоупа этого модуля) или
лимит на уровне reverse-proxy (nginx limit_req).

Конфигурация:
    NGT_RATE_LIMIT_RPS    — устойчивая частота, запросов/сек (0 = выключено)
    NGT_RATE_LIMIT_BURST  — размер всплеска (ёмкость ведра)
"""

import time
import threading
from typing import Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class _TokenBucket:
    __slots__ = ("tokens", "last")

    def __init__(self, capacity: float, now: float):
        self.tokens = capacity
        self.last = now


class RateLimiter:
    """Token bucket per key. Потокобезопасен."""

    def __init__(self, rps: float, burst: int):
        self.rps = float(rps)
        self.burst = float(burst)
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> Tuple[bool, float]:
        """Возвращает (разрешено, retry_after_seconds)."""
        if self.rps <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _TokenBucket(self.burst, now)
                self._buckets[key] = b
            # Пополняем ведро
            elapsed = now - b.last
            b.last = now
            b.tokens = min(self.burst, b.tokens + elapsed * self.rps)
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            # Сколько ждать до следующего токена
            retry_after = (1.0 - b.tokens) / self.rps
            return False, retry_after

    def cleanup(self, max_idle: float = 300.0) -> int:
        """Удаляет давно неиспользуемые ведра (вызывать периодически)."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, b in self._buckets.items() if now - b.last > max_idle]
            for k in stale:
                del self._buckets[k]
            return len(stale)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Применяет RateLimiter ко всем запросам, кроме исключённых путей."""

    EXEMPT_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/metrics"})

    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self._limiter = limiter

    def _key(self, request: Request) -> str:
        # Ключ: API-key если есть, иначе client IP (учитываем X-Forwarded-For
        # за reverse-proxy — берём первый адрес из цепочки).
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key[:32]}"
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return f"ip:{fwd.split(',')[0].strip()}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        allowed, retry_after = self._limiter.allow(self._key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)

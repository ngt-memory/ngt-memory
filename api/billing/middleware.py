"""
Billing Middleware — проверка API-ключа и лимитов на каждый запрос.

Логика:
  1. Запрос без ключа → 401
  2. Невалидный ключ → 401
  3. Лимит исчерпан → 429 Too Many Requests
  4. Ключ валиден + лимит не исчерпан → пропускаем, инкрементируем usage

Пути-исключения (не требуют ключа):
  - /health
  - /docs, /openapi.json
  - /billing/*
"""

import logging
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.billing.key_manager import KeyManager
from api.billing.models import PLAN_LIMITS, PLAN_NAMES

logger = logging.getLogger(__name__)

# Пути, не требующие API-ключа
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}

PUBLIC_PREFIXES = (
    "/billing/",
)


class BillingMiddleware(BaseHTTPMiddleware):
    """
    Middleware: проверяет X-Api-Key, лимиты, трекает usage.
    """

    def __init__(self, app, key_manager: KeyManager, enabled: bool = True):
        super().__init__(app)
        self.km = key_manager
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        # Если billing отключён — пропускаем
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        # Публичные пути — без ключа
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)

        # OPTIONS (CORS preflight) — без ключа
        if request.method == "OPTIONS":
            return await call_next(request)

        # Извлекаем ключ
        api_key = (
            request.headers.get("X-Api-Key")
            or request.query_params.get("api_key")
        )

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "api_key_required",
                    "message": "API key required. Pass it via X-Api-Key header or ?api_key= query param.",
                    "docs": "https://github.com/ngt-memory/ngt-memory#authentication",
                },
            )

        # Валидируем ключ
        record = self.km.validate_key(api_key)
        if not record:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_api_key",
                    "message": "Invalid or deactivated API key.",
                },
            )

        # Проверяем лимит
        allowed, used, limit = self.km.check_limit(api_key)
        if not allowed:
            plan_name = PLAN_NAMES.get(record.plan, record.plan.value)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Daily limit reached ({limit} requests/day on {plan_name}). "
                               f"Upgrade your plan for more requests.",
                    "used_today": used,
                    "daily_limit": limit,
                    "plan": record.plan.value,
                    "upgrade_url": "https://ngt-memory.dev/pricing",
                },
                headers={"Retry-After": "86400"},
            )

        # Сохраняем ключ в request state для downstream
        request.state.api_key = api_key
        request.state.plan = record.plan
        request.state.user_email = record.user_email

        # Выполняем запрос
        response = await call_next(request)

        # Инкрементируем usage (после успешного ответа)
        if response.status_code < 400:
            self.km.increment_usage(api_key)

        # Добавляем заголовки с лимитами
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Used"] = str(used + 1)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - used - 1))

        return response

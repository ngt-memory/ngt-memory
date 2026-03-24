"""
Structured logging и request_id middleware для NGT Memory API.

Использование:
    from api.logging_config import setup_logging, RequestIdMiddleware

    setup_logging()
    app.add_middleware(RequestIdMiddleware)
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ── Context variable для request_id ──────────────────────────────────────────

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    return request_id_var.get()


# ── JSON Formatter ───────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Форматирует лог-записи как JSON с request_id."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Дополнительные поля из extra
        for key in ("session_id", "latency_ms", "status_code", "method", "path"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, ensure_ascii=False)


# ── Setup ────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", json_format: bool = True):
    """
    Настраивает логирование для всего приложения.

    Args:
        level: уровень логирования (INFO, DEBUG, WARNING, ERROR)
        json_format: если True — JSON формат, иначе — human-readable
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Убираем дефолтные хендлеры
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    root.addHandler(handler)


# ── Request ID Middleware ────────────────────────────────────────────────────

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Добавляет уникальный request_id к каждому запросу.
    - Если клиент передал X-Request-ID — используем его.
    - Иначе — генерируем UUID4.
    - request_id доступен через get_request_id() в любом месте кода.
    - Добавляется в response header X-Request-ID.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)

"""
Prometheus-метрики для NGT Memory API.

Если prometheus_client не установлен — модуль работает как no-op:
декоратор/контекст ничего не пишут, эндпоинт /metrics возвращает 501.
Это позволяет держать метрики опциональной зависимостью.

Экспортируемые метрики:
    ngt_requests_total{endpoint,status}         — счётчик запросов
    ngt_request_duration_seconds{endpoint}      — гистограмма латентности
    ngt_active_sessions                         — gauge активных сессий
    ngt_memory_entries_total                    — gauge суммарных записей
    ngt_llm_tokens_total{direction}             — счётчик токенов (in/out)
    ngt_memories_used                           — гистограмма воспоминаний/ход
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("ngt_api.metrics")

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"


if PROMETHEUS_AVAILABLE:
    REQUESTS = Counter(
        "ngt_requests_total", "Total API requests", ["endpoint", "status"],
    )
    DURATION = Histogram(
        "ngt_request_duration_seconds", "Request duration", ["endpoint"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    ACTIVE_SESSIONS = Gauge("ngt_active_sessions", "Active sessions")
    MEMORY_ENTRIES = Gauge("ngt_memory_entries_total", "Total memory entries across sessions")
    TOKENS = Counter("ngt_llm_tokens_total", "LLM tokens", ["direction"])
    MEMORIES_USED = Histogram(
        "ngt_memories_used", "Memories injected per turn",
        buckets=(0, 1, 2, 3, 5, 10, 20),
    )
else:
    REQUESTS = DURATION = ACTIVE_SESSIONS = MEMORY_ENTRIES = TOKENS = MEMORIES_USED = None


@contextmanager
def track_request(endpoint: str):
    """Контекст-менеджер: меряет длительность и считает запрос.

    Использование:
        with track_request("/chat") as mark:
            ... обработка ...
            mark(200)   # зафиксировать статус
    """
    status_holder = {"code": 200}

    def mark(code: int):
        status_holder["code"] = code

    t0 = time.perf_counter()
    try:
        yield mark
    except Exception:
        status_holder["code"] = 500
        raise
    finally:
        if PROMETHEUS_AVAILABLE:
            DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
            REQUESTS.labels(endpoint=endpoint, status=str(status_holder["code"])).inc()


def record_chat(tokens_in: int, tokens_out: int, memories_count: int) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    TOKENS.labels(direction="in").inc(tokens_in)
    TOKENS.labels(direction="out").inc(tokens_out)
    MEMORIES_USED.observe(memories_count)


def update_session_gauges(active: int, total_entries: int) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    ACTIVE_SESSIONS.set(active)
    MEMORY_ENTRIES.set(total_entries)


def render() -> bytes:
    if not PROMETHEUS_AVAILABLE:
        return b""
    return generate_latest()

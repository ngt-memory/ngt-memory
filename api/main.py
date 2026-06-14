"""
NGT Memory API — FastAPI REST сервер.

Endpoints:
    POST /chat          — диалог с NGT Memory
    POST /store         — сохранить факт в память
    POST /retrieve      — поиск по памяти
    POST /session/reset — сбросить память сессии
    GET  /session/{id}/stats — статистика сессии
    GET  /health        — статус сервера
    GET  /metrics       — Prometheus-метрики (если включены)

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 9190 --reload

Или через Docker:
    docker-compose up

Backend сессий выбирается через NGT_SESSION_BACKEND:
    memory — in-process (1 worker, см. README)
    redis  — multi-worker / multi-instance (требует redis + safetensors)
"""

import asyncio
import secrets
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from api.config import settings
from api.models import (
    ChatRequest, ChatResponse, MemoryItem,
    StoreRequest, StoreResponse,
    RetrieveRequest, RetrieveResponse,
    NewSessionRequest, ResetResponse,
    SessionStatsResponse, HealthResponse,
)
from api.session_store_base import SessionStoreBase
from api.logging_config import setup_logging, RequestIdMiddleware, get_request_id
from api.rate_limit import RateLimiter, RateLimitMiddleware
from api.ownership import SessionOwnership, OwnershipError, derive_owner_id
from api import metrics

# ── Billing (опциональный модуль — не нужен для self-hosted) ──────────────────
try:
    from api.billing.key_manager import KeyManager
    from api.billing.middleware import BillingMiddleware
    from api.billing.stripe_handler import router as billing_router
    from api.billing.yukassa_handler import router as yukassa_router
    from api.billing.free_register import router as free_register_router
    _BILLING_AVAILABLE = True
except ImportError:
    _BILLING_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────

setup_logging(level=settings.log_level, json_format=settings.log_json)
logger = logging.getLogger("ngt_api")

# ── Global state ──────────────────────────────────────────────────────────────

store: Optional[SessionStoreBase] = None
_rate_limiter: Optional[RateLimiter] = None
_ownership: Optional[SessionOwnership] = None


def _build_store() -> SessionStoreBase:
    """Создаёт session store согласно settings.session_backend."""
    if settings.session_backend == "redis":
        from api.session_store_redis import RedisSessionStore
        logger.info("Session backend: REDIS (%s) — multi-worker ready", settings.redis_url)
        return RedisSessionStore(
            openai_api_key=settings.openai_api_key.get_secret_value(),
            redis_url=settings.redis_url,
            base_url=settings.openai_base_url or None,
            model=settings.chat_model,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            memory_top_k=settings.memory_top_k,
            memory_threshold=settings.memory_threshold,
            use_graph=settings.use_graph,
            session_ttl_seconds=settings.session_ttl,
        )

    from api.session_store import SessionStore
    logger.info("Session backend: MEMORY (in-process, single worker)")
    return SessionStore(
        openai_api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url or None,
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        memory_top_k=settings.memory_top_k,
        memory_threshold=settings.memory_threshold,
        use_graph=settings.use_graph,
        session_ttl_seconds=settings.session_ttl,
        max_sessions=settings.max_sessions,
        persist_dir=settings.persist_dir,
        max_total_entries=settings.max_total_entries,
        max_total_bytes=settings.max_total_bytes,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, _rate_limiter
    logger.info(f"NGT Memory API v{settings.version} запускается...")
    logger.info(f"  Base URL:        {settings.openai_base_url or 'OpenAI (default)'}")
    logger.info(f"  Chat model:      {settings.chat_model}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  Embedding dim:   {settings.embedding_dim}")
    logger.info(f"  Memory top_k:    {settings.memory_top_k}")
    logger.info(f"  Use graph:       {settings.use_graph}")
    logger.info(f"  Session TTL:     {settings.session_ttl}s")
    logger.info(f"  Max sessions:    {settings.max_sessions}")
    logger.info(f"  Memory budget:   {settings.max_total_entries} entries / {settings.max_total_mb} MB")
    logger.info(f"  Persist dir:     {settings.persist_dir or '(disabled — in-memory only)'}")
    logger.info(f"  Rate limit:      {settings.rate_limit_rps or 'off'} rps / burst {settings.rate_limit_burst}")
    logger.info(f"  Metrics:         {'on' if (settings.metrics_enabled and metrics.PROMETHEUS_AVAILABLE) else 'off'}")
    logger.info(f"  Ownership:       {settings.session_ownership}")

    store = _build_store()

    # Реестр владения сессиями. Активен по политике session_ownership.
    global _ownership
    ownership_active = settings.session_ownership == "strict" or (
        settings.session_ownership == "auto" and settings.billing_enabled
    )
    if ownership_active:
        # При redis-backend переиспользуем тот же клиент, что и у store —
        # владение тогда корректно шарится между воркерами.
        redis_client = getattr(store, "_redis", None) if settings.session_backend == "redis" else None
        _ownership = SessionOwnership(redis=redis_client, persist_dir=settings.persist_dir)
        logger.info(
            "Session ownership: %s (storage=%s)",
            settings.session_ownership,
            "redis" if redis_client is not None else "memory+disk",
        )
    else:
        logger.info("Session ownership: off (single-tenant / no per-user keys)")

    # Фоновое периодическое сохранение сессий на диск (только in-memory backend)
    async def _periodic_persist():
        while True:
            await asyncio.sleep(settings.persist_interval)
            try:
                n = await asyncio.to_thread(store.save_all)
                if n:
                    logger.info(f"persisted {n} sessions to disk")
            except Exception:
                logger.exception("periodic session persist failed")

    # Периодическая очистка rate-limiter ведёр
    async def _periodic_rl_cleanup():
        while True:
            await asyncio.sleep(300)
            if _rate_limiter is not None:
                try:
                    _rate_limiter.cleanup()
                except Exception:
                    logger.exception("rate limiter cleanup failed")

    persist_task = (
        asyncio.create_task(_periodic_persist())
        if (settings.persist_dir and settings.session_backend == "memory")
        else None
    )
    rl_task = (
        asyncio.create_task(_periodic_rl_cleanup())
        if settings.rate_limit_rps > 0 else None
    )

    logger.info("SessionStore инициализирован. API готов.")
    yield

    for task in (persist_task, rl_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if settings.persist_dir and settings.session_backend == "memory":
        n = await asyncio.to_thread(store.save_all)
        logger.info(f"shutdown: persisted {n} sessions to disk")

    await store.aclose()
    logger.info("NGT Memory API остановлен.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NGT Memory API",
    description=(
        "Persistent memory layer for LLM applications. "
        "Store, retrieve, and chat with context-aware AI powered by NGT Memory."
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.root_path,
)

# CORS: спека запрещает wildcard origin вместе с credentials. Starlette в этом
# режиме эхо-отражает любой Origin с Allow-Credentials: true → credentialed-запросы
# (cookies, auth headers) были бы разрешены с любого сайта. При "*" в origins
# отключаем credentials и предупреждаем в логе.
_cors_origins = settings.cors_origins_list
_cors_allow_credentials = "*" not in _cors_origins
if not _cors_allow_credentials:
    logger.warning(
        "NGT_CORS_ORIGINS contains wildcard '*' — allow_credentials disabled. "
        "Set explicit origins (e.g. NGT_CORS_ORIGINS=https://ngt-memory.ru) "
        "to enable credentialed cross-origin requests."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (если включён) — до RequestId, чтобы 429 тоже получал request_id
if settings.rate_limit_rps > 0:
    _rate_limiter = RateLimiter(rps=settings.rate_limit_rps, burst=settings.rate_limit_burst)
    app.add_middleware(RateLimitMiddleware, limiter=_rate_limiter)

app.add_middleware(RequestIdMiddleware)

# ── Billing (опциональный — подключается автоматически если api/billing/ есть)

if _BILLING_AVAILABLE and settings.billing_enabled:
    _km = KeyManager()
    app.add_middleware(BillingMiddleware, key_manager=_km, enabled=True)
    app.include_router(billing_router)
    app.include_router(yukassa_router)
    app.include_router(free_register_router)
    logger.info("Billing module loaded — API keys + payments active")
elif _BILLING_AVAILABLE and not settings.billing_enabled:
    logger.info("Billing module found but disabled (NGT_BILLING_ENABLED=false)")
else:
    logger.info("Billing module not installed — open access (self-hosted mode)")

# ── Auth (опциональная) ───────────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Проверяет NGT_API_SECRET если он установлен (timing-safe).

    secrets.compare_digest сравнивает за константное время — обычный `!=`
    прерывается на первом несовпавшем байте и допускает timing-атаку
    (посимвольный подбор ключа по микрозадержкам).
    """
    if not settings.api_secret:
        return
    provided = (x_api_key or "").encode("utf-8")
    expected = settings.api_secret.encode("utf-8")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Session ownership (правка 6) ──────────────────────────────────────────────

def _caller_owner_id(x_api_key: Optional[str]) -> Optional[str]:
    """Идентичность вызывающего для проверки владения сессией.

    Возвращает owner_id (хэш ключа), либо None если владение выключено.
    В режиме strict требует наличие X-API-Key (иначе сессии неотличимы
    по владельцу — это ошибка конфигурации/клиента, отвечаем 401).
    """
    if _ownership is None:
        return None  # владение выключено — единое пространство
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="X-API-Key required for session access (ownership enforced)",
        )
    return derive_owner_id(x_api_key)


async def _check_session_owner(session_id: str, owner_id: Optional[str], *, claim: bool = True) -> None:
    """Закрепляет/сверяет владение. 403 при доступе к чужой сессии.

    claim=True  — claim-on-first-use: бесхозную сессию закрепляет за
                  вызывающим (для эндпоинтов, создающих/мутирующих сессию).
    claim=False — только сверка: бесхозную сессию НЕ закрепляет (для
                  read-only /stats и precheck в /reset — чтобы простой
                  опрос несуществующего id не плодил «пустых» владельцев).
    """
    if _ownership is None or owner_id is None:
        return
    if claim:
        try:
            await _ownership.ensure(session_id, owner_id)
        except OwnershipError:
            logger.warning("ownership denied: session=%s caller=%s…", session_id, (owner_id or "")[:8])
            raise HTTPException(status_code=403, detail="Access to this session is forbidden")
        return
    # verify-only
    current = await _ownership.owner_of(session_id)
    if current is not None and current != owner_id:
        logger.warning("ownership denied: session=%s caller=%s…", session_id, (owner_id or "")[:8])
        raise HTTPException(status_code=403, detail="Access to this session is forbidden")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _memory_items(memories: list) -> list[MemoryItem]:
    return [
        MemoryItem(
            text=m.get("text", ""),
            score=round(m.get("score", 0.0), 4),
            domain=m.get("domain"),
            concepts=m.get("concepts"),
            metadata=m.get("metadata"),
        )
        for m in memories
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Проверка состояния сервера."""
    return HealthResponse(
        status="ok",
        version=settings.version,
        active_sessions=store.active_sessions() if store else 0,
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
    )


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus-метрики. 501 если prometheus_client не установлен или выключено."""
    if not (settings.metrics_enabled and metrics.PROMETHEUS_AVAILABLE):
        raise HTTPException(status_code=501, detail="Metrics disabled or prometheus_client not installed")
    # Обновляем gauges перед отдачей
    try:
        metrics.update_session_gauges(store.active_sessions(), store.total_entries())
    except Exception:
        logger.debug("failed to update session gauges", exc_info=True)
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE_LATEST)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Диалог с NGT Memory",
    description=(
        "Отправляет сообщение пользователя в LLM. Если use_memory=True, "
        "сначала извлекает релевантные воспоминания из NGT Memory и инжектирует их в контекст. "
        "Ответ и вопрос автоматически сохраняются в памяти сессии."
    ),
)
async def chat(
    request: ChatRequest,
    _: None = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    owner_id = _caller_owner_id(x_api_key)
    with metrics.track_request("/chat") as mark:
        # Лок сериализует конкурентные запросы одного session_id:
        # без него два параллельных /chat чередуются на await и перемешивают
        # _chat_history / _stats / память.
        async with store.get_lock(request.session_id):
            await _check_session_owner(request.session_id, owner_id)
            wrapper = await store.get_or_create_async(request.session_id)

            try:
                if request.use_memory:
                    result = await wrapper.achat(request.message)
                else:
                    result = await wrapper.achat_no_memory(request.message)
            except Exception as e:
                logger.error(f"chat error session={request.session_id}: {e}")
                mark(500)
                raise HTTPException(status_code=500, detail=str(e))

            # Состояние изменилось — фиксируем (no-op для in-memory, запись для Redis)
            await store.commit(request.session_id)

        memories = _memory_items(result.get("memories_used", []))
        mark(200)

    logger.info(
        f"chat session={request.session_id} "
        f"memories={len(memories)} "
        f"tokens_in={result.get('tokens_in', 0)} "
        f"latency={result.get('latency_ms', 0):.0f}ms"
    )
    metrics.record_chat(
        result.get("tokens_in", 0), result.get("tokens_out", 0), len(memories),
    )

    return ChatResponse(
        response=result["response"],
        session_id=request.session_id,
        memories_used=memories,
        memories_count=len(memories),
        tokens_in=result.get("tokens_in", 0),
        tokens_out=result.get("tokens_out", 0),
        latency_ms=round(result.get("latency_ms", 0), 1),
        memory_entries=wrapper.memory_entries_count,
        profile=wrapper.profile.as_dict() or None,
        profile_updates=result.get("profile_updates") or None,
    )


@app.post(
    "/store",
    response_model=StoreResponse,
    tags=["Memory"],
    summary="Сохранить факт в память",
    description=(
        "Напрямую сохраняет текст в NGT Memory сессии. "
        "Полезно для предварительной загрузки контекста (профиль пользователя, документы, факты)."
    ),
)
async def store_memory(
    request: StoreRequest,
    _: None = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    owner_id = _caller_owner_id(x_api_key)
    with metrics.track_request("/store") as mark:
        async with store.get_lock(request.session_id):
            await _check_session_owner(request.session_id, owner_id)
            wrapper = await store.get_or_create_async(request.session_id)

            try:
                emb = await wrapper.aembed_text(request.text)
                wrapper.memory.store(
                    embedding=emb,
                    text=request.text,
                    concepts=request.concepts,
                    metadata=request.metadata or {},
                    domain=request.domain or "general",
                )
            except Exception as e:
                logger.error(f"store error session={request.session_id}: {e}")
                mark(500)
                raise HTTPException(status_code=500, detail=str(e))

            await store.commit(request.session_id)

        entries = wrapper.memory_entries_count
        mark(200)

    logger.info(f"store session={request.session_id} total_entries={entries}")

    return StoreResponse(
        success=True,
        session_id=request.session_id,
        memory_entries=entries,
        message=f"Stored. Total memory entries: {entries}",
    )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Memory"],
    summary="Поиск по памяти",
    description=(
        "Ищет релевантные факты в NGT Memory по семантическому запросу. "
        "Поддерживает graph-boosted retrieval для кросс-доменных запросов."
    ),
)
async def retrieve_memory(
    request: RetrieveRequest,
    _: None = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    owner_id = _caller_owner_id(x_api_key)
    with metrics.track_request("/retrieve") as mark:
        # Лок: /retrieve читает память (_rebuild_entry_index / _emb_buffer),
        # параллельный /chat или /store на тот же session_id мутирует эти же
        # структуры — без лока возможна гонка (кривой индекс или исключение).
        async with store.get_lock(request.session_id):
            await _check_session_owner(request.session_id, owner_id)
            wrapper = await store.get_or_create_async(request.session_id)

            try:
                query_emb = await wrapper.aembed_text(request.query)
                results = wrapper.memory.retrieve(
                    query_embedding=query_emb,
                    top_k=request.top_k,
                    use_graph=request.use_graph,
                )
                filtered = [r for r in results if r.get("score", 0) >= request.threshold]
            except Exception as e:
                logger.error(f"retrieve error session={request.session_id}: {e}")
                mark(500)
                raise HTTPException(status_code=500, detail=str(e))

            # retrieve может пересоздать lazy embedding-индекс — фиксируем для Redis
            await store.commit(request.session_id)

        items = _memory_items(filtered)
        mark(200)

    # PRIVACY: текст запроса — пользовательские данные (медицина, финансы),
    # в логи не пишем; только длина и метаданные.
    logger.info(f"retrieve session={request.session_id} query_len={len(request.query)} found={len(items)}")

    return RetrieveResponse(
        results=items,
        count=len(items),
        session_id=request.session_id,
        query=request.query,
    )


@app.post(
    "/session/reset",
    response_model=ResetResponse,
    tags=["Session"],
    summary="Сбросить память сессии",
    description="Полностью очищает NGT Memory для указанной сессии.",
)
async def reset_session(
    request: NewSessionRequest,
    _: None = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    owner_id = _caller_owner_id(x_api_key)
    async with store.get_lock(request.session_id):
        # Сверяем владение ДО сброса — чужой не должен сбрасывать чужую сессию.
        # ensure здесь не закрепляет новую (для несуществующей сессии владельца
        # ещё нет → claim вернёт True, что корректно: сбрасывать нечего).
        await _check_session_owner(request.session_id, owner_id, claim=False)
        deleted = await store.reset_async(request.session_id)
        # Сброс = полная очистка: снимаем и владение, иначе id останется
        # «занятым» за прежним владельцем и его не сможет занять никто другой.
        if _ownership is not None:
            await _ownership.release(request.session_id)
    logger.info(f"reset session={request.session_id} existed={deleted}")

    return ResetResponse(
        success=True,
        session_id=request.session_id,
        message=f"Session '{request.session_id}' {'reset' if deleted else 'was already empty'}.",
    )


@app.get(
    "/session/{session_id}/stats",
    response_model=SessionStatsResponse,
    tags=["Session"],
    summary="Статистика сессии",
    description="Возвращает метрики памяти и производительности для сессии.",
)
async def session_stats(
    session_id: str,
    _: None = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    owner_id = _caller_owner_id(x_api_key)
    async with store.get_lock(session_id):
        await _check_session_owner(session_id, owner_id, claim=False)
        wrapper = await store.get_async(session_id)
        if wrapper is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        s = wrapper.get_stats()

    return SessionStatsResponse(
        session_id=session_id,
        memory_entries=s["memory_entries"],
        graph_edges=s["graph_edges"],
        graph_concepts=s["graph_concepts"],
        total_turns=s["total_turns"],
        total_memories_used=s["total_memories_used"],
        avg_memories_per_turn=s["avg_memories_per_turn"],
        total_tokens_in=s["total_tokens_in"],
        total_tokens_out=s["total_tokens_out"],
        avg_embed_ms=s["avg_embed_ms"],
        avg_retrieve_ms=s["avg_retrieve_ms"],
        avg_chat_ms=s["avg_chat_ms"],
    )


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    rid = get_request_id()
    logger.error(
        f"Unhandled error: {exc}",
        exc_info=True,
        extra={"request_id": rid, "method": request.method, "path": str(request.url.path)},
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": rid,
        },
    )


# ── Console entrypoint ────────────────────────────────────────────────────────

def run() -> None:
    """Запускает API через uvicorn: команда `ngt-api` после pip install."""
    import os
    import uvicorn
    # Multi-worker безопасен только с redis-backend
    workers = int(os.environ.get("NGT_WORKERS", "1"))
    if workers > 1 and settings.session_backend != "redis":
        logger.warning(
            "NGT_WORKERS=%d с backend=memory небезопасно (сессии не шарятся между "
            "процессами). Принудительно workers=1. Используйте NGT_SESSION_BACKEND=redis.",
            workers,
        )
        workers = 1
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("NGT_HOST", "0.0.0.0"),
        port=int(os.environ.get("NGT_PORT", "9190")),
        workers=workers,
    )


if __name__ == "__main__":
    run()

"""
NGT Memory API — FastAPI REST сервер.

Endpoints:
    POST /chat          — диалог с NGT Memory
    POST /store         — сохранить факт в память
    POST /retrieve      — поиск по памяти
    POST /session/reset — сбросить память сессии
    GET  /session/{id}/stats — статистика сессии
    GET  /health        — статус сервера

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 9190 --reload

Или через Docker:
    docker-compose up
"""

import asyncio
import secrets
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.models import (
    ChatRequest, ChatResponse, MemoryItem,
    StoreRequest, StoreResponse,
    RetrieveRequest, RetrieveResponse,
    NewSessionRequest, ResetResponse,
    SessionStatsResponse, HealthResponse,
)
from api.session_store import SessionStore
from api.logging_config import setup_logging, RequestIdMiddleware, get_request_id

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

store: Optional[SessionStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    logger.info(f"NGT Memory API v{settings.version} запускается...")
    logger.info(f"  Base URL:        {settings.openai_base_url or 'OpenAI (default)'}")
    logger.info(f"  Chat model:      {settings.chat_model}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  Embedding dim:   {settings.embedding_dim}")
    logger.info(f"  Memory top_k:    {settings.memory_top_k}")
    logger.info(f"  Use graph:       {settings.use_graph}")
    logger.info(f"  Session TTL:     {settings.session_ttl}s")
    logger.info(f"  Max sessions:    {settings.max_sessions}")
    logger.info(f"  Persist dir:     {settings.persist_dir or '(disabled — in-memory only)'}")

    store = SessionStore(
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
    )

    # Фоновое периодическое сохранение сессий на диск
    async def _periodic_persist():
        while True:
            await asyncio.sleep(settings.persist_interval)
            try:
                n = await asyncio.to_thread(store.save_all)
                if n:
                    logger.info(f"persisted {n} sessions to disk")
            except Exception:
                logger.exception("periodic session persist failed")

    persist_task = (
        asyncio.create_task(_periodic_persist()) if settings.persist_dir else None
    )

    logger.info("SessionStore инициализирован. API готов.")
    yield

    if persist_task is not None:
        persist_task.cancel()
        try:
            await persist_task
        except asyncio.CancelledError:
            pass
    if settings.persist_dir:
        n = await asyncio.to_thread(store.save_all)
        logger.info(f"shutdown: persisted {n} sessions to disk")

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
):
    t_start = time.perf_counter()
    # Лок сериализует конкурентные запросы одного session_id:
    # без него два параллельных /chat чередуются на await и перемешивают
    # _chat_history / _stats / память.
    async with store.get_lock(request.session_id):
        wrapper = store.get_or_create(request.session_id)

        try:
            if request.use_memory:
                result = await wrapper.achat(request.message)
            else:
                result = await wrapper.achat_no_memory(request.message)

        except Exception as e:
            logger.error(f"chat error session={request.session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    memories = _memory_items(result.get("memories_used", []))

    logger.info(
        f"chat session={request.session_id} "
        f"memories={len(memories)} "
        f"tokens_in={result.get('tokens_in', 0)} "
        f"latency={result.get('latency_ms', 0):.0f}ms"
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
):
    async with store.get_lock(request.session_id):
        wrapper = store.get_or_create(request.session_id)

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
            raise HTTPException(status_code=500, detail=str(e))

    entries = wrapper.memory_entries_count
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
):
    wrapper = store.get_or_create(request.session_id)

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
        raise HTTPException(status_code=500, detail=str(e))

    items = _memory_items(filtered)
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
):
    async with store.get_lock(request.session_id):
        deleted = store.reset(request.session_id)
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
):
    wrapper = store.get(session_id)
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
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("NGT_HOST", "0.0.0.0"),
        port=int(os.environ.get("NGT_PORT", "9190")),
        workers=1,  # in-memory сессии — см. README
    )


if __name__ == "__main__":
    run()

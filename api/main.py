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

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    ChatRequest, ChatResponse, MemoryItem,
    StoreRequest, StoreResponse,
    RetrieveRequest, RetrieveResponse,
    NewSessionRequest, ResetResponse,
    SessionStatsResponse, HealthResponse,
)
from api.session_store import SessionStore
from api.billing.key_manager import KeyManager
from api.billing.middleware import BillingMiddleware
from api.billing.stripe_handler import router as billing_router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ngt_api")

# ── Config from env ───────────────────────────────────────────────────────────

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
API_SECRET_KEY    = os.environ.get("NGT_API_SECRET", "")  # опционально: защита endpoint'ов
BILLING_ENABLED   = os.environ.get("NGT_BILLING_ENABLED", "false").lower() == "true"
CHAT_MODEL        = os.environ.get("NGT_CHAT_MODEL", "gpt-4.1-nano")
EMBEDDING_MODEL   = os.environ.get("NGT_EMBEDDING_MODEL", "text-embedding-3-small")
MEMORY_TOP_K      = int(os.environ.get("NGT_MEMORY_TOP_K", "5"))
MEMORY_THRESHOLD  = float(os.environ.get("NGT_MEMORY_THRESHOLD", "0.25"))
USE_GRAPH         = os.environ.get("NGT_USE_GRAPH", "true").lower() == "true"
SESSION_TTL       = int(os.environ.get("NGT_SESSION_TTL", "3600"))
MAX_SESSIONS      = int(os.environ.get("NGT_MAX_SESSIONS", "100"))

VERSION = "0.23.0"

# ── Global state ──────────────────────────────────────────────────────────────

store: Optional[SessionStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не установлен. Добавьте в .env файл.")

    logger.info(f"NGT Memory API v{VERSION} запускается...")
    logger.info(f"  Chat model:      {CHAT_MODEL}")
    logger.info(f"  Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"  Memory top_k:    {MEMORY_TOP_K}")
    logger.info(f"  Use graph:       {USE_GRAPH}")
    logger.info(f"  Session TTL:     {SESSION_TTL}s")
    logger.info(f"  Max sessions:    {MAX_SESSIONS}")

    store = SessionStore(
        openai_api_key=OPENAI_API_KEY,
        model=CHAT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        memory_top_k=MEMORY_TOP_K,
        memory_threshold=MEMORY_THRESHOLD,
        use_graph=USE_GRAPH,
        session_ttl_seconds=SESSION_TTL,
        max_sessions=MAX_SESSIONS,
    )

    logger.info("SessionStore инициализирован. API готов.")
    yield

    logger.info("NGT Memory API остановлен.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NGT Memory API",
    description=(
        "Persistent memory layer for LLM applications. "
        "Store, retrieve, and chat with context-aware AI powered by NGT Memory."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/api",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Billing (Stripe) ─────────────────────────────────────────────────────────

if BILLING_ENABLED:
    _km = KeyManager()
    app.add_middleware(BillingMiddleware, key_manager=_km, enabled=True)
    app.include_router(billing_router)
    logger.info("Billing ENABLED — API keys + rate limits active")
else:
    logger.info("Billing DISABLED — open access (dev mode)")

# ── Auth (опциональная) ───────────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Проверяет NGT_API_SECRET если он установлен."""
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
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
        version=VERSION,
        active_sessions=store.active_sessions() if store else 0,
        model=CHAT_MODEL,
        embedding_model=EMBEDDING_MODEL,
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
        memory_entries=len(wrapper.memory._entries),
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
    wrapper = store.get_or_create(request.session_id)

    try:
        emb = await wrapper._aembed(request.text)
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

    entries = len(wrapper.memory._entries)
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
        query_emb = await wrapper._aembed(request.query)
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
    logger.info(f"retrieve session={request.session_id} query='{request.query[:40]}' found={len(items)}")

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
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

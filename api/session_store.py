"""
Менеджер сессий для NGT Memory API (in-memory реализация).

Каждая сессия = изолированный NGTMemoryLLMWrapper со своей памятью.
Горячие сессии живут в памяти; при заданном persist_dir состояние
сохраняется на диск (перед eviction, периодически и на shutdown)
и лениво поднимается обратно в get_or_create — сессии переживают
рестарт сервера и TTL.

Для multi-worker production рекомендуется реализовать SessionStoreBase
с Redis/Postgres backend.
"""

import asyncio
import hashlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("ngt_api.session_store")

from ngt.core.llm_wrapper import NGTMemoryLLMWrapper
from api.session_store_base import SessionStoreBase


class SessionStore(SessionStoreBase):
    """
    Thread-safe хранилище сессий.

    Каждая сессия идентифицируется строкой session_id.
    Автоматически удаляет неактивные сессии через session_ttl_seconds.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4.1-nano",
        embedding_model: str = "text-embedding-3-small",
        memory_top_k: int = 5,
        memory_threshold: float = 0.25,
        use_graph: bool = True,
        session_ttl_seconds: int = 3600,
        max_sessions: int = 100,
        base_url: Optional[str] = None,
        embedding_dim: int = 1536,
        persist_dir: str = "",
    ):
        self._api_key = openai_api_key
        self._base_url = base_url
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._memory_top_k = memory_top_k
        self._memory_threshold = memory_threshold
        self._use_graph = use_graph
        self._session_ttl = session_ttl_seconds
        self._max_sessions = max_sessions

        self._sessions: Dict[str, NGTMemoryLLMWrapper] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Персистентность на диск (пустой persist_dir = выключена)
        self._persist_dir: Optional[Path] = Path(persist_dir) if persist_dir else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        # asyncio.Lock на сессию — сериализует конкурентные запросы одного session_id
        self._session_locks: Dict[str, asyncio.Lock] = {}

    # ── Persistence helpers ─────────────────────────────────────────

    def _session_path(self, session_id: str) -> Path:
        """Базовый путь файлов сессии. Санитизация + sha256-суффикс —
        защита от path traversal и коллизий после санитизации."""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:40]
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        return self._persist_dir / f"{safe}-{digest}"

    def _persist_session(self, session_id: str, wrapper: NGTMemoryLLMWrapper) -> bool:
        """Сохраняет одну сессию на диск. Ошибка не роняет вызывающего."""
        if self._persist_dir is None:
            return False
        try:
            wrapper.save_state(self._session_path(session_id))
            return True
        except Exception:
            logger.exception("failed to persist session %s", session_id)
            return False

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Лок сессии для эндпоинтов: async with store.get_lock(sid): ..."""
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    def save_all(self) -> int:
        """Сохраняет все активные сессии на диск. Возвращает число сохранённых."""
        if self._persist_dir is None:
            return 0
        with self._lock:
            items = list(self._sessions.items())
        saved = 0
        for sid, wrapper in items:
            if self._persist_session(sid, wrapper):
                saved += 1
        return saved

    def get_or_create(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Возвращает существующую сессию или создаёт новую."""
        with self._lock:
            self._evict_stale()

            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    self._evict_oldest()

                wrapper = NGTMemoryLLMWrapper(
                    openai_api_key=self._api_key,
                    base_url=self._base_url,
                    model=self._model,
                    embedding_model=self._embedding_model,
                    embedding_dim=self._embedding_dim,
                    memory_top_k=self._memory_top_k,
                    memory_threshold=self._memory_threshold,
                    use_graph=self._use_graph,
                    verbose=False,
                )
                # Лениво поднимаем сессию с диска, если она сохранялась ранее
                if self._persist_dir is not None:
                    try:
                        if wrapper.load_state(self._session_path(session_id)):
                            logger.info("session %s restored from disk", session_id)
                    except Exception:
                        logger.exception("failed to restore session %s — starting fresh", session_id)
                self._sessions[session_id] = wrapper

            self._last_access[session_id] = time.time()
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        """Возвращает сессию или None если не существует."""
        with self._lock:
            if session_id in self._sessions:
                self._last_access[session_id] = time.time()
                return self._sessions[session_id]
            return None

    def reset(self, session_id: str) -> bool:
        """Удаляет сессию (следующий запрос создаст новую с чистой памятью)."""
        with self._lock:
            existed = session_id in self._sessions
            if existed:
                del self._sessions[session_id]
                del self._last_access[session_id]
            self._session_locks.pop(session_id, None)
            # Удаляем и файлы — reset означает полную очистку памяти
            if self._persist_dir is not None:
                base = self._session_path(session_id)
                for suffix in (".memory.pt", ".session.pt"):
                    f = Path(str(base) + suffix)
                    if f.exists():
                        f.unlink()
                        existed = True
            return existed

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def session_ids(self):
        with self._lock:
            return list(self._sessions.keys())

    def _evict_stale(self):
        """Удаляет сессии старше TTL (с сохранением на диск)."""
        now = time.time()
        stale = [
            sid for sid, ts in self._last_access.items()
            if now - ts > self._session_ttl
        ]
        for sid in stale:
            if not self._evict_one(sid):
                continue

    def _evict_oldest(self):
        """Удаляет самую старую незанятую сессию при переполнении."""
        if not self._last_access:
            return
        for sid in sorted(self._last_access, key=self._last_access.get):
            if self._evict_one(sid):
                return

    def _evict_one(self, session_id: str) -> bool:
        """Вытесняет сессию: пропускает занятые, сохраняет на диск перед удалением."""
        lock = self._session_locks.get(session_id)
        if lock is not None and lock.locked():
            return False  # сессия сейчас обрабатывает запрос — не трогаем
        wrapper = self._sessions.get(session_id)
        if wrapper is None:
            return False
        self._persist_session(session_id, wrapper)
        del self._sessions[session_id]
        del self._last_access[session_id]
        self._session_locks.pop(session_id, None)
        return True

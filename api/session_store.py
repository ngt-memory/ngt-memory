"""
Менеджер сессий для NGT Memory API.

Каждая сессия = изолированный NGTMemoryLLMWrapper со своей памятью.
Сессии хранятся в памяти (in-memory). При перезапуске сервера — сброс.
"""

import threading
import time
from typing import Dict, Optional

from ngt.core.llm_wrapper import NGTMemoryLLMWrapper


class SessionStore:
    """
    Thread-safe хранилище сессий.

    Каждая сессия идентифицируется строкой session_id.
    Автоматически удаляет неактивные сессии через session_ttl_seconds.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        memory_top_k: int = 5,
        memory_threshold: float = 0.25,
        use_graph: bool = True,
        session_ttl_seconds: int = 3600,
        max_sessions: int = 100,
    ):
        self._api_key = openai_api_key
        self._model = model
        self._embedding_model = embedding_model
        self._memory_top_k = memory_top_k
        self._memory_threshold = memory_threshold
        self._use_graph = use_graph
        self._session_ttl = session_ttl_seconds
        self._max_sessions = max_sessions

        self._sessions: Dict[str, NGTMemoryLLMWrapper] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Возвращает существующую сессию или создаёт новую."""
        with self._lock:
            self._evict_stale()

            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    self._evict_oldest()

                self._sessions[session_id] = NGTMemoryLLMWrapper(
                    openai_api_key=self._api_key,
                    model=self._model,
                    embedding_model=self._embedding_model,
                    memory_top_k=self._memory_top_k,
                    memory_threshold=self._memory_threshold,
                    use_graph=self._use_graph,
                    verbose=False,
                )

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
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._last_access[session_id]
                return True
            return False

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def session_ids(self):
        with self._lock:
            return list(self._sessions.keys())

    def _evict_stale(self):
        """Удаляет сессии старше TTL."""
        now = time.time()
        stale = [
            sid for sid, ts in self._last_access.items()
            if now - ts > self._session_ttl
        ]
        for sid in stale:
            del self._sessions[sid]
            del self._last_access[sid]

    def _evict_oldest(self):
        """Удаляет самую старую сессию при переполнении."""
        if not self._last_access:
            return
        oldest = min(self._last_access, key=self._last_access.get)
        del self._sessions[oldest]
        del self._last_access[oldest]

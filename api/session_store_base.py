"""
Абстрактный интерфейс для session storage.

Позволяет подменять реализацию хранилища сессий
(in-memory, Redis, Postgres) без изменения API-слоя.

Использование:
    from api.session_store_base import SessionStoreBase

    class RedisSessionStore(SessionStoreBase):
        ...
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ngt.core.llm_wrapper import NGTMemoryLLMWrapper


class SessionStoreBase(ABC):
    """Базовый интерфейс для всех реализаций session store."""

    @abstractmethod
    def get_or_create(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Возвращает существующую сессию или создаёт новую."""
        ...

    @abstractmethod
    def get(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        """Возвращает сессию или None если не существует."""
        ...

    @abstractmethod
    def reset(self, session_id: str) -> bool:
        """Удаляет сессию. Возвращает True если сессия существовала."""
        ...

    @abstractmethod
    def active_sessions(self) -> int:
        """Количество активных сессий."""
        ...

    # ── Неабстрактные дефолты (реализации могут переопределить) ──────

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """asyncio.Lock на сессию — сериализует конкурентные запросы одного session_id."""
        if not hasattr(self, "_base_session_locks"):
            self._base_session_locks: Dict[str, asyncio.Lock] = {}
        lock = self._base_session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._base_session_locks[session_id] = lock
        return lock

    def save_all(self) -> int:
        """Сохраняет все сессии на диск. Возвращает число сохранённых. По умолчанию no-op."""
        return 0

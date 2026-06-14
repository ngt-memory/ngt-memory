"""
Абстрактный интерфейс для session storage.

Позволяет подменять реализацию хранилища сессий
(in-memory, Redis, Postgres) без изменения API-слоя.

Использование:
    from api.session_store_base import SessionStoreBase

    class RedisSessionStore(SessionStoreBase):
        ...

Async-контракт (используется эндпоинтами):
    async with store.get_lock(sid):
        wrapper = await store.get_or_create_async(sid)
        ... мутация wrapper ...
        await store.commit(sid)     # no-op для in-memory, запись для Redis

Sync-методы (get_or_create/get/reset) остаются для тестов и скриптов.
In-memory стор реализует и то, и другое; Redis-стор — только async.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional

from ngt.core.llm_wrapper import NGTMemoryLLMWrapper


class SessionStoreBase(ABC):
    """Базовый интерфейс для всех реализаций session store."""

    @abstractmethod
    def get_or_create(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Возвращает существующую сессию или создаёт новую (sync)."""
        ...

    @abstractmethod
    def get(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        """Возвращает сессию или None если не существует (sync)."""
        ...

    @abstractmethod
    def reset(self, session_id: str) -> bool:
        """Удаляет сессию. Возвращает True если сессия существовала (sync)."""
        ...

    @abstractmethod
    def active_sessions(self) -> int:
        """Количество активных сессий."""
        ...

    # ── Async API (дефолтные обёртки поверх sync; Redis переопределяет) ──

    async def get_or_create_async(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Async-версия get_or_create. По умолчанию — sync в потоке."""
        return await asyncio.to_thread(self.get_or_create, session_id)

    async def get_async(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        return self.get(session_id)

    async def reset_async(self, session_id: str) -> bool:
        return self.reset(session_id)

    async def commit(self, session_id: str) -> None:
        """Фиксирует состояние сессии после мутации.

        Для in-memory — no-op (объект изменён по ссылке).
        Для Redis — сериализует и пишет состояние обратно в стор.
        """
        return None

    async def aclose(self) -> None:
        """Закрывает соединения backend (Redis). По умолчанию no-op."""
        return None

    # ── Неабстрактные дефолты ────────────────────────────────────────

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

    def total_entries(self) -> int:
        """Суммарное число записей по всем активным сессиям."""
        return 0

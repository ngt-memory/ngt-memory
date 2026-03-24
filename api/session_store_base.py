"""
Абстрактный интерфейс для session storage.

Позволяет подменять реализацию хранилища сессий
(in-memory, Redis, Postgres) без изменения API-слоя.

Использование:
    from api.session_store_base import SessionStoreBase

    class RedisSessionStore(SessionStoreBase):
        ...
"""

from abc import ABC, abstractmethod
from typing import Optional

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

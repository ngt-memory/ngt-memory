"""
SessionOwnership — привязка session_id к владельцу (идентичности вызывающего).

ЗАЧЕМ. Без этого session_id — единственная граница доступа, и она ни к чему
не привязана. В мультиарендном хостинге любой, кто подберёт чужой session_id,
прочитает чужую память через /retrieve. Этот реестр привязывает каждую сессию
к идентичности первого обратившегося (хэш API-ключа) и отклоняет доступ
с чужой идентичностью.

ПОЛИТИКА claim-on-first-use:
  - Сессия без владельца → первый обратившийся становится владельцем (claim).
  - Сессия с владельцем → доступ только этой идентичности, иначе отказ.
  - release() снимает владение (вызывается при сбросе сессии её владельцем).

КОГДА АКТИВНО. Только когда вызывающих можно различить (см. config
session_ownership): режим биллинга с персональными ключами. Для self-hosted
(без auth) и shared-secret (один общий ключ) владение выключено — идентичность
там одна на всех, разделять нечего, поведение остаётся прежним.

ХРАНИЛИЩЕ:
  - backend=memory: dict в процессе + опциональная персистентность в JSON-файл
    (owners.json в persist_dir), чтобы владение пережило рестарт.
  - backend=redis: hash ngt:owners {session_id: owner_id} с reuse того же
    клиента, что и RedisSessionStore. Корректно при N воркерах.

owner_id — это sha256(api_key)[:32], сам ключ нигде не хранится.
"""

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("ngt_api.ownership")


def derive_owner_id(api_key: str) -> str:
    """owner_id = усечённый sha256 ключа. Сам ключ не хранится."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


class OwnershipError(Exception):
    """Доступ к сессии с чужой идентичностью."""


class SessionOwnership:
    """Реестр владения. backend выбирается наличием redis-клиента."""

    def __init__(
        self,
        redis=None,
        persist_dir: str = "",
        redis_key: str = "ngt:owners",
    ):
        self._redis = redis
        self._redis_key = redis_key
        self._lock = threading.Lock()
        self._owners: Dict[str, str] = {}

        # Персистентность владения на диск (только memory backend)
        self._path: Optional[Path] = None
        if redis is None and persist_dir:
            self._path = Path(persist_dir) / "owners.json"
            self._load_from_disk()

    # ── disk (memory backend) ────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            self._owners = json.loads(self._path.read_text(encoding="utf-8"))
            logger.info("loaded %d session-owner mappings from disk", len(self._owners))
        except Exception:
            logger.exception("failed to load owners.json — starting empty")
            self._owners = {}

    def _flush_to_disk(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._owners, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.exception("failed to persist owners.json")

    # ── core policy ──────────────────────────────────────────────────

    async def claim(self, session_id: str, owner_id: str) -> bool:
        """Закрепляет владение. Возвращает True, если вызывающий — владелец
        (только что закрепил или уже владел), False — если владеет кто-то другой.

        Атомарность:
          - redis: HSETNX (атомарный claim) + последующая сверка.
          - memory: под threading-локом.
        """
        if self._redis is not None:
            # HSETNX вернёт 1, если поля не было (мы стали владельцем),
            # 0 — если уже есть (надо сверить, наш ли это owner_id).
            created = await self._redis.hsetnx(self._redis_key, session_id, owner_id)
            if created == 1:
                return True
            current = await self._redis.hget(self._redis_key, session_id)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
            return current == owner_id

        with self._lock:
            current = self._owners.get(session_id)
            if current is None:
                self._owners[session_id] = owner_id
                self._flush_to_disk()
                return True
            return current == owner_id

    async def owner_of(self, session_id: str) -> Optional[str]:
        if self._redis is not None:
            current = await self._redis.hget(self._redis_key, session_id)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
            return current
        with self._lock:
            return self._owners.get(session_id)

    async def release(self, session_id: str) -> None:
        """Снимает владение (при сбросе сессии владельцем)."""
        if self._redis is not None:
            await self._redis.hdel(self._redis_key, session_id)
            return
        with self._lock:
            if session_id in self._owners:
                del self._owners[session_id]
                self._flush_to_disk()

    async def ensure(self, session_id: str, owner_id: Optional[str]) -> None:
        """Гарантирует доступ вызывающего к сессии. Бросает OwnershipError при
        чужом владельце. Если owner_id is None (владение выключено) — no-op.
        """
        if owner_id is None:
            return
        ok = await self.claim(session_id, owner_id)
        if not ok:
            raise OwnershipError(session_id)

    def count(self) -> int:
        """Число известных привязок (для memory backend; redis — HLEN не нужен здесь)."""
        if self._redis is not None:
            return -1  # неизвестно без обращения к redis
        with self._lock:
            return len(self._owners)

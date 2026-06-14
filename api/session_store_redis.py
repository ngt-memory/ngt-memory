"""
RedisSessionStore — session backend на Redis для multi-worker / multi-instance.

Архитектура:
  - Состояние сессии (memory v2 json + safetensors-байты + session json)
    хранится в Redis по ключам ngt:sess:{id}:{part} с TTL.
  - Каждый воркер держит локальный hot-cache десериализованных wrapper'ов;
    после каждого мутирующего запроса состояние сериализуется обратно в Redis,
    а версия (ngt:sess:{id}:ver) инкрементится. Перед использованием
    кэша версия сверяется — если другой воркер изменил сессию, локальная
    копия перечитывается.
  - Межпроцессная сериализация запросов одного session_id — через
    Redis-лок (SET NX PX + проверка владельца при освобождении).

Это даёт корректность без sticky sessions: любой воркер может обслужить
любой session_id. Цена — сериализация состояния на каждый мутирующий
запрос (для сессий в тысячи записей заметно; для типичных диалогов —
единицы–десятки мс).

Зависимость: redis>=5.0 (async client).

Использование (api/main.py выбирает по NGT_SESSION_BACKEND=redis):
    store = RedisSessionStore(redis_url="redis://localhost:6379/0", ...)
"""

import asyncio
import io
import json
import logging
import time
import uuid
from typing import Dict, Optional

logger = logging.getLogger("ngt_api.session_store_redis")

from ngt.core.llm_wrapper import NGTMemoryLLMWrapper
from api.session_store_base import SessionStoreBase
from api import persistence as p

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    REDIS_AVAILABLE = False

try:
    from safetensors.torch import save as _st_save_bytes, load as _st_load_bytes
except ImportError:  # pragma: no cover
    _st_save_bytes = _st_load_bytes = None

import torch


_LOCK_TTL_MS = 60_000          # авто-освобождение зависшего лока
_LOCK_RETRY_DELAY = 0.05       # 50ms между попытками захвата

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class _RedisSessionLock:
    """Распределённый лок одного session_id (SET NX PX, безопасный release)."""

    def __init__(self, redis, key: str):
        self._redis = redis
        self._key = key
        self._token = uuid.uuid4().hex

    async def __aenter__(self):
        while True:
            ok = await self._redis.set(self._key, self._token, nx=True, px=_LOCK_TTL_MS)
            if ok:
                return self
            await asyncio.sleep(_LOCK_RETRY_DELAY)

    async def __aexit__(self, *exc):
        try:
            await self._redis.eval(_RELEASE_LUA, 1, self._key, self._token)
        except Exception:
            logger.exception("failed to release redis lock %s", self._key)


class RedisSessionStore(SessionStoreBase):
    """Session store с состоянием в Redis. Корректен при N воркерах."""

    def __init__(
        self,
        openai_api_key: str,
        redis_url: str = "redis://localhost:6379/0",
        model: str = "gpt-4.1-nano",
        embedding_model: str = "text-embedding-3-small",
        memory_top_k: int = 5,
        memory_threshold: float = 0.25,
        use_graph: bool = True,
        session_ttl_seconds: int = 3600,
        base_url: Optional[str] = None,
        embedding_dim: int = 1536,
        key_prefix: str = "ngt:sess:",
        **_ignored,
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis не установлен — pip install 'redis>=5.0'")
        if _st_save_bytes is None:
            raise RuntimeError("safetensors не установлен — pip install 'safetensors>=0.4'")

        self._redis = aioredis.from_url(redis_url, decode_responses=False)
        self._prefix = key_prefix
        self._ttl = session_ttl_seconds

        self._api_key = openai_api_key
        self._base_url = base_url
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._memory_top_k = memory_top_k
        self._memory_threshold = memory_threshold
        self._use_graph = use_graph

        # Локальный hot-cache: session_id → (wrapper, version)
        self._cache: Dict[str, tuple] = {}

    # ── Ключи ────────────────────────────────────────────────────────

    def _k(self, sid: str, part: str) -> str:
        return f"{self._prefix}{sid}:{part}"

    # ── Сериализация wrapper ↔ Redis ─────────────────────────────────

    def _serialize(self, wrapper: NGTMemoryLLMWrapper) -> Dict[str, bytes]:
        """wrapper → {mem_json, mem_tensors, sess_json} (всё bytes)."""
        memory = wrapper.memory
        tensors: Dict[str, torch.Tensor] = {}
        entries_meta = {}
        for eid, e in memory._entries.items():
            tensors[f"entry/{eid}"] = e.embedding.detach().cpu().clone().contiguous()
            entries_meta[str(eid)] = {
                "text": e.text, "metadata": e.metadata, "timestamp": e.timestamp,
                "importance": e.importance, "access_count": e.access_count,
                "concept_ids": e.concept_ids,
            }
        concepts_meta = {}
        for nid, c in memory.associations._id_to_concept.items():
            tensors[f"concept/{nid}"] = c.embedding.detach().cpu().clone().contiguous()
            concepts_meta[str(nid)] = {
                "name": c.name, "metadata": c.metadata,
                "created_at": c.created_at, "last_accessed": c.last_accessed,
                "access_count": c.access_count, "strength": c.strength,
            }
        mem_json = json.dumps({
            "format_version": p.FORMAT_VERSION,
            "embedding_dim": memory.embedding_dim,
            "max_entries": memory.max_entries,
            "entries": entries_meta,
            "next_entry_id": memory._next_entry_id,
            "concepts": concepts_meta,
            "next_concept_id": memory.associations._next_id,
            "graph_edges": {f"{a},{b}": w for (a, b), w in memory.associations._edges.items()},
            "session_id": memory._session_id,
            "stats": memory.stats,
        }, ensure_ascii=False).encode("utf-8")

        if not tensors:
            tensors = {"_empty": torch.zeros(1)}
        mem_tensors = _st_save_bytes(tensors)

        sess_json = json.dumps({
            "format_version": p.FORMAT_VERSION,
            "profile": p.profile_to_dict(wrapper.profile),
            "chat_history": wrapper._chat_history,
            "stats": wrapper._stats,
        }, ensure_ascii=False).encode("utf-8")

        return {"mem_json": mem_json, "mem_tensors": mem_tensors, "sess_json": sess_json}

    def _deserialize(self, blobs: Dict[str, Optional[bytes]]) -> NGTMemoryLLMWrapper:
        from ngt.core.llm_memory import NGTMemoryForLLM
        from ngt.core.memory_entry import MemoryEntry
        from ngt.core.association_graph import ConceptNode

        wrapper = self._new_wrapper()
        mem_json = blobs.get("mem_json")
        if not mem_json:
            return wrapper

        meta = json.loads(mem_json.decode("utf-8"))
        tensors = _st_load_bytes(blobs["mem_tensors"]) if blobs.get("mem_tensors") else {}

        memory = NGTMemoryForLLM(
            embedding_dim=meta["embedding_dim"],
            max_entries=meta.get("max_entries", 10000),
            device="cpu",
        )
        for eid_str, edata in meta.get("entries", {}).items():
            eid = int(eid_str)
            emb = tensors.get(f"entry/{eid}")
            if emb is None:
                continue
            entry = MemoryEntry(
                entry_id=eid, text=edata["text"], embedding=emb,
                metadata=edata.get("metadata", {}),
                importance=edata.get("importance", 1.0),
                concept_ids=edata.get("concept_ids", []),
            )
            entry.timestamp = edata.get("timestamp", time.time())
            entry.access_count = edata.get("access_count", 0)
            memory._entries[eid] = entry
        memory._next_entry_id = meta.get("next_entry_id", 0)
        memory._entry_id_list = list(memory._entries.keys())
        memory._index_dirty = True
        for eid, entry in memory._entries.items():
            for cid in entry.concept_ids:
                memory._concept_to_entries.setdefault(cid, []).append(eid)
        for nid_str, cdata in meta.get("concepts", {}).items():
            nid = int(nid_str)
            emb = tensors.get(f"concept/{nid}")
            if emb is None:
                continue
            concept = ConceptNode(nid, cdata["name"], emb, cdata.get("metadata", {}))
            concept.created_at = cdata.get("created_at", time.time())
            concept.last_accessed = cdata.get("last_accessed", time.time())
            concept.access_count = cdata.get("access_count", 1)
            concept.strength = cdata.get("strength", 1.0)
            memory.associations._id_to_concept[nid] = concept
            memory.associations._name_to_id[concept.name] = nid
            memory.associations._ensure_emb_capacity(nid + 1)
            memory.associations._embeddings[nid] = concept.embedding
            memory.associations._active_ids.append(nid)
        memory.associations._next_id = meta.get("next_concept_id", 0)
        memory.associations._emb_dirty = True
        for key_str, w in meta.get("graph_edges", {}).items():
            a, b = (int(x) for x in key_str.split(","))
            memory.associations._edges[(a, b)] = w
            memory.associations._adj.setdefault(a, {})[b] = w
            memory.associations._adj.setdefault(b, {})[a] = w
        memory.stats = meta.get("stats", memory.stats)
        wrapper.memory = memory

        sess_json = blobs.get("sess_json")
        if sess_json:
            sdata = json.loads(sess_json.decode("utf-8"))
            wrapper.profile = p.profile_from_dict(sdata.get("profile") or {})
            wrapper._chat_history = sdata.get("chat_history", [])
            wrapper._stats = sdata.get("stats", wrapper._stats)
        return wrapper

    def _new_wrapper(self) -> NGTMemoryLLMWrapper:
        return NGTMemoryLLMWrapper(
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

    # ── Async API (используется эндпоинтами) ─────────────────────────

    def get_lock(self, session_id: str):
        """Распределённый лок: async with store.get_lock(sid)."""
        return _RedisSessionLock(self._redis, self._k(session_id, "lock"))

    async def get_or_create_async(self, session_id: str) -> NGTMemoryLLMWrapper:
        """Вызывать ПОД get_lock(session_id)."""
        ver_raw = await self._redis.get(self._k(session_id, "ver"))
        remote_ver = int(ver_raw) if ver_raw else 0

        cached = self._cache.get(session_id)
        if cached is not None and cached[1] == remote_ver:
            return cached[0]

        if remote_ver == 0:
            wrapper = self._new_wrapper()
        else:
            blobs = {}
            for part in ("mem_json", "mem_tensors", "sess_json"):
                blobs[part] = await self._redis.get(self._k(session_id, part))
            wrapper = await asyncio.to_thread(self._deserialize, blobs)

        self._cache[session_id] = (wrapper, remote_ver)
        return wrapper

    async def commit(self, session_id: str) -> None:
        """Сериализует состояние в Redis. Вызывать после мутирующего запроса,
        всё ещё под session-локом."""
        cached = self._cache.get(session_id)
        if cached is None:
            return
        wrapper, _ = cached
        blobs = await asyncio.to_thread(self._serialize, wrapper)
        pipe = self._redis.pipeline()
        for part, data in blobs.items():
            pipe.set(self._k(session_id, part), data, ex=self._ttl)
        pipe.incr(self._k(session_id, "ver"))
        pipe.expire(self._k(session_id, "ver"), self._ttl)
        results = await pipe.execute()
        new_ver = int(results[-2])  # результат INCR
        self._cache[session_id] = (wrapper, new_ver)

    async def reset_async(self, session_id: str) -> bool:
        keys = [self._k(session_id, part)
                for part in ("mem_json", "mem_tensors", "sess_json", "ver")]
        deleted = await self._redis.delete(*keys)
        self._cache.pop(session_id, None)
        return deleted > 0

    async def get_async(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        ver = await self._redis.get(self._k(session_id, "ver"))
        if not ver:
            return None
        return await self.get_or_create_async(session_id)

    async def aclose(self) -> None:
        await self._redis.aclose()

    # ── Sync-заглушки SessionStoreBase (Redis-стор — async-only) ─────

    def get_or_create(self, session_id: str) -> NGTMemoryLLMWrapper:
        raise NotImplementedError("RedisSessionStore — async-only: get_or_create_async")

    def get(self, session_id: str) -> Optional[NGTMemoryLLMWrapper]:
        raise NotImplementedError("RedisSessionStore — async-only: get_async")

    def reset(self, session_id: str) -> bool:
        raise NotImplementedError("RedisSessionStore — async-only: reset_async")

    def active_sessions(self) -> int:
        # Точный подсчёт требует SCAN — для /health достаточно размера hot-cache
        return len(self._cache)

    def save_all(self) -> int:
        return 0  # состояние и так в Redis после каждого commit

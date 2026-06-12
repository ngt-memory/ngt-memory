"""
Тесты API-слоя NGT Memory (без сетевых вызовов).

OpenAI мокается через monkeypatch на NGTMemoryLLMWrapper:
- aembed_text/_aembed → детерминированный тензор (seed = abs(hash(text)) % 2**31,
  нормализованный): одинаковый текст даёт одинаковый embedding →
  store + retrieve того же текста гарантирует similarity 1.0;
- achat → пишет в память/историю как настоящий, возвращает "echo: {msg}".

settings.persist_dir → tmp_path. TestClient используется контекст-менеджером,
иначе lifespan (создание SessionStore) не отработает.
"""

import asyncio
import logging
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from api.config import settings
from ngt.core.llm_wrapper import NGTMemoryLLMWrapper


# ============ Mocks ============

def _det_embedding(dim: int, text: str) -> torch.Tensor:
    """Детерминированный нормализованный embedding: один текст → один вектор."""
    g = torch.Generator().manual_seed(abs(hash(text)) % 2**31)
    e = torch.randn(dim, generator=g)
    return e / e.norm()


async def _fake_aembed(self, text: str) -> torch.Tensor:
    self._stats["total_embed_calls"] += 1
    self._stats["latency_embed_ms"].append(0.1)
    return _det_embedding(self.embedding_dim, text)


def _fake_embed(self, text: str) -> torch.Tensor:
    self._stats["total_embed_calls"] += 1
    self._stats["latency_embed_ms"].append(0.1)
    return _det_embedding(self.embedding_dim, text)


async def _fake_achat(self, user_message: str, domain=None) -> dict:
    """Как настоящий achat: embed → retrieve → store → история, но без OpenAI."""
    turn = self._stats["total_turns"]
    emb = await self.aembed_text(user_message)
    memories = self._retrieve_memories(emb)
    self.profile.extract_and_update(user_message, confidence=1.0, source="user_explicit")
    self._store(user_message, emb, role="user", turn=turn)
    self.memory.flush_hebbian()
    response = f"echo: {user_message}"
    self._chat_history.append({"role": "user", "content": user_message})
    self._chat_history.append({"role": "assistant", "content": response})
    self._stats["total_turns"] += 1
    self._stats["total_memories_used"] += len(memories)
    self._stats["total_tokens_in"] += 10
    self._stats["total_tokens_out"] += 5
    self._stats["latency_chat_ms"].append(0.2)
    self._stats["total_chat_calls"] += 1
    return {
        "response": response,
        "memories_used": memories,
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 0.2,
        "profile_updates": [],
    }


async def _fake_achat_no_memory(self, user_message: str) -> dict:
    self._chat_history.append({"role": "user", "content": user_message})
    response = f"echo: {user_message}"
    self._chat_history.append({"role": "assistant", "content": response})
    return {
        "response": response,
        "memories_used": [],
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 0.2,
    }


@pytest.fixture
def mocked_wrapper(monkeypatch):
    """Подменяет все OpenAI-зависимые методы wrapper'а на детерминированные моки."""
    monkeypatch.setattr(NGTMemoryLLMWrapper, "aembed_text", _fake_aembed)
    monkeypatch.setattr(NGTMemoryLLMWrapper, "_aembed", _fake_aembed)
    monkeypatch.setattr(NGTMemoryLLMWrapper, "embed_text", _fake_embed)
    monkeypatch.setattr(NGTMemoryLLMWrapper, "_embed", _fake_embed)
    monkeypatch.setattr(NGTMemoryLLMWrapper, "achat", _fake_achat)
    monkeypatch.setattr(NGTMemoryLLMWrapper, "achat_no_memory", _fake_achat_no_memory)
    return monkeypatch


@pytest.fixture
def client(mocked_wrapper, monkeypatch, tmp_path):
    """TestClient с persist_dir → tmp_path; context manager — lifespan отрабатывает."""
    monkeypatch.setattr(settings, "persist_dir", str(tmp_path))
    monkeypatch.setattr(settings, "api_secret", "")
    from api.main import app
    with TestClient(app) as c:
        yield c


# ============ 1. /health ============

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == settings.version


# ============ 2-3. /store + /retrieve ============

def test_store(client):
    r = client.post("/store", json={"text": "Patient is allergic to penicillin",
                                    "session_id": "s_store"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["memory_entries"] == 1


def test_store_then_retrieve_same_text(client):
    text = "User prefers vegetarian restaurants in Kyoto"
    client.post("/store", json={"text": text, "session_id": "s_sr"})
    r = client.post("/retrieve", json={"query": text, "session_id": "s_sr",
                                       "top_k": 5, "threshold": 0.25})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert any(item["text"] == text for item in body["results"])
    assert body["results"][0]["score"] >= 0.25


# ============ 4. Изоляция сессий ============

def test_session_isolation(client):
    text = "Secret fact only for session A"
    client.post("/store", json={"text": text, "session_id": "iso_A"})
    r = client.post("/retrieve", json={"query": text, "session_id": "iso_B",
                                       "top_k": 5, "threshold": 0.1})
    assert r.status_code == 200
    assert r.json()["count"] == 0


# ============ 5-6. /chat ============

def test_chat_grows_memory(client):
    r1 = client.post("/chat", json={"message": "I am planning a trip to Kyoto",
                                    "session_id": "c1"})
    assert r1.status_code == 200
    e1 = r1.json()["memory_entries"]
    r2 = client.post("/chat", json={"message": "My budget is three thousand dollars",
                                    "session_id": "c1"})
    assert r2.json()["memory_entries"] > e1


def test_chat_no_memory(client):
    r = client.post("/chat", json={"message": "hello there friend",
                                   "session_id": "c2", "use_memory": False})
    assert r.status_code == 200
    assert r.json()["memories_count"] == 0


def test_chat_after_store_uses_memory(client):
    text = "User is vegetarian and avoids meat dishes"
    client.post("/store", json={"text": text, "session_id": "c3"})
    r = client.post("/chat", json={"message": text, "session_id": "c3"})
    assert r.status_code == 200
    assert r.json()["memories_count"] >= 1


# ============ 7. /session/reset ============

def test_reset_clears_memory(client):
    text = "Fact to be wiped by reset"
    client.post("/store", json={"text": text, "session_id": "rst"})
    r = client.post("/session/reset", json={"session_id": "rst"})
    assert r.status_code == 200
    assert r.json()["success"] is True
    r2 = client.post("/retrieve", json={"query": text, "session_id": "rst",
                                        "top_k": 5, "threshold": 0.1})
    assert r2.json()["count"] == 0


def test_reset_nonexistent_session(client):
    r = client.post("/session/reset", json={"session_id": "never_existed_xyz"})
    assert r.status_code == 200
    assert "already empty" in r.json()["message"]


# ============ 8. /session/{id}/stats ============

def test_stats_after_chat(client):
    client.post("/chat", json={"message": "one message for stats", "session_id": "st1"})
    r = client.get("/session/st1/stats")
    assert r.status_code == 200
    assert r.json()["total_turns"] == 1


def test_stats_unknown_session(client):
    r = client.get("/session/unknown_session_404/stats")
    assert r.status_code == 404


# ============ 9. Auth (связка с задачей 1: timing-safe verify_api_key) ============

@pytest.fixture
def auth_client(mocked_wrapper, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "persist_dir", str(tmp_path))
    monkeypatch.setattr(settings, "api_secret", "topsecret-key")
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_auth_missing_key_401(auth_client):
    r = auth_client.post("/store", json={"text": "any fact here", "session_id": "a"})
    assert r.status_code == 401


def test_auth_wrong_key_401(auth_client):
    r = auth_client.post("/store", json={"text": "any fact here", "session_id": "a"},
                         headers={"X-Api-Key": "wrong"})
    assert r.status_code == 401


def test_auth_correct_key_200(auth_client):
    r = auth_client.post("/store", json={"text": "any fact here", "session_id": "a"},
                         headers={"X-Api-Key": "topsecret-key"})
    assert r.status_code == 200


def test_auth_health_open(auth_client):
    assert auth_client.get("/health").status_code == 200


# ============ PII canary (задача 3) ============

def test_no_pii_in_logs(client, caplog):
    canary = "PII_CANARY_12345"
    with caplog.at_level(logging.DEBUG):
        client.post("/chat", json={"message": f"my secret is {canary}", "session_id": "pii"})
        client.post("/store", json={"text": f"stored secret {canary}", "session_id": "pii"})
        client.post("/retrieve", json={"query": f"find {canary}", "session_id": "pii"})
    for record in caplog.records:
        assert canary not in record.getMessage(), \
            f"PII утекла в лог: {record.getMessage()}"


# ============ 10. Конкурентность: лок сериализует один session_id ============

def test_concurrent_chat_serialized(client, monkeypatch):
    events = []

    async def slow_achat(self, user_message, domain=None):
        events.append(("start", user_message))
        await asyncio.sleep(0.05)
        events.append(("end", user_message))
        return {"response": f"echo: {user_message}", "memories_used": [],
                "tokens_in": 1, "tokens_out": 1, "latency_ms": 50.0,
                "profile_updates": []}

    monkeypatch.setattr(NGTMemoryLLMWrapper, "achat", slow_achat)

    import httpx
    from api.main import app

    async def fire():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await asyncio.gather(
                ac.post("/chat", json={"message": "msg-one", "session_id": "conc"}),
                ac.post("/chat", json={"message": "msg-two", "session_id": "conc"}),
            )

    asyncio.run(fire())

    assert len(events) == 4
    # Пары start/end НЕ чередуются: после start идёт end того же сообщения
    for i in (0, 2):
        assert events[i][0] == "start" and events[i + 1][0] == "end"
        assert events[i][1] == events[i + 1][1], f"interleaved: {events}"


# ============ 11. SessionStore: персистентность, санитизация, eviction ============

@pytest.fixture
def make_store(mocked_wrapper, tmp_path):
    from api.session_store import SessionStore

    def _make(**kw):
        params = dict(
            openai_api_key="sk-dummy",
            embedding_dim=64,
            persist_dir=str(tmp_path),
        )
        params.update(kw)
        return SessionStore(**params)

    return _make


def _add_fact(wrapper, text):
    emb = _det_embedding(wrapper.embedding_dim, text)
    wrapper.memory.store(embedding=emb, text=text, domain="test")


def test_store_save_all_and_restore(make_store):
    s1 = make_store()
    w = s1.get_or_create("persist_me")
    _add_fact(w, "fact that must survive restart")
    assert s1.save_all() == 1

    s2 = make_store()  # «рестарт сервера»
    w2 = s2.get_or_create("persist_me")
    assert w2.memory_entries_count == 1


def test_session_path_sanitized(make_store):
    s = make_store()
    p = s._session_path("../../etc/passwd")
    assert "/" not in p.name and ".." not in p.name
    # файл остаётся внутри persist_dir
    assert p.parent == s._persist_dir


def test_eviction_skips_locked_session(make_store):
    s = make_store()
    s.get_or_create("busy")

    async def check():
        lock = s.get_lock("busy")
        await lock.acquire()
        try:
            assert s._evict_one("busy") is False  # занятую не трогаем
        finally:
            lock.release()
        assert s._evict_one("busy") is True  # свободную — можно

    asyncio.run(check())


def test_reset_removes_both_files(make_store, tmp_path):
    s = make_store()
    w = s.get_or_create("wipe_me")
    _add_fact(w, "to be deleted")
    s.save_all()
    base = s._session_path("wipe_me")
    from pathlib import Path
    assert Path(str(base) + ".memory.pt").exists()
    assert Path(str(base) + ".session.pt").exists()

    s.reset("wipe_me")
    assert not Path(str(base) + ".memory.pt").exists()
    assert not Path(str(base) + ".session.pt").exists()


def test_overflow_evicted_session_restored_from_disk(make_store):
    s = make_store(max_sessions=3)
    for i in range(5):
        w = s.get_or_create(f"sess_{i}")
        _add_fact(w, f"unique fact for session {i}")
    assert s.active_sessions() <= 3
    # sess_0 вытеснена на диск при overflow — поднимается обратно с данными
    assert "sess_0" not in s.session_ids()
    w0 = s.get_or_create("sess_0")
    assert w0.memory_entries_count == 1

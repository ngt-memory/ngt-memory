"""
Тест правки 6 — привязка сессии к владельцу.
Проверяем три режима: strict (изоляция по ключу), off (общий доступ),
и что владелец сохраняется/снимается корректно.
"""
import os
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
os.environ["NGT_PERSIST_DIR"] = ""
os.environ["NGT_RATE_LIMIT_RPS"] = "0"

import torch
from unittest.mock import patch

async def _fake_aembed(self, text):
    g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
    v = torch.randn(1536, generator=g)
    return v / v.norm()

async def _fake_achat(self, message, domain=None):
    return {"response": f"echo: {message}", "memories_used": [],
            "tokens_in": 10, "tokens_out": 5, "latency_ms": 1.0, "profile_updates": None}

def _client(ownership="strict", **env):
    for k, v in env.items():
        os.environ[k] = v
    os.environ["NGT_SESSION_OWNERSHIP"] = ownership
    import importlib, api.config, api.main
    importlib.reload(api.config)
    importlib.reload(api.main)
    from fastapi.testclient import TestClient
    return TestClient(api.main.app)

def test_strict_isolation():
    with patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper._aembed", _fake_aembed), \
         patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper.achat", _fake_achat):
        # strict, но billing off — ownership должен включиться (strict = всегда)
        with _client("strict") as c:
            ALICE = {"X-API-Key": "sk-alice-secret"}
            BOB   = {"X-API-Key": "sk-bob-secret"}

            # Alice создаёт сессию "shared-id"
            r = c.post("/store", json={"session_id": "shared-id", "text": "Alice private medical note"}, headers=ALICE)
            assert r.status_code == 200, r.text

            # Bob пытается прочитать ТУ ЖЕ сессию → 403
            r = c.post("/retrieve", json={"session_id": "shared-id", "query": "medical"}, headers=BOB)
            assert r.status_code == 403, f"Bob должен получить 403, получил {r.status_code}: {r.text}"

            # Bob не может и чатиться в чужую сессию
            r = c.post("/chat", json={"session_id": "shared-id", "message": "hi"}, headers=BOB)
            assert r.status_code == 403, r.status_code

            # Bob не может посмотреть статистику
            r = c.get("/session/shared-id/stats", headers=BOB)
            assert r.status_code == 403, r.status_code

            # Bob не может сбросить чужую сессию
            r = c.post("/session/reset", json={"session_id": "shared-id"}, headers=BOB)
            assert r.status_code == 403, r.status_code

            # Alice — владелец, всё работает
            r = c.post("/retrieve", json={"session_id": "shared-id", "query": "medical"}, headers=ALICE)
            assert r.status_code == 200, r.text
            r = c.get("/session/shared-id/stats", headers=ALICE)
            assert r.status_code == 200, r.text

            # Запрос без ключа в strict → 401
            r = c.post("/store", json={"session_id": "x", "text": "no key here"})
            assert r.status_code == 401, r.status_code
        print("PASS strict_isolation (Bob заблокирован, Alice владеет, no-key→401)")

def test_reset_releases_ownership():
    with patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper._aembed", _fake_aembed):
        with _client("strict") as c:
            ALICE = {"X-API-Key": "sk-alice-secret"}
            BOB   = {"X-API-Key": "sk-bob-secret"}
            # Alice создаёт и сбрасывает
            c.post("/store", json={"session_id": "reclaim", "text": "alice data here"}, headers=ALICE)
            r = c.post("/session/reset", json={"session_id": "reclaim"}, headers=ALICE)
            assert r.status_code == 200, r.text
            # После сброса id освобождён — Bob может занять
            r = c.post("/store", json={"session_id": "reclaim", "text": "now bob data"}, headers=BOB)
            assert r.status_code == 200, f"Bob должен занять освобождённый id, получил {r.status_code}"
            # Теперь Alice — чужая для этого id
            r = c.post("/retrieve", json={"session_id": "reclaim", "query": "data"}, headers=ALICE)
            assert r.status_code == 403, r.status_code
        print("PASS reset_releases_ownership (id переходит к новому владельцу)")

def test_off_mode_no_regression():
    with patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper._aembed", _fake_aembed):
        with _client("off") as c:
            # Без ownership — разные ключи делят одну сессию (как в 0.23)
            r = c.post("/store", json={"session_id": "open", "text": "shared data here"}, headers={"X-API-Key": "k1"})
            assert r.status_code == 200, r.text
            r = c.post("/retrieve", json={"session_id": "open", "query": "data"}, headers={"X-API-Key": "k2"})
            assert r.status_code == 200, r.text  # другой ключ — доступ есть
            # И вообще без ключа работает
            r = c.post("/retrieve", json={"session_id": "open", "query": "data"})
            assert r.status_code == 200, r.text
        print("PASS off_mode_no_regression (self-hosted поведение сохранено)")

if __name__ == "__main__":
    test_strict_isolation()
    test_reset_releases_ownership()
    test_off_mode_no_regression()
    print("\n=== ТЕСТЫ ВЛАДЕНИЯ (правка 6) ПРОШЛИ ===")

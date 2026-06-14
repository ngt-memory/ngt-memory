"""
Сквозной тест правки 1 (лок + async-restore) и правки 5 (rate limit, metrics)
через FastAPI TestClient. OpenAI замокан — реального API не дёргаем.
"""
import os, asyncio
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
os.environ["NGT_PERSIST_DIR"] = ""           # in-memory
os.environ["NGT_RATE_LIMIT_RPS"] = "1000"        # включить rate limit
os.environ["NGT_RATE_LIMIT_BURST"] = "1000"
os.environ["NGT_METRICS_ENABLED"] = "true"

import torch
from unittest.mock import patch, AsyncMock, MagicMock

# Мок embedding (детерминированный по тексту) и chat
def _fake_emb(dim=1536):
    v = torch.randn(dim)
    return v / v.norm()

async def _fake_aembed(self, text):
    # стабильный вектор по хэшу текста
    g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
    v = torch.randn(1536, generator=g)
    return v / v.norm()

async def _fake_achat(self, message, domain=None):
    return {"response": f"echo: {message}", "memories_used": [],
            "tokens_in": 10, "tokens_out": 5, "latency_ms": 1.0,
            "profile_updates": None}

def test_endpoints_with_locking():
    with patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper._aembed", _fake_aembed), \
         patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper.achat", _fake_achat):
        from fastapi.testclient import TestClient
        # импортируем main ПОСЛЕ установки env
        import importlib
        import api.config, api.main
        importlib.reload(api.config)
        importlib.reload(api.main)
        from api.main import app

        with TestClient(app) as client:
            # health
            r = client.get("/health")
            assert r.status_code == 200, r.text
            assert r.json()["version"] == "0.24.0"

            # store
            r = client.post("/store", json={"session_id": "s1", "text": "I love hiking in mountains"})
            assert r.status_code == 200, r.text
            assert r.json()["memory_entries"] >= 1

            # retrieve (теперь под локом)
            r = client.post("/retrieve", json={"session_id": "s1", "query": "outdoor activities", "top_k": 3})
            assert r.status_code == 200, r.text
            assert "results" in r.json()

            # chat
            r = client.post("/chat", json={"session_id": "s1", "message": "hello there"})
            assert r.status_code == 200, r.text
            assert r.json()["response"].startswith("echo:")

            # stats
            r = client.get("/session/s1/stats")
            assert r.status_code == 200, r.text

            # metrics
            r = client.get("/metrics")
            assert r.status_code == 200, r.text
            assert b"ngt_requests_total" in r.content

            # reset
            r = client.post("/session/reset", json={"session_id": "s1"})
            assert r.status_code == 200, r.text

        print("PASS endpoints_with_locking (health/store/retrieve/chat/stats/metrics/reset)")

def test_rate_limit_429():
    os.environ["NGT_RATE_LIMIT_RPS"] = "5"
    os.environ["NGT_RATE_LIMIT_BURST"] = "3"
    with patch("ngt.core.llm_wrapper.NGTMemoryLLMWrapper._aembed", _fake_aembed):
        from fastapi.testclient import TestClient
        import importlib, api.config, api.main
        importlib.reload(api.config)
        importlib.reload(api.main)
        from api.main import app
        with TestClient(app) as client:
            # burst=3 → 4-й быстрый запрос на /store должен дать 429
            codes = []
            for _ in range(6):
                r = client.post("/store", json={"session_id": "rl", "text": "spam message here"})
                codes.append(r.status_code)
            assert 429 in codes, codes
            # health не лимитируется
            assert client.get("/health").status_code == 200
        print(f"PASS rate_limit_429 (codes={codes})")

if __name__ == "__main__":
    test_endpoints_with_locking()
    test_rate_limit_429()
    print("\n=== API-ТЕСТЫ (лок + rate limit + metrics) ПРОШЛИ ===")

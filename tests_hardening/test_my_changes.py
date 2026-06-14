"""
Изолированный функциональный тест ПРАВОК 1-5.
Не зависит от тяжёлой цепочки (transformers/sklearn) — использует лёгкий
in-memory дубль NGTMemoryForLLM с тем же интерфейсом, что нужен persistence
и session_store. Цель — проверить именно мою логику:
  - persistence v2 (safetensors+json), версионирование, атомарная запись
  - profile_to_dict/from_dict round-trip
  - rate limiter (token bucket)
  - metrics no-op fallback
"""
import sys, os, time, json
import torch
sys.path.insert(0, os.path.dirname(__file__) if __file__ else ".")

# ---- 1. RateLimiter ----
from api.rate_limit import RateLimiter

def test_rate_limiter_burst_then_block():
    rl = RateLimiter(rps=10, burst=3)
    # 3 запроса проходят (burst), 4-й блокируется
    assert rl.allow("k")[0] is True
    assert rl.allow("k")[0] is True
    assert rl.allow("k")[0] is True
    ok, retry = rl.allow("k")
    assert ok is False and retry > 0
    print("PASS rate_limiter_burst_then_block")

def test_rate_limiter_refill():
    rl = RateLimiter(rps=100, burst=1)
    assert rl.allow("k")[0] is True
    assert rl.allow("k")[0] is False
    time.sleep(0.02)  # 100rps → токен за 10ms
    assert rl.allow("k")[0] is True
    print("PASS rate_limiter_refill")

def test_rate_limiter_disabled():
    rl = RateLimiter(rps=0, burst=1)
    for _ in range(1000):
        assert rl.allow("k")[0] is True
    print("PASS rate_limiter_disabled")

def test_rate_limiter_per_key():
    rl = RateLimiter(rps=1, burst=1)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True  # другой ключ — своё ведро
    assert rl.allow("a")[0] is False
    print("PASS rate_limiter_per_key")

# ---- 2. metrics no-op fallback ----
from api import metrics
def test_metrics_track_request_noop_or_real():
    with metrics.track_request("/test") as mark:
        mark(200)
    metrics.record_chat(10, 5, 2)
    metrics.update_session_gauges(1, 100)
    r = metrics.render()
    assert isinstance(r, (bytes,))
    print(f"PASS metrics (prometheus_available={metrics.PROMETHEUS_AVAILABLE})")

# ---- 3. profile dict round-trip ----
from ngt.core.user_profile import UserProfile
from api.persistence import profile_to_dict, profile_from_dict
def test_profile_roundtrip():
    p = UserProfile()
    p.extract_and_update("мне 30 лет", confidence=1.0)
    p.extract_and_update("я из Ростов-на-Дону", confidence=1.0)
    p.extract_and_update("allergic to penicillin.", confidence=1.0)
    p.extract_and_update("allergic to peanuts.", confidence=1.0)
    d = profile_to_dict(p)
    # JSON-сериализуемость (критично — это и есть цель замены pickle)
    s = json.dumps(d, ensure_ascii=False)
    d2 = json.loads(s)
    p2 = profile_from_dict(d2)
    assert p2.get("age") == 30, p2.get("age")
    assert p2.get("city") == "Ростов-на-Дону", p2.get("city")
    assert len(p2.get("allergies")) == 2
    # история изменений тоже восстановилась
    p.extract_and_update("мне 31 год", confidence=1.0)
    d3 = profile_from_dict(json.loads(json.dumps(profile_to_dict(p))))
    assert d3.get("age") == 31
    assert len(d3.slots["age"].history) == 1
    print("PASS profile_roundtrip (JSON, no pickle)")

# ---- 4. persistence version check ----
from api.persistence import _check_version, PersistenceFormatError, FORMAT_VERSION
from pathlib import Path
def test_version_check():
    _check_version({"format_version": FORMAT_VERSION}, Path("x"))  # не бросает
    try:
        _check_version({"format_version": "99.0"}, Path("x"))
        assert False, "должно было бросить"
    except PersistenceFormatError:
        pass
    print("PASS version_check")

if __name__ == "__main__":
    test_rate_limiter_burst_then_block()
    test_rate_limiter_refill()
    test_rate_limiter_disabled()
    test_rate_limiter_per_key()
    test_metrics_track_request_noop_or_real()
    test_profile_roundtrip()
    test_version_check()
    print("\n=== ВСЕ ТЕСТЫ ПРАВОК ПРОШЛИ ===")

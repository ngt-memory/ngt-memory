"""
Правка 7 — ЖИВОЙ интеграционный тест Redis-бэкенда против настоящего redis-server.
Проверяет то, что раньше НЕ было проверено: round-trip через Redis, изоляцию
состояния между разными session_id, конкурентный доступ нескольких "воркеров"
к ОДНОЙ сессии (каждый воркер = свой экземпляр стора со своим hot-cache, как
в multi-worker деплое), версионную инвалидацию кэша, распределённый лок и
владение через Redis-hash.
"""
import os, asyncio, time
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"
import torch

from api.session_store_redis import RedisSessionStore, REDIS_AVAILABLE
from api.ownership import SessionOwnership, derive_owner_id, OwnershipError

REDIS_URL = "redis://localhost:6379/0"

def _make_store():
    return RedisSessionStore(
        openai_api_key="sk-test-dummy", redis_url=REDIS_URL,
        embedding_dim=32, session_ttl_seconds=3600,
    )

async def test_roundtrip_via_redis():
    """Сессия, записанная одним стором, читается другим (как другим воркером)."""
    s1 = _make_store()
    s2 = _make_store()
    sid = "rt-session"
    async with s1.get_lock(sid):
        w = await s1.get_or_create_async(sid)
        w.memory.store(embedding=torch.randn(32), text="fact about quantum physics",
                       concepts=["quantum", "physics"], domain="sci")
        w.memory.flush_hebbian()
        await s1.commit(sid)
    # Второй стор (пустой hot-cache) должен поднять состояние из Redis
    async with s2.get_lock(sid):
        w2 = await s2.get_or_create_async(sid)
        assert w2.memory.num_entries == 1, w2.memory.num_entries
        assert w2.memory.associations.num_concepts == 2
        results = w2.memory.retrieve(torch.randn(32), top_k=3)
        assert isinstance(results, list)
    await s1.aclose(); await s2.aclose()
    print("PASS roundtrip_via_redis (воркер B видит запись воркера A)")

async def test_version_cache_invalidation():
    """Hot-cache воркера B инвалидируется, когда A изменил сессию."""
    s_a = _make_store()
    s_b = _make_store()
    sid = "ver-session"
    # A создаёт, B читает (кэширует version=1)
    async with s_a.get_lock(sid):
        w = await s_a.get_or_create_async(sid)
        w.memory.store(embedding=torch.randn(32), text="first entry here", concepts=["a"])
        await s_a.commit(sid)
    async with s_b.get_lock(sid):
        wb = await s_b.get_or_create_async(sid)
        assert wb.memory.num_entries == 1
    # A добавляет вторую запись (version→2)
    async with s_a.get_lock(sid):
        wa = await s_a.get_or_create_async(sid)
        wa.memory.store(embedding=torch.randn(32), text="second entry here", concepts=["b"])
        await s_a.commit(sid)
    # B снова читает — должен увидеть version mismatch и перечитать (2 записи)
    async with s_b.get_lock(sid):
        wb2 = await s_b.get_or_create_async(sid)
        assert wb2.memory.num_entries == 2, f"кэш B не инвалидировался: {wb2.memory.num_entries}"
    await s_a.aclose(); await s_b.aclose()
    print("PASS version_cache_invalidation (stale-кэш перечитан)")

async def test_concurrent_writers_same_session():
    """N воркеров параллельно пишут в одну сессию под распределённым локом.
    Лок должен сериализовать доступ; без гонок и без потери коммитов внутри
    критической секции."""
    sid = "concurrent-session"
    N = 8
    stores = [_make_store() for _ in range(N)]

    async def worker(store, idx):
        # каждый воркер под локом добавляет свою запись поверх актуального состояния
        async with store.get_lock(sid):
            w = await store.get_or_create_async(sid)
            before = w.memory.num_entries
            w.memory.store(embedding=torch.randn(32), text=f"worker {idx} entry payload",
                           concepts=[f"w{idx}"])
            await store.commit(sid)
            after = w.memory.num_entries
            # под локом мы видели согласованное before→after (+1)
            assert after == before + 1

    await asyncio.gather(*(worker(stores[i], i) for i in range(N)))

    # Финальное состояние: ровно N записей (никто не затёр чужой коммит)
    checker = _make_store()
    async with checker.get_lock(sid):
        w = await checker.get_or_create_async(sid)
        assert w.memory.num_entries == N, f"ожидали {N} записей, получили {w.memory.num_entries}"
    for s in stores: await s.aclose()
    await checker.aclose()
    print(f"PASS concurrent_writers_same_session ({N} воркеров, {N} записей, лок держит)")

async def test_distributed_lock_mutual_exclusion():
    """Прямая проверка распределённого лока: пока один держит, второй ждёт."""
    s = _make_store()
    sid = "lock-session"
    order = []
    async def holder(tag, hold):
        async with s.get_lock(sid):
            order.append(f"{tag}-enter")
            await asyncio.sleep(hold)
            order.append(f"{tag}-exit")
    # Запускаем A (держит 0.3s) и чуть позже B; B не должен войти пока A внутри
    await asyncio.gather(holder("A", 0.3), holder("B", 0.05))
    # Корректная сериализация: A полностью внутри до входа B (или наоборот),
    # т.е. enter/exit не чередуются крест-накрест
    assert order in (
        ["A-enter","A-exit","B-enter","B-exit"],
        ["B-enter","B-exit","A-enter","A-exit"],
    ), order
    await s.aclose()
    print(f"PASS distributed_lock_mutual_exclusion (порядок={order})")

async def test_reset_via_redis():
    s = _make_store()
    sid = "reset-session"
    async with s.get_lock(sid):
        w = await s.get_or_create_async(sid)
        w.memory.store(embedding=torch.randn(32), text="to be deleted soon", concepts=["x"])
        await s.commit(sid)
    ok = await s.reset_async(sid)
    assert ok is True
    # После сброса — пустая новая сессия
    async with s.get_lock(sid):
        w2 = await s.get_or_create_async(sid)
        assert w2.memory.num_entries == 0
    await s.aclose()
    print("PASS reset_via_redis")

async def test_ownership_via_redis():
    """Владение через Redis-hash шарится между воркерами."""
    import redis.asyncio as aioredis
    r = aioredis.from_url(REDIS_URL, decode_responses=False)
    own_a = SessionOwnership(redis=r)   # воркер A
    own_b = SessionOwnership(redis=r)   # воркер B (тот же Redis)
    alice = derive_owner_id("sk-alice")
    bob = derive_owner_id("sk-bob")
    sid = "owned-session"
    # Alice захватывает через воркер A
    assert await own_a.claim(sid, alice) is True
    # Bob через воркер B — отказ (видит чужого владельца в Redis)
    assert await own_b.claim(sid, bob) is False
    # Alice через воркер B — проходит (тот же владелец)
    assert await own_b.claim(sid, alice) is True
    # ensure бросает для Bob
    try:
        await own_b.ensure(sid, bob); assert False
    except OwnershipError:
        pass
    # release освобождает — Bob теперь может занять
    await own_a.release(sid)
    assert await own_b.claim(sid, bob) is True
    await r.aclose()
    print("PASS ownership_via_redis (владение шарится между воркерами)")

async def main():
    if not REDIS_AVAILABLE:
        print("SKIP — redis client недоступен"); return
    # Идемпотентность: чистим тестовые сессии перед прогоном, иначе записи
    # прошлого запуска накопятся (реальная сессия так и должна — но тесту
    # нужно детерминированное начальное состояние).
    import redis.asyncio as aioredis
    _r = aioredis.from_url(REDIS_URL, decode_responses=False)
    _keys = await _r.keys("ngt:sess:*") + await _r.keys("ngt:owners")
    if _keys:
        await _r.delete(*_keys)
    await _r.aclose()

    await test_roundtrip_via_redis()
    await test_version_cache_invalidation()
    await test_concurrent_writers_same_session()
    await test_distributed_lock_mutual_exclusion()
    await test_reset_via_redis()
    await test_ownership_via_redis()
    print("\n=== ЖИВЫЕ REDIS-ТЕСТЫ (правка 7) ПРОШЛИ ===")

if __name__ == "__main__":
    asyncio.run(main())

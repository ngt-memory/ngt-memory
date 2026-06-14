"""
Интеграционный тест правок 2 и 3 на реальном NGTMemoryForLLM:
  - persistence v2: save_memory/load_memory round-trip (тензоры + граф + entries)
  - session_state round-trip (профиль/история/статистика)
  - атомарность записи (tmp→rename)
  - миграция легаси .pt → v2
  - бюджет памяти: _over_budget / _enforce_budget / eviction
"""
import os, sys, tempfile, json
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
import torch
from pathlib import Path

from ngt.core.llm_memory import NGTMemoryForLLM
from ngt.core.user_profile import UserProfile
from api import persistence as p


def _make_memory(n_entries=5, dim=32):
    m = NGTMemoryForLLM(embedding_dim=dim, max_entries=1000)
    for i in range(n_entries):
        emb = torch.randn(dim)
        m.store(embedding=emb, text=f"Fact number {i} about cats and dogs",
                concepts=[f"concept_{i}", "animals"], domain="test")
    m.flush_hebbian()
    return m


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sess"
        m = _make_memory(5, dim=32)
        orig_entries = m.num_entries
        orig_concepts = m.associations.num_concepts
        orig_edges = m.associations.num_edges
        # query до сохранения
        q = torch.randn(32)
        before = m.retrieve(q, top_k=3)

        p.save_memory(m, base)
        assert Path(str(base) + ".memory.safetensors").exists()
        assert Path(str(base) + ".memory.json").exists()

        m2 = p.load_memory(base)
        assert m2 is not None
        assert m2.num_entries == orig_entries, (m2.num_entries, orig_entries)
        assert m2.associations.num_concepts == orig_concepts
        assert m2.associations.num_edges == orig_edges
        # retrieve работает после загрузки
        after = m2.retrieve(q, top_k=3)
        assert len(after) == len(before)
        # тексты записей идентичны
        for eid in m._entries:
            assert m._entries[eid].text == m2._entries[eid].text
        # embeddings совпадают (с точностью float32)
        for eid in m._entries:
            assert torch.allclose(m._entries[eid].embedding, m2._entries[eid].embedding, atol=1e-5)
        print("PASS persistence_roundtrip (entries/concepts/edges/embeddings)")


def test_session_state_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sess"
        prof = UserProfile()
        prof.extract_and_update("мне 28 лет", confidence=1.0)
        prof.extract_and_update("allergic to penicillin.", confidence=1.0)
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        stats = {"total_turns": 3}
        p.save_session_state(base, prof, history, stats)
        loaded = p.load_session_state(base)
        assert loaded is not None
        prof2, hist2, stats2 = loaded
        assert prof2.get("age") == 28
        assert prof2.get("allergies") == ["penicillin"]
        assert hist2 == history
        assert stats2["total_turns"] == 3
        print("PASS session_state_roundtrip")


def test_atomic_write_no_partial():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sess"
        m = _make_memory(3, dim=16)
        p.save_memory(m, base)
        # .tmp файлов не осталось
        leftover = list(Path(d).glob("*.tmp"))
        assert not leftover, leftover
        print("PASS atomic_write_no_partial")


def test_legacy_pt_migration():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sess"
        # Создаём легаси .pt через старый NGTMemoryForLLM.save
        m = _make_memory(4, dim=24)
        m.save(Path(str(base) + ".memory.pt"))
        # session.pt тоже (имитируем старый формат wrapper)
        torch.save({"profile": UserProfile(), "chat_history": [], "stats": {}},
                   str(base) + ".session.pt")
        assert p.legacy_pt_exists(base)

        # Грузим через wrapper.load_state (должен взять легаси и подготовить миграцию)
        from ngt.core.llm_wrapper import NGTMemoryLLMWrapper
        w = NGTMemoryLLMWrapper(openai_api_key="sk-test-dummy", embedding_dim=24)
        ok = w.load_state(base)
        assert ok is True
        assert w.memory.num_entries == 4, w.memory.num_entries

        # После save_state — должен появиться v2, и грузиться уже как v2
        w.save_state(base)
        assert Path(str(base) + ".memory.safetensors").exists()
        assert Path(str(base) + ".memory.json").exists()
        print("PASS legacy_pt_migration (.pt → v2)")


def test_version_incompatible_rejected():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "sess"
        m = _make_memory(2, dim=16)
        p.save_memory(m, base)
        # портим версию
        js = Path(str(base) + ".memory.json")
        data = json.loads(js.read_text())
        data["format_version"] = "99.0"
        js.write_text(json.dumps(data))
        try:
            p.load_memory(base)
            assert False, "должно было бросить PersistenceFormatError"
        except p.PersistenceFormatError:
            pass
        print("PASS version_incompatible_rejected")


def test_memory_budget_eviction():
    from api.session_store import SessionStore, _estimate_wrapper_bytes
    # Бюджет на ~2 сессии: считаем байты одной сессии и ставим лимит чуть выше
    store = SessionStore(
        openai_api_key="sk-test-dummy", embedding_dim=32,
        max_sessions=100,
        max_total_entries=100,   # высокий — лимитировать будем байтами
        max_total_bytes=1,       # драконовский — заставит вытеснять
    )
    # Вручную кладём сессии (минуя реальный OpenAI)
    for i in range(3):
        w = store._create_wrapper()
        for j in range(3):
            w.memory.store(embedding=torch.randn(32), text=f"s{i} fact {j}",
                           concepts=["x", "y"], domain="t")
        store._sessions[f"s{i}"] = w
        store._last_access[f"s{i}"] = float(i)
    # бюджет превышен
    assert store._over_budget() is True
    evicted = store._enforce_budget()
    assert evicted >= 1
    # старейшая (s0) вытеснена первой
    assert "s0" not in store._sessions
    print(f"PASS memory_budget_eviction (вытеснено {evicted})")


def test_budget_respects_busy_sessions():
    """Занятые (locked) сессии не вытесняются даже при превышении бюджета."""
    import asyncio
    from api.session_store import SessionStore
    store = SessionStore(openai_api_key="sk-test-dummy", embedding_dim=16,
                         max_total_bytes=1, max_total_entries=100)
    w = store._create_wrapper()
    w.memory.store(embedding=torch.randn(16), text="busy session fact", concepts=["a","b"])
    store._sessions["busy"] = w
    store._last_access["busy"] = 0.0
    # помечаем сессию занятой
    lock = store.get_lock("busy")

    async def run():
        async with lock:
            # под локом — eviction не должен её тронуть
            store._enforce_budget()
            assert "busy" in store._sessions
    asyncio.run(run())
    print("PASS budget_respects_busy_sessions")


if __name__ == "__main__":
    test_persistence_roundtrip()
    test_session_state_roundtrip()
    test_atomic_write_no_partial()
    test_legacy_pt_migration()
    test_version_incompatible_rejected()
    test_memory_budget_eviction()
    test_budget_respects_busy_sessions()
    print("\n=== ВСЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПРОШЛИ ===")

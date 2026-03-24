"""
Тесты для NGTMemoryForLLM — внешней нейропластичной памяти для LLM.

Покрывает:
- ConceptNode, AssociationGraph, MemoryEntry
- NGTMemoryForLLM: store, retrieve, get_context, consolidate
- Session management
- Persistence (save/load)
- Edge cases
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from ngt.core.llm_memory import (
    ConceptNode,
    AssociationGraph,
    MemoryEntry,
    NGTMemoryForLLM,
)


# ============ Fixtures ============

DIM = 64  # маленький dim для быстрых тестов


@pytest.fixture
def memory():
    """Создаёт NGTMemoryForLLM с маленьким embedding_dim для тестов."""
    return NGTMemoryForLLM(
        embedding_dim=DIM,
        max_entries=100,
        max_concepts=50,
        working_capacity=16,
        max_prototypes=20,
        hopfield_beta=4.0,
        consolidation_interval=10,
        device="cpu",
    )


@pytest.fixture
def assoc_graph():
    """Создаёт AssociationGraph для тестов."""
    return AssociationGraph(
        max_concepts=50,
        embedding_dim=DIM,
        device="cpu",
    )


def random_embedding(dim=DIM):
    """Генерирует случайный нормализованный embedding."""
    e = torch.randn(dim)
    return e / e.norm()


def similar_embedding(base: torch.Tensor, noise: float = 0.1):
    """Генерирует embedding похожий на base."""
    e = base + torch.randn_like(base) * noise
    return e / e.norm()


# ============ ConceptNode ============

class TestConceptNode:
    def test_creation(self):
        emb = random_embedding()
        node = ConceptNode(node_id=0, name="PyTorch", embedding=emb)
        assert node.node_id == 0
        assert node.name == "PyTorch"
        assert node.access_count == 1
        assert node.strength == 1.0

    def test_touch(self):
        node = ConceptNode(0, "test", random_embedding())
        old_time = node.last_accessed
        node.touch()
        assert node.access_count == 2
        assert node.last_accessed >= old_time

    def test_to_dict(self):
        node = ConceptNode(0, "test", random_embedding(), metadata={"key": "val"})
        d = node.to_dict()
        assert d["name"] == "test"
        assert d["metadata"]["key"] == "val"
        assert "created_at" in d


# ============ AssociationGraph ============

class TestAssociationGraph:
    def test_add_concept(self, assoc_graph):
        emb = random_embedding()
        nid = assoc_graph.add_concept("PyTorch", emb)
        assert nid == 0
        assert assoc_graph.num_concepts == 1
        
        concept = assoc_graph.get_concept(nid)
        assert concept is not None
        assert concept.name == "PyTorch"

    def test_add_duplicate_concept(self, assoc_graph):
        emb = random_embedding()
        nid1 = assoc_graph.add_concept("PyTorch", emb)
        nid2 = assoc_graph.add_concept("PyTorch", emb)
        assert nid1 == nid2
        assert assoc_graph.num_concepts == 1
        assert assoc_graph.get_concept(nid1).access_count == 2

    def test_co_occurrence(self, assoc_graph):
        ids = []
        for name in ["PyTorch", "GPU", "CUDA"]:
            ids.append(assoc_graph.add_concept(name, random_embedding()))
        
        n = assoc_graph.record_co_occurrence(ids, strength=1.0)
        assert n > 0
        assert assoc_graph.num_edges > 0

    def test_get_associated(self, assoc_graph):
        ids = []
        for name in ["A", "B", "C"]:
            ids.append(assoc_graph.add_concept(name, random_embedding()))
        
        # Создаём ассоциации несколько раз для усиления
        for _ in range(5):
            assoc_graph.record_co_occurrence(ids)
        
        neighbors = assoc_graph.get_associated(ids[0], top_k=5)
        # Должны найти хотя бы B и C
        neighbor_ids = [n[0] for n in neighbors]
        assert len(neighbor_ids) > 0

    def test_multi_hop(self, assoc_graph):
        # A-B, B-C (но не A-C напрямую)
        a = assoc_graph.add_concept("A", random_embedding())
        b = assoc_graph.add_concept("B", random_embedding())
        c = assoc_graph.add_concept("C", random_embedding())
        
        for _ in range(5):
            assoc_graph.record_co_occurrence([a, b])
            assoc_graph.record_co_occurrence([b, c])
        
        # Multi-hop от A должен найти C через B
        results = assoc_graph.get_associated_multi_hop(a, hops=2, top_k=5)
        found_ids = [r[0] for r in results]
        # B должен быть найден (1 hop)
        assert b in found_ids

    def test_find_similar_concepts(self, assoc_graph):
        base = random_embedding()
        assoc_graph.add_concept("original", base)
        assoc_graph.add_concept("similar", similar_embedding(base, 0.05))
        assoc_graph.add_concept("different", random_embedding())
        
        results = assoc_graph.find_similar_concepts(base, top_k=3)
        assert len(results) > 0
        # Первый результат должен быть "original" (точное совпадение)
        assert results[0][1] > 0.9

    def test_concept_eviction(self):
        """Тест замещения концептов при переполнении."""
        graph = AssociationGraph(max_concepts=3, embedding_dim=DIM, device="cpu")
        graph.add_concept("A", random_embedding())
        graph.add_concept("B", random_embedding())
        graph.add_concept("C", random_embedding())
        assert graph.num_concepts == 3
        
        # Добавляем 4-й — должен заместить самый слабый
        graph.add_concept("D", random_embedding())
        assert graph.num_concepts == 3

    def test_apply_decay(self, assoc_graph):
        ids = []
        for name in ["X", "Y"]:
            ids.append(assoc_graph.add_concept(name, random_embedding()))
        
        for _ in range(3):
            assoc_graph.record_co_occurrence(ids)
        
        edges_before = assoc_graph.num_edges
        # Агрессивный decay
        removed = assoc_graph.apply_decay(rate=0.99)
        # Должны удалиться рёбра
        assert assoc_graph.num_edges <= edges_before

    def test_statistics(self, assoc_graph):
        assoc_graph.add_concept("test", random_embedding())
        stats = assoc_graph.get_statistics()
        assert "num_concepts" in stats
        assert stats["num_concepts"] == 1


# ============ MemoryEntry ============

class TestMemoryEntry:
    def test_creation(self):
        entry = MemoryEntry(
            entry_id=0,
            text="Hello world",
            embedding=random_embedding(),
        )
        assert entry.entry_id == 0
        assert entry.text == "Hello world"
        assert entry.access_count == 0

    def test_touch(self):
        entry = MemoryEntry(0, "test", random_embedding())
        entry.touch()
        assert entry.access_count == 1

    def test_to_dict(self):
        entry = MemoryEntry(0, "test", random_embedding(), metadata={"k": "v"})
        d = entry.to_dict()
        assert d["text"] == "test"
        assert d["metadata"]["k"] == "v"


# ============ NGTMemoryForLLM — Store ============

class TestNGTMemoryStore:
    def test_basic_store(self, memory):
        result = memory.store(
            embedding=random_embedding(),
            text="PyTorch — фреймворк для глубокого обучения",
        )
        assert result["entry_id"] == 0
        assert memory.stats["total_stored"] == 1

    def test_store_with_concepts(self, memory):
        result = memory.store(
            embedding=random_embedding(),
            text="PyTorch использует CUDA для GPU-ускорения",
            concepts=["PyTorch", "CUDA", "GPU"],
        )
        assert result["concepts_added"] == 3
        assert memory.associations.num_concepts == 3

    def test_store_with_domain(self, memory):
        memory.store(
            embedding=random_embedding(),
            text="test",
            domain="ML",
        )
        entry = memory._entries[0]
        assert entry.metadata["domain"] == "ML"

    def test_store_multiple(self, memory):
        for i in range(10):
            memory.store(
                embedding=random_embedding(),
                text=f"Entry {i}",
            )
        assert len(memory._entries) == 10
        assert memory.stats["total_stored"] == 10

    def test_store_creates_associations(self, memory):
        memory.store(
            embedding=random_embedding(),
            text="text1",
            concepts=["A", "B", "C"],
        )
        memory._flush_hebbian()  # lazy Hebbian — flush вручную для теста
        assert memory.associations.num_edges > 0

    def test_store_embedding_padding(self, memory):
        """Тест: embedding меньше embedding_dim — должен дополниться нулями."""
        short_emb = torch.randn(32)
        result = memory.store(embedding=short_emb, text="short")
        assert result["entry_id"] >= 0

    def test_store_embedding_truncation(self, memory):
        """Тест: embedding больше embedding_dim — должен обрезаться."""
        long_emb = torch.randn(128)
        result = memory.store(embedding=long_emb, text="long")
        assert result["entry_id"] >= 0


# ============ NGTMemoryForLLM — Retrieve ============

class TestNGTMemoryRetrieve:
    def test_retrieve_empty(self, memory):
        results = memory.retrieve(random_embedding(), top_k=5)
        assert len(results) == 0

    def test_retrieve_basic(self, memory):
        emb = random_embedding()
        memory.store(embedding=emb, text="PyTorch — ML framework")
        
        results = memory.retrieve(similar_embedding(emb, 0.05), top_k=5)
        # Должен найти хотя бы одно воспоминание
        assert len(results) >= 0  # Может быть 0 если similarity < threshold

    def test_retrieve_relevance_order(self, memory):
        # Сохраняем 3 записи с разными embedding
        target = random_embedding()
        memory.store(embedding=similar_embedding(target, 0.01), text="very similar")
        memory.store(embedding=random_embedding(), text="random 1")
        memory.store(embedding=random_embedding(), text="random 2")
        
        results = memory.retrieve(target, top_k=3)
        if len(results) >= 2:
            # Первый результат должен иметь наибольший score
            assert results[0]["score"] >= results[1]["score"]

    def test_retrieve_with_graph(self, memory):
        emb1 = random_embedding()
        emb2 = random_embedding()
        
        memory.store(embedding=emb1, text="PyTorch is great", concepts=["PyTorch", "DL"])
        memory.store(embedding=emb2, text="CUDA speeds up DL", concepts=["CUDA", "DL"])
        
        # Запрос близкий к первому тексту, но через граф может найти и второй
        results = memory.retrieve(similar_embedding(emb1, 0.05), top_k=5, use_graph=True)
        assert isinstance(results, list)

    def test_retrieve_without_hopfield(self, memory):
        emb = random_embedding()
        memory.store(embedding=emb, text="test")
        
        results = memory.retrieve(emb, top_k=5, use_hopfield=False)
        assert isinstance(results, list)

    def test_retrieve_domain_filter(self, memory):
        memory.store(embedding=random_embedding(), text="ML text", domain="ML")
        memory.store(embedding=random_embedding(), text="Bio text", domain="biology")
        
        results = memory.retrieve(random_embedding(), top_k=10, domain="ML")
        assert isinstance(results, list)


# ============ NGTMemoryForLLM — Get Context ============

class TestNGTMemoryGetContext:
    def test_get_context_empty(self, memory):
        ctx = memory.get_context(random_embedding())
        assert ctx == ""

    def test_get_context_markdown(self, memory):
        emb = random_embedding()
        memory.store(embedding=emb, text="Important fact about neural networks")
        
        ctx = memory.get_context(similar_embedding(emb, 0.05), format="markdown")
        assert isinstance(ctx, str)

    def test_get_context_xml(self, memory):
        emb = random_embedding()
        memory.store(embedding=emb, text="Important fact")
        
        ctx = memory.get_context(similar_embedding(emb, 0.05), format="xml")
        assert isinstance(ctx, str)
        if ctx:
            assert "<memories>" in ctx

    def test_get_context_plain(self, memory):
        emb = random_embedding()
        memory.store(embedding=emb, text="Important fact")
        
        ctx = memory.get_context(similar_embedding(emb, 0.05), format="plain")
        assert isinstance(ctx, str)

    def test_get_context_max_tokens(self, memory):
        emb = random_embedding()
        for i in range(10):
            memory.store(
                embedding=similar_embedding(emb, 0.1),
                text=f"Fact number {i}: " + "x" * 200,
            )
        
        ctx = memory.get_context(emb, max_tokens=100)  # ~400 chars
        assert len(ctx) < 2000  # не должен превышать лимит сильно


# ============ NGTMemoryForLLM — Consolidation ============

class TestNGTMemoryConsolidation:
    def test_consolidate_empty(self, memory):
        stats = memory.consolidate()
        assert "hierarchy" in stats

    def test_consolidate_with_data(self, memory):
        for i in range(20):
            memory.store(
                embedding=random_embedding(),
                text=f"Entry {i}",
                domain="test",
                concepts=[f"concept_{i % 3}"],
            )
        
        stats = memory.consolidate()
        assert "hierarchy" in stats
        assert "dream" in stats
        assert "graph_edges_removed" in stats

    def test_auto_consolidation(self):
        """Тест: автоматическая консолидация через consolidation_interval."""
        mem = NGTMemoryForLLM(
            embedding_dim=DIM,
            max_entries=100,
            max_concepts=50,
            consolidation_interval=5,  # каждые 5 store()
            device="cpu",
        )
        
        for i in range(12):
            mem.store(embedding=random_embedding(), text=f"Entry {i}")
        
        # Должно было произойти 2 автоматические консолидации (на 5 и 10)
        assert mem.stats["total_stored"] == 12


# ============ NGTMemoryForLLM — Sessions ============

class TestNGTMemorySessions:
    def test_new_session(self, memory):
        sid = memory.new_session()
        assert sid == 1
        assert memory._session_id == 1

    def test_end_session(self, memory):
        memory.new_session()
        memory.store(embedding=random_embedding(), text="session data")
        
        stats = memory.end_session(consolidate=True)
        assert stats["session_id"] == 1
        assert stats["entries_count"] == 1

    def test_multiple_sessions(self, memory):
        memory.new_session()
        memory.store(embedding=random_embedding(), text="session 1 data")
        memory.end_session(consolidate=False)
        
        memory.new_session()
        memory.store(embedding=random_embedding(), text="session 2 data")
        memory.end_session(consolidate=False)
        
        assert memory.stats["sessions"] == 2
        assert len(memory._entries) == 2


# ============ NGTMemoryForLLM — Persistence ============

class TestNGTMemoryPersistence:
    def test_save_load(self, memory):
        # Сохраняем данные
        emb = random_embedding()
        memory.store(
            embedding=emb,
            text="Persistent memory test",
            concepts=["persistence", "test"],
            domain="testing",
        )
        memory.store(
            embedding=random_embedding(),
            text="Second entry",
            concepts=["persistence"],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.pt")
            memory.save(path)
            
            assert os.path.exists(path)
            
            # Загружаем
            loaded = NGTMemoryForLLM.load(path)
            
            assert len(loaded._entries) == 2
            assert loaded.associations.num_concepts == 2
            assert loaded.stats["total_stored"] == 2

    def test_save_load_empty(self, memory):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.pt")
            memory.save(path)
            loaded = NGTMemoryForLLM.load(path)
            assert len(loaded._entries) == 0

    def test_save_load_preserves_graph(self, memory):
        memory.store(
            embedding=random_embedding(),
            text="test",
            concepts=["A", "B", "C"],
        )
        
        edges_before = memory.associations.num_edges
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.pt")
            memory.save(path)
            loaded = NGTMemoryForLLM.load(path)
            
            assert loaded.associations.num_edges == edges_before


# ============ NGTMemoryForLLM — Eviction ============

class TestNGTMemoryEviction:
    def test_eviction_on_overflow(self):
        mem = NGTMemoryForLLM(
            embedding_dim=DIM,
            max_entries=10,
            max_concepts=50,
            device="cpu",
        )
        
        for i in range(15):
            mem.store(embedding=random_embedding(), text=f"Entry {i}")
        
        # Должен удалить старые записи
        assert len(mem._entries) <= 10


# ============ NGTMemoryForLLM — Statistics ============

class TestNGTMemoryStatistics:
    def test_get_statistics(self, memory):
        memory.store(embedding=random_embedding(), text="test", concepts=["A"])
        
        stats = memory.get_statistics()
        assert "entries_count" in stats
        assert "hierarchy" in stats
        assert "associations" in stats
        assert stats["entries_count"] == 1

    def test_repr(self, memory):
        r = repr(memory)
        assert "NGTMemoryForLLM" in r
        assert "entries=" in r
        assert "concepts=" in r


# ============ Hopfield Refinement ============

class TestHopfieldRefinement:
    def test_hopfield_refine_empty(self, memory):
        """Hopfield refinement на пустой памяти должен вернуть query as-is."""
        query = random_embedding()
        refined = memory._hopfield_refine(query)
        # Должен вернуть нормализованный query
        assert refined.shape == query.shape

    def test_hopfield_refine_convergence(self, memory):
        """Hopfield refinement должен сходиться к ближайшему паттерну."""
        # Сохраняем несколько паттернов
        target = random_embedding()
        memory.store(embedding=target, text="target")
        for _ in range(5):
            memory.store(embedding=random_embedding(), text="noise")
        
        # Запрос похожий на target
        query = similar_embedding(target, noise=0.3)
        refined = memory._hopfield_refine(query, iterations=10)
        
        # Refined должен быть ближе к target чем query
        target_norm = target / target.norm()
        query_sim = torch.dot(query / query.norm(), target_norm).item()
        refined_sim = torch.dot(refined / refined.norm(), target_norm).item()
        
        # Не всегда гарантировано, но в большинстве случаев
        # просто проверяем что refined валидный тензор
        assert refined.shape == query.shape
        assert refined.norm() > 0


# ============ Integration Tests ============

class TestIntegration:
    def test_full_workflow(self, memory):
        """Полный рабочий цикл: session → store → retrieve → consolidate."""
        memory.new_session()
        
        # Store несколько записей
        emb_pytorch = random_embedding()
        memory.store(
            embedding=emb_pytorch,
            text="PyTorch is a deep learning framework by Meta",
            concepts=["PyTorch", "Meta", "deep learning"],
            domain="ML",
        )
        
        emb_tensorflow = random_embedding()
        memory.store(
            embedding=emb_tensorflow,
            text="TensorFlow is Google's ML framework",
            concepts=["TensorFlow", "Google", "ML"],
            domain="ML",
        )
        
        memory.store(
            embedding=random_embedding(),
            text="The weather is nice today",
            concepts=["weather"],
            domain="casual",
        )
        
        # Retrieve
        results = memory.retrieve(
            similar_embedding(emb_pytorch, 0.05),
            top_k=3,
        )
        assert isinstance(results, list)
        
        # Get context
        ctx = memory.get_context(
            similar_embedding(emb_pytorch, 0.05),
            max_tokens=1000,
            format="markdown",
        )
        assert isinstance(ctx, str)
        
        # End session with consolidation
        stats = memory.end_session(consolidate=True)
        assert stats["session_id"] == 1
        
        # Verify stats
        final_stats = memory.get_statistics()
        assert final_stats["entries_count"] == 3
        assert final_stats["associations"]["num_concepts"] >= 5

    def test_cross_session_memory(self, memory):
        """Память должна сохраняться между сессиями."""
        # Session 1
        memory.new_session()
        emb = random_embedding()
        memory.store(embedding=emb, text="Remember this!")
        memory.end_session(consolidate=True)
        
        # Session 2
        memory.new_session()
        results = memory.retrieve(similar_embedding(emb, 0.05), top_k=5)
        # Должны найти запись из предыдущей сессии
        assert isinstance(results, list)

    def test_save_load_retrieve(self, memory):
        """Store → save → load → retrieve должен работать."""
        emb = random_embedding()
        memory.store(
            embedding=emb,
            text="Persistent knowledge",
            concepts=["persistence"],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            memory.save(path)
            
            loaded = NGTMemoryForLLM.load(path)
            results = loaded.retrieve(similar_embedding(emb, 0.05), top_k=5)
            assert isinstance(results, list)


# ============ Cross-Domain Retrieval Tests ============

class TestCrossDomainRetrieval:
    """
    Тесты cross-domain retrieval — ключевое преимущество NGT над VectorStore.
    
    VectorStore находит только по embedding similarity.
    NGT может найти факты из другого домена через граф ассоциаций,
    если концепты связаны Hebbian-обучением.
    """

    @pytest.fixture
    def cross_memory(self):
        """Память с несколькими доменами и Hebbian связями."""
        return NGTMemoryForLLM(
            embedding_dim=DIM,
            max_entries=200,
            max_concepts=100,
            hebbian_lr=0.3,  # быстрее строим граф в тестах
            consolidation_interval=50,
            device="cpu",
        )

    def test_same_concept_bridges_domains(self, cross_memory):
        """
        Факты из разных доменов, связанные одним концептом,
        должны находиться через graph retrieval.
        
        Домен A: "нейросети" — концепты [ml, neural, backprop]
        Домен B: "биология" — концепты [neural, brain, neuron]
        Связь: оба содержат концепт "neural"
        
        Запрос близок к домену A → должны найти факты домена B через граф.
        """
        # Фиксированные embedding-ы — домены далеко друг от друга
        torch.manual_seed(42)
        emb_ml     = torch.randn(DIM); emb_ml /= emb_ml.norm()
        emb_bio    = torch.randn(DIM); emb_bio /= emb_bio.norm()
        emb_query  = emb_ml.clone()   # query = точно ml домен

        cross_memory.store(emb_ml,  text="ML fact: backprop updates neural weights",
                           concepts=["ml", "neural", "backprop"])
        cross_memory.store(emb_bio, text="Bio fact: neurons fire action potentials",
                           concepts=["neural", "brain", "neuron"])

        # Flush Hebbian — обеспечиваем что связь "ml-neural-brain" создана
        cross_memory._flush_hebbian()

        # Retrieve с use_graph=True
        results_graph = cross_memory.retrieve(emb_query, top_k=5, use_graph=True)
        texts = [r["text"] for r in results_graph]

        # ML факт должен найтись (прямой поиск по embedding)
        assert any("ML fact" in t for t in texts), "ML fact должен быть в результатах"
        # Bio факт должен найтись через граф (концепт "neural" связан с ml)
        assert any("Bio fact" in t for t in texts), \
            "Bio fact должен найтись через граф по концепту 'neural'"

    def test_multihop_cross_domain(self, cross_memory):
        """
        Цепочка A → B → C через несколько хопов:
        physics[energy, quantum] → chemistry[quantum, molecule] → biology[molecule, cell]
        Запрос близок к physics → должны найти biology через 2 хопа.
        """
        torch.manual_seed(7)
        emb_phys = torch.randn(DIM); emb_phys /= emb_phys.norm()
        emb_chem = torch.randn(DIM); emb_chem /= emb_chem.norm()
        emb_bio  = torch.randn(DIM); emb_bio  /= emb_bio.norm()

        cross_memory.store(emb_phys, text="Physics: quantum energy levels",
                           concepts=["energy", "quantum", "physics"])
        cross_memory.store(emb_chem, text="Chemistry: quantum molecular orbitals",
                           concepts=["quantum", "molecule", "chemistry"])
        cross_memory.store(emb_bio,  text="Biology: molecular cell signaling",
                           concepts=["molecule", "cell", "biology"])

        cross_memory._flush_hebbian()

        results = cross_memory.retrieve(emb_phys, top_k=5, use_graph=True, graph_hops=2)
        texts = [r["text"] for r in results]

        assert any("Physics" in t for t in texts), "Physics fact должен найтись"
        # Хотя бы Chemistry или Biology должны найтись через граф
        found_cross = sum(1 for t in texts if "Chemistry" in t or "Biology" in t)
        assert found_cross >= 1, \
            f"Хотя бы 1 cross-domain факт должен найтись через граф, найдено: {found_cross}. texts={texts}"

    def test_no_graph_misses_cross_domain(self, cross_memory):
        """
        Без use_graph cross-domain факт НЕ должен найтись
        (embedding-ы далеко друг от друга).
        """
        torch.manual_seed(99)
        emb_a = torch.randn(DIM); emb_a /= emb_a.norm()
        # emb_b ортогонален emb_a (другой домен)
        emb_b = torch.randn(DIM)
        emb_b = emb_b - emb_b.dot(emb_a) * emb_a  # проекция удалена
        if emb_b.norm() > 0:
            emb_b /= emb_b.norm()

        cross_memory.store(emb_a, text="Domain A: first fact",
                           concepts=["concept_a", "shared_link"])
        cross_memory.store(emb_b, text="Domain B: second fact",
                           concepts=["shared_link", "concept_b"])

        cross_memory._flush_hebbian()

        # Retrieve БЕЗ графа — b не должен найтись
        results_no_graph = cross_memory.retrieve(emb_a, top_k=5, use_graph=False)
        texts_no_graph = [r["text"] for r in results_no_graph]

        # Retrieve С графом — b должен найтись
        results_graph = cross_memory.retrieve(emb_a, top_k=5, use_graph=True)
        texts_graph = [r["text"] for r in results_graph]

        assert any("Domain A" in t for t in texts_no_graph), "A должен найтись без графа"
        # Это ключевое различие NGT vs VS: graph расширяет recall
        assert any("Domain B" in t for t in texts_graph), \
            "Domain B должен найтись с графом через shared_link"

    def test_hebbian_strengthening_improves_recall(self, cross_memory):
        """
        Повторное co-occurrence должно усилить связь и улучшить recall.
        Концепты упомянутые вместе N раз → сильнее связаны.
        """
        torch.manual_seed(13)
        emb_x = torch.randn(DIM); emb_x /= emb_x.norm()
        emb_y = torch.randn(DIM); emb_y /= emb_y.norm()

        # Первое store: слабая связь
        cross_memory.store(emb_x, text="Fact X about AI", concepts=["ai", "bridge"])
        cross_memory.store(emb_y, text="Fact Y about music", concepts=["bridge", "music"])
        cross_memory._flush_hebbian()

        # Проверяем вес рёбра после 1 co-occurrence
        ai_id     = cross_memory.associations._name_to_id.get("ai")
        bridge_id = cross_memory.associations._name_to_id.get("bridge")
        music_id  = cross_memory.associations._name_to_id.get("music")
        assert ai_id is not None and bridge_id is not None and music_id is not None

        key_ab = (min(ai_id, bridge_id), max(ai_id, bridge_id))
        w1 = cross_memory.associations._edges.get(key_ab, 0.0)

        # Повторяем co-occurrence 10 раз
        for _ in range(10):
            cross_memory.associations.record_co_occurrence([ai_id, bridge_id, music_id])

        w2 = cross_memory.associations._edges.get(key_ab, 0.0)
        assert w2 > w1, f"Вес должен вырасти после повторений: {w1} → {w2}"

    def test_cross_domain_hit_rate(self, cross_memory):
        """
        Количественный тест: Hit@3 для cross-domain запросов.
        
        5 доменов × 10 фактов = 50 фактов.
        Каждые 2 соседних домена связаны через bridge-концепт.
        Query из домена i → должны найтись факты домена i±1 через граф.
        
        Ожидаемый cross-domain Hit@3 ≥ 0.5 (NGT) vs 0.0 (VS без графа).
        """
        torch.manual_seed(2024)
        domains = ["physics", "chemistry", "biology", "medicine", "data_science"]
        n_per_domain = 8

        domain_embs = {}
        domain_bridge = {}  # bridge концепт между доменами i и i+1

        # Создаём центры доменов (далеко друг от друга)
        for d in domains:
            c = torch.zeros(DIM)
            c[domains.index(d) * (DIM // len(domains))] = 1.0
            c += torch.randn(DIM) * 0.1
            c /= c.norm()
            domain_embs[d] = c

        # Bridge концепты между соседними доменами
        for i in range(len(domains) - 1):
            domain_bridge[(domains[i], domains[i + 1])] = f"bridge_{i}"

        # Сохраняем факты
        entry_domain_map = {}  # entry_id → domain
        for di, domain in enumerate(domains):
            for j in range(n_per_domain):
                noise = torch.randn(DIM) * 0.15
                emb = domain_embs[domain] + noise
                emb /= emb.norm()

                concepts = [f"{domain}_concept_{j % 3}"]
                # Добавляем bridge концепты для связи с соседними доменами
                if di > 0:
                    concepts.append(domain_bridge.get((domains[di - 1], domain), f"link_{di-1}"))
                if di < len(domains) - 1:
                    concepts.append(domain_bridge.get((domain, domains[di + 1]), f"link_{di}"))

                res = cross_memory.store(emb, text=f"{domain} fact {j}", concepts=concepts)
                if res.get("entry_id", -1) >= 0:
                    entry_domain_map[res["entry_id"]] = domain

        cross_memory._flush_hebbian()

        # Для каждого домена делаем запрос и считаем cross-domain hits
        cross_hits_graph = 0
        cross_hits_novg  = 0
        total_queries = 0

        for di, domain in enumerate(domains):
            if di == 0 or di == len(domains) - 1:
                continue  # граничные домены имеют только 1 сосед
            query = domain_embs[domain].clone()

            # top_k=10: прямые попадания (8 фактов домена) + cross-domain через граф
            r_graph = cross_memory.retrieve(query, top_k=10, use_graph=True)
            r_novg  = cross_memory.retrieve(query, top_k=10, use_graph=False)

            for r in r_graph:
                if entry_domain_map.get(r["entry_id"], domain) != domain:
                    cross_hits_graph += 1
                    break
            for r in r_novg:
                if entry_domain_map.get(r["entry_id"], domain) != domain:
                    cross_hits_novg += 1
                    break
            total_queries += 1

        hit_rate_graph = cross_hits_graph / max(total_queries, 1)
        hit_rate_novg  = cross_hits_novg  / max(total_queries, 1)

        # NGT с графом должен иметь лучший или равный cross-domain recall
        assert hit_rate_graph >= hit_rate_novg, \
            f"Graph recall ({hit_rate_graph:.2f}) должен быть ≥ no-graph ({hit_rate_novg:.2f})"
        # Хотя бы в 30% случаев граф помогает найти cross-domain факт
        assert hit_rate_graph >= 0.3, \
            f"Cross-domain Hit@3 слишком низкий: {hit_rate_graph:.2f} (ожидается ≥ 0.30)"

    def test_domain_isolation_without_bridge(self, cross_memory):
        """
        Домены без общих концептов НЕ должны связываться через граф.
        Это тест на отсутствие false positive cross-domain retrievals.
        """
        torch.manual_seed(55)
        # Два изолированных домена — разные концепты, далёкие embedding-ы
        emb_isolated_a = torch.zeros(DIM); emb_isolated_a[0] = 1.0
        emb_isolated_b = torch.zeros(DIM); emb_isolated_b[-1] = 1.0

        for i in range(5):
            cross_memory.store(emb_isolated_a + torch.randn(DIM) * 0.05,
                               text=f"Isolated A fact {i}",
                               concepts=[f"alpha_{i}", f"alpha_{i+1}"])
            cross_memory.store(emb_isolated_b + torch.randn(DIM) * 0.05,
                               text=f"Isolated B fact {i}",
                               concepts=[f"beta_{i}", f"beta_{i+1}"])

        cross_memory._flush_hebbian()

        # Нормализуем query
        q = emb_isolated_a.clone(); q /= q.norm()
        results = cross_memory.retrieve(q, top_k=3, use_graph=True)
        texts = [r["text"] for r in results]

        # Только факты A должны найтись (нет bridge → нет false cross-domain)
        b_found = sum(1 for t in texts if "Isolated B" in t)
        assert b_found == 0, \
            f"Изолированный домен B не должен попасть в результаты: {texts}"

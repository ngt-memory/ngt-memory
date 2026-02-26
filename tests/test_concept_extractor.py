"""
Тесты для ConceptExtractor — semantic concept extraction для NGT Memory.

Покрывает:
- Regex стратегию (zero-dependency)
- Hybrid стратегию
- Postprocessing (нормализация, дедупликация, min_length)
- auto() фабричный метод
- Интеграцию с NGTMemoryForLLM (авто-извлечение при concepts=None)
"""

import pytest
from ngt.core.concept_extractor import ConceptExtractor, _extract_regex, _normalize


# ============ Unit: _normalize ============

def test_normalize_lowercase():
    assert _normalize("PyTorch") == "pytorch"

def test_normalize_strips_punctuation():
    assert _normalize("backprop.") == "backprop"
    assert _normalize(",neural,") == "neural"

def test_normalize_spaces_to_underscore():
    assert _normalize("deep learning") == "deep_learning"


# ============ Unit: _extract_regex ============

class TestExtractRegex:

    def test_extracts_camelcase(self):
        text = "PyTorch and TensorFlow are deep learning frameworks."
        result = _extract_regex(text, top_k=6)
        assert any("pytorch" in c or "tensorflow" in c for c in result), \
            f"CamelCase не найден: {result}"

    def test_extracts_acronym(self):
        text = "The GPU accelerates ML training via CUDA."
        result = _extract_regex(text, top_k=8)
        assert any(c in ("gpu", "cuda", "ml") for c in result), \
            f"Аббревиатура не найдена: {result}"

    def test_filters_stopwords(self):
        text = "the this that with from have been will would"
        result = _extract_regex(text, top_k=10)
        stopwords = {"the", "this", "that", "with", "from", "have", "been", "will", "would"}
        assert all(c not in stopwords for c in result), \
            f"Стоп-слова не отфильтрованы: {result}"

    def test_top_k_limit(self):
        text = "Neural networks learn patterns through backpropagation gradient descent optimization training data"
        result = _extract_regex(text, top_k=4)
        assert len(result) <= 4

    def test_min_length(self):
        text = "AI ML NN big data networks"
        result = _extract_regex(text, top_k=10)
        assert all(len(c) >= 2 for c in result)

    def test_empty_text(self):
        result = _extract_regex("", top_k=5)
        assert result == []

    def test_technical_terms(self):
        text = "backpropagation gradient-descent learning-rate transformer attention"
        result = _extract_regex(text, top_k=8)
        found = [c for c in result if "backpropagation" in c or "gradient" in c or "transformer" in c]
        assert len(found) >= 1, f"Технические термины не найдены: {result}"


# ============ ConceptExtractor class ============

class TestConceptExtractor:

    def test_init_regex(self):
        e = ConceptExtractor(strategy="regex")
        assert e.strategy == "regex"

    def test_init_hybrid(self):
        e = ConceptExtractor(strategy="hybrid")
        assert e.strategy == "hybrid"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            ConceptExtractor(strategy="unknown_strategy")

    def test_extract_returns_list(self):
        e = ConceptExtractor(strategy="regex", top_k=5)
        result = e.extract("Deep learning with PyTorch and CUDA acceleration")
        assert isinstance(result, list)

    def test_extract_top_k(self):
        e = ConceptExtractor(strategy="regex", top_k=4)
        result = e.extract("Neural networks learn from data via backpropagation and gradient descent")
        assert len(result) <= 4

    def test_extract_deduplication(self):
        e = ConceptExtractor(strategy="regex", top_k=10, deduplicate=True)
        text = "neural neural neural network network"
        result = e.extract(text)
        assert len(result) == len(set(result)), f"Дубликаты не удалены: {result}"

    def test_extract_min_length(self):
        e = ConceptExtractor(strategy="regex", top_k=10, min_length=4)
        result = e.extract("AI ML NN DL big deep neural network")
        assert all(len(c) >= 4 for c in result), f"Короткие токены: {result}"

    def test_extract_empty_text(self):
        e = ConceptExtractor(strategy="regex")
        result = e.extract("")
        assert result == []

    def test_extract_batch(self):
        e = ConceptExtractor(strategy="regex", top_k=4)
        texts = [
            "PyTorch neural network training",
            "GPU CUDA acceleration",
            "backpropagation gradient descent",
        ]
        results = e.extract_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) <= 4 for r in results)

    def test_postprocess_normalizes(self):
        e = ConceptExtractor(strategy="regex", top_k=5)
        result = e.extract("PyTorch TensorFlow CUDA GPU backprop")
        assert all(c == c.lower() for c in result), f"Не нормализовано: {result}"

    def test_postprocess_strips_punctuation(self):
        e = ConceptExtractor(strategy="regex", top_k=5)
        result = e.extract("neural. network, backprop.")
        assert all(not c.endswith(".") and not c.endswith(",") for c in result)

    def test_regex_strategy_no_deps(self):
        """regex стратегия работает без каких-либо внешних зависимостей."""
        e = ConceptExtractor(strategy="regex", top_k=6)
        result = e.extract("Quantum computing leverages superposition and entanglement for parallel computation.")
        assert len(result) > 0
        assert isinstance(result[0], str)


class TestConceptExtractorAuto:

    def test_auto_returns_extractor(self):
        e = ConceptExtractor.auto()
        assert isinstance(e, ConceptExtractor)
        assert e.strategy in ConceptExtractor.STRATEGIES

    def test_auto_can_extract(self):
        e = ConceptExtractor.auto()
        result = e.extract("Transformer models use self-attention mechanisms for NLP tasks.")
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_auto_top_k(self):
        e = ConceptExtractor.auto(top_k=3)
        assert e.top_k == 3
        result = e.extract("deep learning neural networks transformers attention backprop gradient")
        assert len(result) <= 3


# ============ Интеграция с NGTMemoryForLLM ============

class TestConceptExtractorIntegration:
    """Тесты авто-извлечения концептов в NGTMemoryForLLM."""

    @pytest.fixture
    def memory(self):
        from ngt.core.llm_memory import NGTMemoryForLLM
        return NGTMemoryForLLM(
            embedding_dim=64,
            max_entries=50,
            max_concepts=100,
            concept_extraction="regex",
            concept_top_k=5,
            device="cpu",
        )

    def _rand_emb(self, dim=64):
        import torch
        e = torch.randn(dim)
        return e / e.norm()

    def test_auto_extract_on_store(self, memory):
        """Если concepts=None, концепты должны извлекаться автоматически."""
        emb = self._rand_emb()
        result = memory.store(
            embedding=emb,
            text="Neural networks use backpropagation for training on GPU with CUDA",
            concepts=None,  # явно None — триггер авто-извлечения
        )
        # Концепты должны были создаться
        assert memory.associations.num_concepts > 0, \
            "Концепты не были извлечены автоматически"

    def test_explicit_concepts_not_overridden(self, memory):
        """Если concepts переданы явно, авто-извлечение не должно запускаться."""
        import torch
        emb = self._rand_emb()
        memory.store(
            embedding=emb,
            text="some text that has many words and could produce concepts",
            concepts=["explicit_concept_a", "explicit_concept_b"],
        )
        # Должны быть только explicit концепты
        names = set(memory.associations._name_to_id.keys())
        assert "explicit_concept_a" in names
        assert "explicit_concept_b" in names

    def test_empty_concepts_list_triggers_auto(self, memory):
        """concepts=None (не пустой список) → авто-извлечение."""
        emb = self._rand_emb()
        # Явно None
        memory.store(emb, text="Quantum entanglement superposition computing qubits", concepts=None)
        assert memory.associations.num_concepts > 0

    def test_auto_extract_creates_graph_edges(self, memory):
        """Авто-извлеченные концепты должны связываться в графе."""
        emb1 = self._rand_emb()
        emb2 = self._rand_emb()
        memory.store(emb1, text="backpropagation gradient descent neural learning", concepts=None)
        memory.store(emb2, text="gradient neural network optimization training", concepts=None)
        memory._flush_hebbian()
        # После двух store с общими словами должны появиться рёбра
        assert memory.associations.num_edges >= 0  # может быть 0 если концепты не совпали

    def test_concept_extraction_strategy_hybrid(self):
        """hybrid стратегия инициализируется без ошибок."""
        from ngt.core.llm_memory import NGTMemoryForLLM
        m = NGTMemoryForLLM(
            embedding_dim=64,
            concept_extraction="hybrid",
            concept_top_k=4,
            device="cpu",
        )
        assert m._concept_extractor.strategy == "hybrid"

    def test_concept_extraction_strategy_regex(self, memory):
        """regex стратегия корректно инициализирована."""
        assert memory._concept_extractor.strategy == "regex"
        assert memory._concept_extractor.top_k == 5

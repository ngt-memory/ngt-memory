"""
ConceptExtractor — семантическое извлечение концептов из текста для NGT Memory.

Три стратегии (в порядке качества):
1. **ner**    — Named Entity Recognition через transformers (dslim/bert-base-NER)
               Извлекает: PERSON, ORG, LOC, MISC + именные группы
2. **tfidf**  — TF-IDF ключевые слова (sklearn), требует корпус для fit
3. **regex**  — Rule-based без зависимостей: noun phrases, CamelCase, аббревиатуры

Выбор стратегии:
    extractor = ConceptExtractor(strategy="ner")   # лучший качество
    extractor = ConceptExtractor(strategy="tfidf") # быстрее, нужен corpus fit
    extractor = ConceptExtractor(strategy="regex") # всегда доступен, нет требований

Использование:
    extractor = ConceptExtractor.auto()  # выбирает лучшую доступную
    concepts = extractor.extract("Neural networks learn from data via backprop", top_k=6)
    # → ["neural_networks", "backprop", "data"]
    
    # Batch
    concepts_list = extractor.extract_batch(texts, top_k=6)
"""

import re
import string
from typing import List, Optional, Dict, Tuple
from collections import Counter


# ============ Strategy: Regex (zero-dependency) ============

# Стоп-слова для фильтрации
_STOPWORDS = frozenset({
    "the", "this", "that", "these", "those", "with", "from", "have", "been",
    "will", "would", "could", "should", "their", "there", "where", "when",
    "what", "which", "who", "whom", "also", "more", "some", "such", "into",
    "than", "then", "they", "them", "were", "your", "about", "other", "after",
    "before", "because", "between", "through", "during", "each", "both",
    "very", "just", "over", "under", "only", "same", "even", "most",
    "while", "since", "though", "whether", "however", "therefore",
    "moreover", "furthermore", "additionally", "consequently",
})

# Паттерны для regex стратегии
_RE_CAMEL    = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')        # CamelCase
_RE_ACRONYM  = re.compile(r'\b[A-Z]{2,6}\b')                           # ACRONYM
_RE_NOUN_PHRASE = re.compile(r'\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b')  # Title Case noun phrase
_RE_TECH     = re.compile(r'\b[a-z]+-?[a-z]*(?:\.[a-z]+)+\b')        # tech.terms, sub.domain
_RE_HYPHEN   = re.compile(r'\b[a-z]{3,}-[a-z]{3,}\b')                # compound-words
_RE_WORD     = re.compile(r'\b[a-zA-Z]{4,}\b')                        # ordinary words ≥4 chars


def _normalize(token: str) -> str:
    """Нормализует токен: lowercase, убирает пунктуацию по краям."""
    return token.strip(string.punctuation).lower().replace(" ", "_")


def _extract_regex(text: str, top_k: int = 8) -> List[str]:
    """Rule-based извлечение концептов без внешних зависимостей."""
    concepts: Dict[str, int] = {}

    # 1. CamelCase и аббревиатуры (высокий приоритет)
    for m in _RE_CAMEL.finditer(text):
        w = _normalize(m.group())
        if len(w) >= 3:
            concepts[w] = concepts.get(w, 0) + 3

    for m in _RE_ACRONYM.finditer(text):
        w = _normalize(m.group())
        if len(w) >= 2 and w not in {"I", "A", "THE", "AND", "OR", "IN", "OF"}:
            concepts[w] = concepts.get(w, 0) + 3

    # 2. Title Case noun phrases
    for m in _RE_NOUN_PHRASE.finditer(text):
        w = _normalize(m.group())
        if len(w) >= 4 and w not in _STOPWORDS:
            concepts[w] = concepts.get(w, 0) + 2

    # 3. Hyphenated compounds
    for m in _RE_HYPHEN.finditer(text):
        w = _normalize(m.group())
        if w not in _STOPWORDS:
            concepts[w] = concepts.get(w, 0) + 2

    # 4. Обычные слова по частоте
    words = _RE_WORD.findall(text.lower())
    word_freq = Counter(w for w in words if w not in _STOPWORDS and len(w) >= 4)
    for w, freq in word_freq.items():
        concepts[w] = concepts.get(w, 0) + freq

    # Сортируем по весу
    sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1])
    return [c for c, _ in sorted_concepts[:top_k] if len(c) >= 3]


# ============ Strategy: NER (transformers) ============

_NER_PIPELINE = None
_NER_MODEL    = "dslim/bert-base-NER"

# Маппинг NER-тегов → более читаемые категории
_ENTITY_PRIORITY = {
    "B-ORG": 4, "I-ORG": 4,
    "B-PER": 3, "I-PER": 3,
    "B-LOC": 3, "I-LOC": 3,
    "B-MISC": 2, "I-MISC": 2,
}


def _load_ner_pipeline(model: str = _NER_MODEL):
    """Ленивая загрузка NER pipeline."""
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        from transformers import pipeline
        _NER_PIPELINE = pipeline(
            "ner",
            model=model,
            aggregation_strategy="simple",
            device=-1,  # CPU
        )
    return _NER_PIPELINE


def _extract_ner(text: str, top_k: int = 8, model: str = _NER_MODEL) -> List[str]:
    """
    NER-based извлечение через transformers.
    Возвращает именованные сущности + noun phrases.
    """
    pipe = _load_ner_pipeline(model)

    # NER сущности
    try:
        entities = pipe(text[:512])  # ограничиваем для bert-base
    except Exception:
        return _extract_regex(text, top_k)

    concepts: Dict[str, float] = {}

    for ent in entities:
        word  = _normalize(ent.get("word", ""))
        score = ent.get("score", 0.5)
        prio  = _ENTITY_PRIORITY.get(ent.get("entity_group", ""), 1)
        if len(word) >= 2 and word not in _STOPWORDS and not word.startswith("##"):
            concepts[word] = concepts.get(word, 0.0) + score * prio

    # Дополняем regex-концептами с низким весом (чтобы покрыть технические термины)
    regex_fallback = _extract_regex(text, top_k=top_k * 2)
    for i, c in enumerate(regex_fallback):
        if c not in concepts:
            concepts[c] = 0.5 / (i + 1)  # убывающий вес

    sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1])
    return [c for c, _ in sorted_concepts[:top_k]]


# ============ Strategy: TF-IDF ============

_TFIDF_VECTORIZER = None
_TFIDF_FEATURE_NAMES = None


def fit_tfidf(corpus: List[str], max_features: int = 3000):
    """
    Обучает TF-IDF на корпусе. Нужно вызвать до extract_tfidf.
    При использовании в NGTMemoryForLLM вызывается автоматически
    на первых N документах.
    """
    global _TFIDF_VECTORIZER, _TFIDF_FEATURE_NAMES
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2),
            token_pattern=r"[a-zA-Z]{3,}",
        )
        vect.fit(corpus)
        _TFIDF_VECTORIZER = vect
        _TFIDF_FEATURE_NAMES = vect.get_feature_names_out()
    except ImportError:
        pass


def _extract_tfidf(text: str, top_k: int = 8) -> List[str]:
    """TF-IDF извлечение (нужен предварительный fit_tfidf)."""
    if _TFIDF_VECTORIZER is None:
        return _extract_regex(text, top_k)

    import numpy as np
    vec = _TFIDF_VECTORIZER.transform([text])
    row = vec.toarray().ravel()
    top_idx = row.argsort()[-top_k * 2:][::-1]
    concepts = [_TFIDF_FEATURE_NAMES[j] for j in top_idx if row[j] > 0]
    # Нормализуем
    normalized = []
    for c in concepts:
        nc = _normalize(c)
        if nc not in _STOPWORDS and len(nc) >= 3:
            normalized.append(nc)
    return normalized[:top_k]


# ============ Main ConceptExtractor class ============

class ConceptExtractor:
    """
    Единый интерфейс для извлечения концептов из текста.

    Стратегии:
        "ner"   — Named Entity Recognition (transformers, лучшее качество)
        "tfidf" — TF-IDF ключевые слова (sklearn, нужен fit)
        "regex" — Rule-based regex (без зависимостей, всегда доступен)
        "hybrid"— NER + regex (рекомендуется для продакшена)
    """

    STRATEGIES = ("ner", "tfidf", "regex", "hybrid")

    def __init__(
        self,
        strategy: str = "hybrid",
        ner_model: str = _NER_MODEL,
        top_k: int = 6,
        min_length: int = 3,
        deduplicate: bool = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy должна быть одной из {self.STRATEGIES}")
        self.strategy   = strategy
        self.ner_model  = ner_model
        self.top_k      = top_k
        self.min_length = min_length
        self.deduplicate = deduplicate
        self._available: Optional[bool] = None  # кэш проверки доступности NER

    # ── Проверка доступности ──────────────────────────────────────────

    def _ner_available(self) -> bool:
        if self._available is None:
            try:
                import transformers  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _tfidf_available(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Публичный API ─────────────────────────────────────────────────

    def extract(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """Извлекает концепты из одного текста."""
        k = top_k or self.top_k
        text = text[:1000]  # обрезаем для скорости

        if self.strategy == "ner":
            if self._ner_available():
                raw = _extract_ner(text, top_k=k, model=self.ner_model)
            else:
                raw = _extract_regex(text, top_k=k)

        elif self.strategy == "tfidf":
            raw = _extract_tfidf(text, top_k=k)

        elif self.strategy == "regex":
            raw = _extract_regex(text, top_k=k)

        elif self.strategy == "hybrid":
            # NER (если доступен) + regex для технических терминов
            if self._ner_available():
                raw = _extract_ner(text, top_k=k, model=self.ner_model)
            else:
                # regex + tfidf если есть
                raw = _extract_regex(text, top_k=k)
                if self._tfidf_available() and _TFIDF_VECTORIZER is not None:
                    tfidf_raw = _extract_tfidf(text, top_k=k)
                    # Объединяем без дублей
                    seen = set(raw)
                    for c in tfidf_raw:
                        if c not in seen and len(raw) < k * 2:
                            raw.append(c)
                            seen.add(c)
        else:
            raw = _extract_regex(text, top_k=k)

        return self._postprocess(raw, k)

    def extract_batch(self, texts: List[str], top_k: Optional[int] = None) -> List[List[str]]:
        """Извлекает концепты для списка текстов."""
        return [self.extract(t, top_k) for t in texts]

    def fit_tfidf_corpus(self, corpus: List[str], max_features: int = 3000):
        """Обучает TF-IDF на корпусе (нужно для strategy='tfidf' или 'hybrid')."""
        fit_tfidf(corpus, max_features)

    # ── Фабричный метод ───────────────────────────────────────────────

    @classmethod
    def auto(cls, top_k: int = 6) -> "ConceptExtractor":
        """
        Выбирает лучшую доступную стратегию:
        transformers → hybrid
        sklearn only → tfidf
        nothing      → regex
        """
        try:
            import transformers  # noqa: F401
            return cls(strategy="hybrid", top_k=top_k)
        except ImportError:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            return cls(strategy="tfidf", top_k=top_k)
        except ImportError:
            pass
        return cls(strategy="regex", top_k=top_k)

    # ── Постобработка ─────────────────────────────────────────────────

    def _postprocess(self, concepts: List[str], top_k: int) -> List[str]:
        result = []
        seen   = set()
        for c in concepts:
            c = c.strip().lower().replace(" ", "_")
            c = c.strip("_-.")
            if len(c) < self.min_length:
                continue
            if self.deduplicate and c in seen:
                continue
            seen.add(c)
            result.append(c)
            if len(result) >= top_k:
                break
        return result

    def __repr__(self) -> str:
        return f"ConceptExtractor(strategy={self.strategy!r}, top_k={self.top_k})"

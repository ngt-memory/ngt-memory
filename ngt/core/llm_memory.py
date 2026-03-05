"""
NGT Memory for LLM — Внешняя нейропластичная память для языковых моделей

NGT используется как внешняя адаптивная память для LLM (LLaMA, Mistral, GPT и др.),
давая им способности, которых нет у vector DB / RAG:

1. **Ассоциативный граф** — связи между концептами (Hebbian learning)
2. **Иерархическая память** — 4 уровня (sensory→working→episodic→semantic)
3. **Hopfield retrieval** — ассоциативный поиск по неполному запросу
4. **Dream consolidation** — офлайн-обобщение эпизодов в семантику
5. **Adaptive forgetting** — автоматическое забывание нерелевантного

Архитектура:
    LLM текст → embedding → NGTMemoryForLLM → релевантные воспоминания → контекст LLM

Использование:
    memory = NGTMemoryForLLM(embedding_dim=4096)
    
    # Сохранить
    memory.store(embedding, text="...", metadata={...})
    
    # Извлечь
    results = memory.retrieve(query_embedding, top_k=5)
    
    # Получить контекст для промпта
    context = memory.get_context(query_embedding, max_tokens=2048)
    
    # Консолидация (между сессиями)
    memory.consolidate()
    
    # Сохранить/загрузить на диск
    memory.save("memory.pt")
    memory = NGTMemoryForLLM.load("memory.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from collections import deque
from pathlib import Path
import json
import time
import math

from ngt.core.graph import DynamicGraph
from ngt.core.hebbian import HebbianPlasticityCore
from ngt.core.concept_extractor import ConceptExtractor
from ngt.core.hierarchical_memory import (
    HierarchicalMemory,
    SensoryBuffer,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
)


class ConceptNode:
    """
    Узел концепта в графе ассоциаций.
    
    Каждый концепт — это сущность (entity), тема (topic) или факт,
    извлечённый из текста. Хранит embedding и метаданные.
    """
    __slots__ = ("node_id", "name", "embedding", "metadata",
                 "created_at", "last_accessed", "access_count", "strength")
    
    def __init__(
        self,
        node_id: int,
        name: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ):
        self.node_id = node_id
        self.name = name
        self.embedding = embedding.detach().cpu()
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.strength = 1.0
    
    def touch(self) -> None:
        """Обновляет время последнего доступа."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "strength": self.strength,
        }


class AssociationGraph:
    """
    Граф ассоциаций между концептами.

    Лёгкая реализация без nn.Module overhead.
    Связи формируются через Hebbian co-occurrence (dict-based граф).

    Поддерживает:
    - Co-occurrence based edge creation
    - Hebbian strengthening повторяющихся ассоциаций
    - Graph walk для поиска связанных концептов
    - Decay неиспользуемых связей
    """

    def __init__(
        self,
        max_concepts: int = 10000,
        embedding_dim: int = 4096,
        hebbian_lr: float = 0.05,
        decay_rate: float = 0.001,
        edge_threshold: float = 0.01,
        max_edges_ratio: float = 0.01,
        device: str = "cpu",
    ):
        self.max_concepts = max_concepts
        self.embedding_dim = embedding_dim
        self.device = device
        self.hebbian_lr = hebbian_lr
        self.decay_rate = decay_rate
        self.edge_threshold = edge_threshold

        # Маппинг concept_name → node_id
        self._name_to_id: Dict[str, int] = {}
        self._id_to_concept: Dict[int, ConceptNode] = {}
        self._next_id = 0

        # Граф рёбер: (i, j) → weight  (i < j всегда)
        self._edges: Dict[Tuple[int, int], float] = {}
        # Adjacency list для быстрого walk: node_id → {neighbor_id: weight}
        self._adj: Dict[int, Dict[int, float]] = {}

        # Embedding matrix для быстрого cosine search
        self._embeddings = torch.zeros(max_concepts, embedding_dim)
        self._active_ids: List[int] = []  # список активных node_id
        self._emb_dirty = True
        self._emb_matrix: Optional[torch.Tensor] = None  # [M, D] нормализованных

    def _rebuild_emb_matrix(self) -> None:
        if not self._active_ids:
            self._emb_matrix = None
            self._emb_dirty = False
            return
        embs = self._embeddings[self._active_ids]  # [M, D]
        norms = embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._emb_matrix = embs / norms
        self._emb_dirty = False

    def add_concept(
        self,
        name: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Добавляет или обновляет концепт. Returns node_id."""
        if name in self._name_to_id:
            node_id = self._name_to_id[name]
            concept = self._id_to_concept[node_id]
            concept.access_count += 1  # touch() без time.time()
            # Лёгкое обновление embedding только каждые 10 посещений
            if concept.access_count % 10 == 0:
                emb_cpu = embedding.detach().cpu()
                concept.embedding = 0.9 * concept.embedding + 0.1 * emb_cpu
                self._embeddings[node_id] = concept.embedding
                self._emb_dirty = True
            return node_id

        if self._next_id >= self.max_concepts:
            node_id = self._evict_weakest()
        else:
            node_id = self._next_id
            self._next_id += 1

        concept = ConceptNode(node_id, name, embedding, metadata)
        self._name_to_id[name] = node_id
        self._id_to_concept[node_id] = concept
        self._embeddings[node_id] = concept.embedding
        self._active_ids.append(node_id)
        self._emb_dirty = True

        return node_id

    def record_co_occurrence(
        self,
        concept_ids: List[int],
        strength: float = 1.0,
    ) -> int:
        """Hebbian: усиливаем/создаём рёбра для всех пар concept_ids."""
        if len(concept_ids) < 2:
            return 0

        created = 0
        valid = [cid for cid in concept_ids if cid in self._id_to_concept]
        lr = self.hebbian_lr * strength

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                key = (min(a, b), max(a, b))
                old = self._edges.get(key, 0.0)
                new_w = old + lr * (1.0 - old)  # Hebbian: Δw = lr*(1-w)
                if new_w < self.edge_threshold:
                    new_w = self.edge_threshold
                self._edges[key] = new_w
                # Обновляем adjacency list
                if a not in self._adj:
                    self._adj[a] = {}
                if b not in self._adj:
                    self._adj[b] = {}
                if key not in self._edges or old == 0.0:
                    created += 1
                self._adj[a][b] = new_w
                self._adj[b][a] = new_w

        return created

    def get_associated(
        self,
        concept_id: int,
        top_k: int = 10,
        min_weight: float = 0.01,
    ) -> List[Tuple[int, float]]:
        """Возвращает связанные концепты. Returns [(node_id, weight), ...]"""
        neighbors = self._adj.get(concept_id, {})
        results = [(nid, w) for nid, w in neighbors.items() if w >= min_weight]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def get_associated_multi_hop(
        self,
        concept_id: int,
        hops: int = 2,
        top_k: int = 10,
        min_weight: float = 0.01,
    ) -> List[Tuple[int, float, int]]:
        """Multi-hop graph walk. Returns [(node_id, weight, distance), ...]"""
        visited: Dict[int, Tuple[float, int]] = {concept_id: (1.0, 0)}
        frontier = [(concept_id, 1.0)]

        for hop in range(1, hops + 1):
            next_frontier = []
            for node_id, parent_weight in frontier:
                for nid, w in self._adj.get(node_id, {}).items():
                    if w < min_weight:
                        continue
                    accumulated = parent_weight * w
                    if nid not in visited or visited[nid][0] < accumulated:
                        visited[nid] = (accumulated, hop)
                        next_frontier.append((nid, accumulated))
            frontier = next_frontier

        del visited[concept_id]
        results = [(nid, weight, dist) for nid, (weight, dist) in visited.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def find_similar_concepts(
        self,
        embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Поиск ближайших концептов по cosine similarity."""
        if not self._active_ids:
            return []

        if self._emb_dirty:
            self._rebuild_emb_matrix()
        if self._emb_matrix is None:
            return []

        query = embedding.detach().cpu().view(-1)
        if query.norm() > 0:
            query = query / query.norm()

        sims = torch.mv(self._emb_matrix, query)  # [M]
        k = min(top_k, sims.shape[0])
        top_sims, top_local = torch.topk(sims, k)

        results = []
        for sim, local_i in zip(top_sims, top_local):
            global_id = self._active_ids[local_i.item()]
            results.append((global_id, sim.item()))

        return results

    def apply_decay(self, rate: Optional[float] = None) -> int:
        """Применяет затухание к слабым связям."""
        r = rate if rate is not None else self.decay_rate
        to_remove = []
        for key, w in self._edges.items():
            new_w = w * (1.0 - r)
            if new_w < self.edge_threshold:
                to_remove.append(key)
            else:
                self._edges[key] = new_w
                a, b = key
                if a in self._adj and b in self._adj[a]:
                    self._adj[a][b] = new_w
                if b in self._adj and a in self._adj[b]:
                    self._adj[b][a] = new_w
        for key in to_remove:
            del self._edges[key]
            a, b = key
            if a in self._adj:
                self._adj[a].pop(b, None)
            if b in self._adj:
                self._adj[b].pop(a, None)
        return len(to_remove)

    def _evict_weakest(self) -> int:
        """Замещает самый слабый/старый концепт."""
        min_score = float("inf")
        min_id = 0
        now = time.time()

        for node_id, concept in self._id_to_concept.items():
            age = now - concept.last_accessed + 1.0
            score = concept.strength * concept.access_count / math.log1p(age)
            if score < min_score:
                min_score = score
                min_id = node_id

        if min_id in self._id_to_concept:
            old_name = self._id_to_concept[min_id].name
            if old_name in self._name_to_id:
                del self._name_to_id[old_name]
            del self._id_to_concept[min_id]
            if min_id in self._active_ids:
                self._active_ids.remove(min_id)
            # Убираем рёбра этого узла
            for neighbor in list(self._adj.get(min_id, {}).keys()):
                key = (min(min_id, neighbor), max(min_id, neighbor))
                self._edges.pop(key, None)
                if neighbor in self._adj:
                    self._adj[neighbor].pop(min_id, None)
            self._adj.pop(min_id, None)
            self._emb_dirty = True

        return min_id

    def get_concept(self, node_id: int) -> Optional[ConceptNode]:
        return self._id_to_concept.get(node_id)

    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        node_id = self._name_to_id.get(name)
        if node_id is not None:
            return self._id_to_concept.get(node_id)
        return None

    @property
    def num_concepts(self) -> int:
        return len(self._id_to_concept)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def get_statistics(self) -> Dict:
        return {
            "num_concepts": self.num_concepts,
            "num_edges": self.num_edges,
            "max_concepts": self.max_concepts,
        }


class MemoryEntry:
    """
    Запись в памяти LLM — один фрагмент текста с embedding.
    """
    __slots__ = ("entry_id", "text", "embedding", "metadata",
                 "timestamp", "importance", "access_count",
                 "concept_ids")
    
    def __init__(
        self,
        entry_id: int,
        text: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
        importance: float = 1.0,
        concept_ids: Optional[List[int]] = None,
    ):
        self.entry_id = entry_id
        self.text = text
        self.embedding = embedding.detach().cpu()
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.importance = importance
        self.access_count = 0
        self.concept_ids = concept_ids or []
    
    def touch(self) -> None:
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "text": self.text,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "concept_ids": self.concept_ids,
        }


class NGTMemoryForLLM:
    """
    Внешняя нейропластичная память для LLM.
    
    Объединяет:
    - HierarchicalMemory (4-уровневая иерархия)
    - AssociationGraph (граф ассоциаций концептов)
    - Hopfield retrieval (ассоциативный поиск)
    - Dream consolidation (офлайн-обобщение)
    
    Преимущества перед RAG / vector DB:
    - Ассоциативные связи между концептами (graph walk)
    - Иерархия памяти (sensory→working→episodic→semantic)
    - Адаптивное забывание (decay по уровням)
    - Консолидация знаний (dream phase)
    - Hopfield-уточнение запросов
    
    Args:
        embedding_dim: размерность embedding LLM (4096 для LLaMA-7B)
        max_entries: максимум записей в episodic памяти
        max_concepts: максимум концептов в графе ассоциаций
        working_capacity: ёмкость рабочей памяти (активный контекст)
        max_prototypes: максимум семантических прототипов
        hebbian_lr: скорость Хеббовского обучения связей
        hopfield_beta: температура Hopfield retrieval
        consolidation_interval: автоматическая консолидация каждые N store()
        device: устройство (cpu/cuda)
    """
    
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_entries: int = 10000,
        max_concepts: int = 5000,
        working_capacity: int = 64,
        max_prototypes: int = 500,
        hebbian_lr: float = 0.05,
        hopfield_beta: float = 8.0,
        consolidation_interval: int = 100,
        decay_rate: float = 0.001,
        concept_extraction: str = "regex",
        concept_top_k: int = 6,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.hopfield_beta = hopfield_beta
        self.consolidation_interval = consolidation_interval
        self.device = device
        
        # 1. Иерархическая память (4 уровня)
        self.hierarchy = HierarchicalMemory(
            pattern_dim=embedding_dim,
            sensory_capacity=128,
            working_capacity=working_capacity,
            episodic_capacity=max_entries,
            max_prototypes=max_prototypes,
            promotion_threshold=0.3,
            consolidation_interval=consolidation_interval,
        )
        
        # 2. Граф ассоциаций концептов
        self.associations = AssociationGraph(
            max_concepts=max_concepts,
            embedding_dim=embedding_dim,
            hebbian_lr=hebbian_lr,
            decay_rate=decay_rate,
            device=device,
        )
        
        # 3. Текстовые записи (entry_id → MemoryEntry)
        self._entries: Dict[int, MemoryEntry] = {}
        self._next_entry_id = 0
        
        # 3b. Embedding index для быстрого retrieve (batch cosine search)
        self._entry_embeddings: Optional[torch.Tensor] = None  # [N, D] — строится лениво
        self._entry_id_list: List[int] = []  # параллельный список entry_id
        self._emb_buffer: List[torch.Tensor] = []  # накапливаем O(1), stack при retrieve
        self._index_dirty = True

        # 3c. Инвертированный индекс концептов: concept_id -> [entry_id, ...]
        # Убирает O(N) loop при graph-based retrieval
        self._concept_to_entries: Dict[int, List[int]] = {}

        # 3d. Lazy Hebbian: буферизуем co-occurrence, flush батчами
        self._pending_cooccurrences: List[List[int]] = []
        self._hebbian_flush_interval: int = 50

        # 3e. Semantic concept extractor (авто-извлечение если concepts=None)
        self._concept_extractor = ConceptExtractor(
            strategy=concept_extraction,
            top_k=concept_top_k,
        )
        
        # 4. Session tracking
        self._session_id = 0
        self._session_entries: List[int] = []  # entry_ids текущей сессии
        
        # 5. Статистика
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "total_consolidated": 0,
            "total_concepts_created": 0,
            "total_associations": 0,
            "sessions": 0,
        }
    
    # ===== STORE =====
    
    def store(
        self,
        embedding: torch.Tensor,
        text: str = "",
        concepts: Optional[List[str]] = None,
        concept_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict] = None,
        importance: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> Dict:
        """
        Сохраняет фрагмент текста с embedding в память.
        
        Args:
            embedding: embedding текста от LLM [embedding_dim]
            text: исходный текст
            concepts: список имён концептов, упомянутых в тексте
            concept_embeddings: {concept_name: embedding} для концептов
            metadata: произвольные метаданные (user, timestamp, source...)
            importance: явная важность (если None — оценивается автоматически)
            domain: домен/тема (для семантической организации)
            
        Returns:
            store_stats: статистика сохранения
        """
        embedding = embedding.detach().view(-1)
        if embedding.shape[0] != self.embedding_dim:
            # Pad или truncate
            if embedding.shape[0] < self.embedding_dim:
                embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[0]))
            else:
                embedding = embedding[:self.embedding_dim]
        
        store_result = {
            "entry_id": -1,
            "concepts_added": 0,
            "associations_created": 0,
            "hierarchy_level": "entry",
        }

        meta = metadata or {}
        meta["domain"] = domain or meta.get("domain", "general")
        meta["session_id"] = self._session_id
        meta["text_preview"] = text[:200] if text else ""

        # importance оценка (без hierarchy.store для скорости)
        if importance is None:
            importance = 1.0
        
        # 2. Авто-извлечение концептов если не переданы явно
        if concepts is None and text:
            concepts = self._concept_extractor.extract(text)

        # 2. Обрабатываем концепты
        concept_ids = []
        if concepts:
            for cname in concepts:
                cemb = embedding  # fallback: используем embedding всего текста
                if concept_embeddings and cname in concept_embeddings:
                    cemb = concept_embeddings[cname]

                node_id = self.associations.add_concept(
                    name=cname,
                    embedding=cemb,
                    metadata={"domain": meta["domain"]},
                )
                concept_ids.append(node_id)
                store_result["concepts_added"] += 1

            # 3. Lazy Hebbian: буферизуем co-occurrence, flush батчами
            if len(concept_ids) >= 2:
                self._pending_cooccurrences.append(concept_ids)
                store_result["associations_created"] = len(concept_ids) * (len(concept_ids) - 1) // 2

            # Flush если буфер заполнен
            if len(self._pending_cooccurrences) >= self._hebbian_flush_interval:
                self._flush_hebbian()

        # 4. Создаём MemoryEntry
        entry_id = self._next_entry_id
        self._next_entry_id += 1

        entry = MemoryEntry(
            entry_id=entry_id,
            text=text,
            embedding=embedding,
            metadata=meta,
            importance=importance,
            concept_ids=concept_ids,
        )
        self._entries[entry_id] = entry
        self._session_entries.append(entry_id)

        # Инкрементальный append в embedding index (O(1) — без копирования матрицы)
        emb_norm = embedding.view(-1)[:self.embedding_dim].clone()
        n = emb_norm.norm()
        if n > 0:
            emb_norm = emb_norm / n
        self._emb_buffer.append(emb_norm)
        self._entry_id_list.append(entry_id)
        self._index_dirty = True  # нужно stack при следующем retrieve

        # Обновляем инвертированный индекс концептов
        for cid in concept_ids:
            if cid not in self._concept_to_entries:
                self._concept_to_entries[cid] = []
            self._concept_to_entries[cid].append(entry_id)

        store_result["entry_id"] = entry_id

        # Evict старых записей если превышен лимит
        if len(self._entries) > self.max_entries:
            self._evict_old_entries()
        
        self.stats["total_stored"] += 1
        self.stats["total_concepts_created"] = self.associations.num_concepts
        
        return store_result
    
    # ===== RETRIEVE =====
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        use_hopfield: bool = False,
        use_graph: bool = True,
        graph_hops: int = 2,
        recency_weight: float = 0.1,
        domain: Optional[str] = None,
    ) -> List[Dict]:
        """
        Извлекает релевантные воспоминания по запросу.
        
        Комбинирует три стратегии:
        1. **Hierarchical retrieval** — поиск по 4 уровням памяти
        2. **Hopfield refinement** — уточнение запроса через ассоциативную сеть
        3. **Graph walk** — поиск связанных концептов через граф
        
        Args:
            query_embedding: embedding запроса [embedding_dim]
            top_k: количество результатов
            use_hopfield: использовать Hopfield уточнение
            use_graph: использовать граф ассоциаций
            graph_hops: глубина graph walk
            recency_weight: вес давности (0 = только relevance, 1 = сильный recency bias)
            domain: фильтр по домену
            
        Returns:
            Список результатов: [{text, similarity, level, concepts, metadata}, ...]
        """
        query = query_embedding.detach().view(-1)
        if query.shape[0] != self.embedding_dim:
            if query.shape[0] < self.embedding_dim:
                query = F.pad(query, (0, self.embedding_dim - query.shape[0]))
            else:
                query = query[:self.embedding_dim]
        
        # 1. Hopfield refinement — уточняем запрос через ассоциативную сеть
        if use_hopfield and len(self._entries) > 0:
            query = self._hopfield_refine(query)
        
        # 2. Прямой entry search (основной путь — не зависит от hierarchy decay)
        direct_entries = self._direct_entry_search(query, top_k=top_k * 3, domain=domain)
        
        # 3. Graph-augmented retrieval — ищем через граф ассоциаций
        #    graph_entry_scores: entry_id → concept_match_score
        graph_entry_scores: Dict[int, float] = {}
        if use_graph and self.associations.num_concepts > 0:
            # 3a. Находим ближайшие концепты к query с их similarity
            similar_concepts = self.associations.find_similar_concepts(query, top_k=5)
            concept_sim_map = {cid: sim for cid, sim in similar_concepts}

            # 3b. O(1) lookup: entries содержащие эти концепты через инвертированный индекс
            # Ограничиваем до top_k*4 entries на концепт для предотвращения O(N) при dense графе
            _max_per_concept = top_k * 4
            for cid, csim in concept_sim_map.items():
                entries_for_cid = self._concept_to_entries.get(cid, [])
                # Берём последние (наиболее свежие) записи
                sample = entries_for_cid[-_max_per_concept:]
                for eid in sample:
                    cur = graph_entry_scores.get(eid, 0.0)
                    if csim > cur:
                        graph_entry_scores[eid] = csim

            # 3c. Multi-hop graph walk — расширяем через ассоциации
            for cid, sim in similar_concepts:
                related = self.associations.get_associated_multi_hop(
                    cid, hops=graph_hops, top_k=5
                )
                for rid, weight, dist in related:
                    hop_score = sim * weight * 0.5
                    entries_for_rid = self._concept_to_entries.get(rid, [])
                    sample = entries_for_rid[-_max_per_concept:]
                    for eid in sample:
                        cur = graph_entry_scores.get(eid, 0.0)
                        if hop_score > cur:
                            graph_entry_scores[eid] = hop_score
        
        # 4. Собираем и ранжируем
        results = self._rank_direct_results(
            query, direct_entries, graph_entry_scores,
            top_k=top_k, recency_weight=recency_weight
        )
        
        self.stats["total_retrieved"] += 1
        
        return results
    
    def _flush_hebbian(self) -> int:
        """Применяет накопленные co-occurrence батчем."""
        if not self._pending_cooccurrences:
            return 0
        total = 0
        for concept_ids in self._pending_cooccurrences:
            n = self.associations.record_co_occurrence(concept_ids)
            total += n
            self.stats["total_associations"] += n
        self._pending_cooccurrences = []
        return total

    def _direct_entry_search(
        self,
        query: torch.Tensor,
        top_k: int = 10,
        domain: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Прямой cosine search по entry embedding index.
        Не зависит от hierarchy episodic strengths/decay.
        
        Returns:
            Список (entry_id, similarity) отсортированный по similarity
        """
        if self._index_dirty:
            self._rebuild_entry_index()
        if self._entry_embeddings is None:
            return []

        q = query.view(-1)[:self.embedding_dim]
        if q.norm() > 0:
            q = q / q.norm()
        
        sims = torch.mv(self._entry_embeddings, q)  # [N]
        k = min(top_k, sims.shape[0])
        top_sims, top_idx = torch.topk(sims, k)
        
        results = []
        for sim, idx in zip(top_sims, top_idx):
            eid = self._entry_id_list[idx.item()]
            # Domain filter
            if domain is not None:
                entry = self._entries.get(eid)
                if entry and entry.metadata.get("domain") != domain:
                    continue
            results.append((eid, sim.item()))
        
        return results
    
    def _rank_direct_results(
        self,
        query: torch.Tensor,
        direct_entries: List[Tuple[int, float]],
        graph_entry_scores: Dict[int, float],
        top_k: int,
        recency_weight: float,
    ) -> List[Dict]:
        """
        Ранжирует результаты из direct search + graph concept scores.
        
        Финальный score = embedding_sim * (1 + concept_match_score).
        Это позволяет graph-matched entries подниматься выше.
        """
        now = time.time()
        scored: List[Dict] = []
        
        seen_ids = set()
        
        for eid, sim in direct_entries:
            entry = self._entries.get(eid)
            if entry is None or eid in seen_ids:
                continue
            seen_ids.add(eid)
            
            age_hours = (now - entry.timestamp) / 3600
            recency_score = math.exp(-0.01 * age_hours)
            
            emb_score = (1.0 - recency_weight) * sim + recency_weight * recency_score
            
            # Concept boost: score * (1 + concept_match_score)
            source = "direct"
            concept_score = graph_entry_scores.get(eid, 0.0)
            if concept_score > 0:
                emb_score *= (1.0 + concept_score)
                source = "direct+graph"
            
            scored.append({
                "entry_id": eid,
                "text": entry.text,
                "similarity": round(sim, 4),
                "concept_score": round(concept_score, 4),
                "recency_score": round(recency_score, 4),
                "score": round(emb_score, 4),
                "level": "entry",
                "domain": entry.metadata.get("domain"),
                "concepts": entry.concept_ids,  # lazy: resolve только для top-k
                "metadata": entry.metadata,
                "source": source,
            })
        
        # Добавляем graph-only entries (не найденные в direct search)
        # Используем concept_score как proxy вместо дорогого torch.norm()
        for eid, concept_score in graph_entry_scores.items():
            if eid in seen_ids:
                continue
            entry = self._entries.get(eid)
            if entry is None:
                continue
            seen_ids.add(eid)

            age_hours = (now - entry.timestamp) / 3600
            recency_score = math.exp(-0.01 * age_hours)
            # Score базируется на concept similarity (без пересчёта embedding)
            emb_score = concept_score * (1.0 - recency_weight) + recency_weight * recency_score

            scored.append({
                "entry_id": eid,
                "text": entry.text,
                "similarity": round(concept_score, 4),
                "concept_score": round(concept_score, 4),
                "recency_score": round(recency_score, 4),
                "score": round(emb_score, 4),
                "level": "graph",
                "domain": entry.metadata.get("domain"),
                "concepts": entry.concept_ids,  # lazy: resolve только для top-k
                "metadata": entry.metadata,
                "source": "graph",
            })

        scored.sort(key=lambda x: -x["score"])
        top = scored[:top_k]
        # Resolve concept names только для финальных top-k
        for r in top:
            if r["concepts"] and isinstance(r["concepts"][0], int):
                r["concepts"] = self._resolve_concept_names(r["concepts"])
        return top
    
    def _hopfield_refine(
        self,
        query: torch.Tensor,
        iterations: int = 1,
    ) -> torch.Tensor:
        """
        Hopfield-like уточнение запроса через entry embedding index.
        
        Запрос итеративно «притягивается» к ближайшим воспоминаниям,
        улучшая recall для неполных или неточных запросов.
        
        Использует _entry_embeddings (не затухает) вместо episodic._patterns
        (которые могут деградировать после consolidation/decay).
        """
        if self._index_dirty:
            self._rebuild_entry_index()

        if self._entry_embeddings is None or self._entry_embeddings.shape[0] == 0:
            return query
        
        x = query.clone()
        memory = self._entry_embeddings  # [N, D] — нормализованные
        
        if x.norm() > 0:
            x = x / x.norm()
        
        # Используем мягкий beta (минимум с hopfield_beta) для лёгкого уточнения
        soft_beta = min(self.hopfield_beta, 4.0)
        
        for _ in range(iterations):
            sims = torch.mv(memory, x)  # [N]
            attention = torch.softmax(soft_beta * sims, dim=0)  # [N]
            x_new = torch.mv(memory.T, attention)  # [D]
            
            if x_new.norm() > 0:
                x_new = x_new / x_new.norm()
            
            if torch.dot(x, x_new).item() > 0.9999:
                break
            x = x_new
        
        return x
    
    def _rank_results(
        self,
        query: torch.Tensor,
        hier_results: List[Dict],
        graph_entries: List[MemoryEntry],
        top_k: int,
        recency_weight: float,
    ) -> List[Dict]:
        """Ранжирует и дедуплицирует результаты из разных источников."""
        
        query_norm = query / query.norm().clamp(min=1e-8)
        now = time.time()
        
        scored_results: Dict[int, Dict] = {}
        
        # Результаты из иерархической памяти
        for hr in hier_results:
            pattern = hr.get("pattern")
            if pattern is None:
                continue
            
            # Ищем соответствующую MemoryEntry
            p_norm = pattern / pattern.norm().clamp(min=1e-8)
            sim = torch.dot(query_norm, p_norm.view(-1)[:self.embedding_dim]).item()
            
            # Ищем entry с ближайшим embedding
            best_entry = self._find_entry_by_pattern(pattern)
            
            entry_id = best_entry.entry_id if best_entry else -1
            text = best_entry.text if best_entry else ""
            meta = best_entry.metadata if best_entry else hr
            timestamp = best_entry.timestamp if best_entry else now
            concepts = best_entry.concept_ids if best_entry else []
            
            # Recency score: экспоненциальное затухание
            age_hours = (now - timestamp) / 3600
            recency_score = math.exp(-0.01 * age_hours)
            
            # Финальный score
            final_score = (1.0 - recency_weight) * sim + recency_weight * recency_score
            
            if entry_id not in scored_results or scored_results[entry_id]["score"] < final_score:
                scored_results[entry_id] = {
                    "entry_id": entry_id,
                    "text": text,
                    "similarity": round(sim, 4),
                    "recency_score": round(recency_score, 4),
                    "score": round(final_score, 4),
                    "level": hr.get("level", "unknown"),
                    "domain": hr.get("domain"),
                    "concepts": self._resolve_concept_names(concepts),
                    "metadata": meta,
                    "source": "hierarchy",
                }
        
        # Результаты из graph walk
        for entry in graph_entries:
            if entry.entry_id in scored_results:
                # Бустим score для записей найденных через граф
                scored_results[entry.entry_id]["score"] *= 1.2
                scored_results[entry.entry_id]["source"] = "hierarchy+graph"
                continue
            
            emb = entry.embedding.view(-1)[:self.embedding_dim]
            emb_norm = emb / emb.norm().clamp(min=1e-8)
            sim = torch.dot(query_norm, emb_norm).item()
            
            age_hours = (now - entry.timestamp) / 3600
            recency_score = math.exp(-0.01 * age_hours)
            final_score = (1.0 - recency_weight) * sim + recency_weight * recency_score
            
            scored_results[entry.entry_id] = {
                "entry_id": entry.entry_id,
                "text": entry.text,
                "similarity": round(sim, 4),
                "recency_score": round(recency_score, 4),
                "score": round(final_score, 4),
                "level": "graph",
                "domain": entry.metadata.get("domain"),
                "concepts": self._resolve_concept_names(entry.concept_ids),
                "metadata": entry.metadata,
                "source": "graph",
            }
        
        # Сортируем и возвращаем top-k
        ranked = sorted(scored_results.values(), key=lambda x: -x["score"])
        return ranked[:top_k]
    
    def _rebuild_entry_index(self) -> None:
        """Перестраивает embedding index для быстрого cosine search."""
        if not self._index_dirty:
            return
        if not self._entries:
            self._entry_embeddings = None
            self._entry_id_list = []
            self._emb_buffer = []
            self._index_dirty = False
            return

        # Если буфер синхронизирован с _entry_id_list — просто stack буфера
        if self._emb_buffer and len(self._emb_buffer) == len(self._entry_id_list):
            self._entry_embeddings = torch.stack(self._emb_buffer)  # [N, D]
            self._index_dirty = False
            return

        # Полный rebuild из entries (после evict/load — буфер мог рассинхронизироваться)
        ids = []
        embs = []
        for eid in self._entry_id_list:
            entry = self._entries.get(eid)
            if entry is None:
                continue
            emb = entry.embedding.view(-1)[:self.embedding_dim]
            n = emb.norm()
            embs.append(emb / n if n > 0 else emb)
            ids.append(eid)

        if embs:
            self._entry_embeddings = torch.stack(embs)  # [N, D]
            self._entry_id_list = ids
            self._emb_buffer = list(self._entry_embeddings)
        else:
            self._entry_embeddings = None
            self._entry_id_list = []
            self._emb_buffer = []
        self._index_dirty = False
    
    def _find_entry_by_pattern(self, pattern: torch.Tensor) -> Optional[MemoryEntry]:
        """Находит MemoryEntry по ближайшему embedding (быстрый batch cosine search)."""
        if not self._entries:
            return None
        
        self._rebuild_entry_index()
        if self._entry_embeddings is None:
            return None
        
        pattern_flat = pattern.view(-1)[:self.embedding_dim]
        if pattern_flat.norm() == 0:
            return None
        query = pattern_flat / pattern_flat.norm()
        
        sims = torch.mv(self._entry_embeddings, query)  # [N]
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()
        
        if best_sim > 0.5:
            eid = self._entry_id_list[best_idx]
            return self._entries.get(eid)
        return None
    
    def _resolve_concept_names(self, concept_ids: List[int]) -> List[str]:
        """Превращает concept_ids в имена."""
        names = []
        for cid in concept_ids:
            concept = self.associations.get_concept(cid)
            if concept:
                names.append(concept.name)
        return names
    
    # ===== GET CONTEXT (для инъекции в промпт LLM) =====
    
    def get_context(
        self,
        query_embedding: torch.Tensor,
        max_tokens: int = 2048,
        top_k: int = 10,
        format: str = "markdown",
        include_concepts: bool = True,
        domain: Optional[str] = None,
    ) -> str:
        """
        Формирует текстовый контекст из релевантных воспоминаний
        для инъекции в промпт LLM.
        
        Args:
            query_embedding: embedding запроса
            max_tokens: приблизительный лимит токенов (~4 chars/token)
            top_k: максимум воспоминаний
            format: формат вывода ("markdown", "plain", "xml")
            include_concepts: включить связанные концепты
            domain: фильтр по домену
            
        Returns:
            Форматированный текст для инъекции в промпт
        """
        results = self.retrieve(
            query_embedding, top_k=top_k, domain=domain
        )
        
        if not results:
            return ""
        
        max_chars = max_tokens * 4  # грубая оценка
        
        if format == "xml":
            return self._format_xml(results, max_chars, include_concepts)
        elif format == "plain":
            return self._format_plain(results, max_chars, include_concepts)
        else:
            return self._format_markdown(results, max_chars, include_concepts)
    
    def _format_markdown(
        self, results: List[Dict], max_chars: int, include_concepts: bool
    ) -> str:
        lines = ["## Relevant Memories\n"]
        chars_used = len(lines[0])
        
        for i, r in enumerate(results):
            text = r.get("text", "")
            if not text:
                continue
            
            header = f"### Memory {i+1} (relevance: {r['score']:.2f}, source: {r['source']})\n"
            
            concepts_line = ""
            if include_concepts and r.get("concepts"):
                concepts_line = f"**Concepts:** {', '.join(r['concepts'])}\n"
            
            block = header + concepts_line + text + "\n\n"
            
            if chars_used + len(block) > max_chars:
                # Обрезаем текст
                remaining = max_chars - chars_used - len(header) - len(concepts_line) - 10
                if remaining > 50:
                    block = header + concepts_line + text[:remaining] + "...\n\n"
                else:
                    break
            
            lines.append(block)
            chars_used += len(block)
        
        return "".join(lines)
    
    def _format_xml(
        self, results: List[Dict], max_chars: int, include_concepts: bool
    ) -> str:
        lines = ["<memories>\n"]
        chars_used = len(lines[0])
        
        for i, r in enumerate(results):
            text = r.get("text", "")
            if not text:
                continue
            
            concepts_attr = ""
            if include_concepts and r.get("concepts"):
                concepts_attr = f' concepts="{",".join(r["concepts"])}"'
            
            entry = (
                f'  <memory rank="{i+1}" relevance="{r["score"]:.2f}" '
                f'source="{r["source"]}"{concepts_attr}>\n'
                f'    {text}\n'
                f'  </memory>\n'
            )
            
            if chars_used + len(entry) > max_chars:
                remaining = max_chars - chars_used - 100
                if remaining > 50:
                    entry = (
                        f'  <memory rank="{i+1}" relevance="{r["score"]:.2f}" '
                        f'source="{r["source"]}"{concepts_attr}>\n'
                        f'    {text[:remaining]}...\n'
                        f'  </memory>\n'
                    )
                else:
                    break
            
            lines.append(entry)
            chars_used += len(entry)
        
        lines.append("</memories>")
        return "".join(lines)
    
    def _format_plain(
        self, results: List[Dict], max_chars: int, include_concepts: bool
    ) -> str:
        lines = []
        chars_used = 0
        
        for i, r in enumerate(results):
            text = r.get("text", "")
            if not text:
                continue
            
            block = f"[{i+1}] {text}\n"
            
            if chars_used + len(block) > max_chars:
                remaining = max_chars - chars_used - 20
                if remaining > 50:
                    block = f"[{i+1}] {text[:remaining]}...\n"
                else:
                    break
            
            lines.append(block)
            chars_used += len(block)
        
        return "".join(lines)
    
    # ===== CONSOLIDATION (Dream Phase) =====
    
    def consolidate(self) -> Dict:
        """
        Консолидация памяти — «сон» между сессиями.
        
        1. Иерархическая консолидация: episodic → semantic
        2. Граф: decay слабых связей
        3. Dream consolidation: агрессивное обобщение
        
        Returns:
            stats: статистика консолидации
        """
        consolidation_stats = {}

        # 0. Flush pending Hebbian updates
        flushed = self._flush_hebbian()
        consolidation_stats["hebbian_flushed"] = flushed

        # 1. Иерархическая консолидация (working → episodic → semantic)
        hier_stats = self.hierarchy.consolidate(min_access=1, min_strength=0.2)
        consolidation_stats["hierarchy"] = hier_stats
        
        # 2. Dream consolidation (мягкая: episodic → semantic без удаления эпизодов)
        dream_stats = self.hierarchy.dream_consolidate()
        consolidation_stats["dream"] = dream_stats
        
        # 3. Graph decay (только слабые связи)
        edges_removed = self.associations.apply_decay()
        consolidation_stats["graph_edges_removed"] = edges_removed
        
        # 4. Мягкий decay: только sensory и working, НЕ episodic/semantic
        #    (episodic/semantic затухают только при явном apply_decay())
        self.hierarchy.sensory.apply_decay(rate=0.3)
        self.hierarchy.working.apply_decay(rate=0.1)
        consolidation_stats["decay"] = "soft (sensory+working only)"
        
        self.stats["total_consolidated"] += 1
        
        return consolidation_stats
    
    # ===== SESSION MANAGEMENT =====
    
    def new_session(self) -> int:
        """
        Начинает новую сессию. Вызывать в начале каждого диалога.
        
        Returns:
            session_id
        """
        self._session_id += 1
        self._session_entries = []
        self.stats["sessions"] += 1
        return self._session_id
    
    def end_session(self, consolidate: bool = True) -> Dict:
        """
        Завершает текущую сессию.
        
        Args:
            consolidate: выполнить консолидацию при завершении
            
        Returns:
            session_stats
        """
        session_stats = {
            "session_id": self._session_id,
            "entries_count": len(self._session_entries),
        }
        
        if consolidate:
            session_stats["consolidation"] = self.consolidate()
        
        return session_stats
    
    # ===== EVICTION =====
    
    def _evict_old_entries(self, target_size: Optional[int] = None) -> int:
        """Удаляет старые/неважные записи для освобождения места."""
        target = target_size or int(self.max_entries * 0.9)
        
        if len(self._entries) <= target:
            return 0
        
        # Score = importance * access_count / age
        now = time.time()
        scored = []
        for eid, entry in self._entries.items():
            age = now - entry.timestamp + 1.0
            score = entry.importance * (1 + entry.access_count) / math.log1p(age)
            scored.append((eid, score))
        
        scored.sort(key=lambda x: x[1])
        
        n_to_remove = len(self._entries) - target
        removed_ids = set()
        for eid, _ in scored[:n_to_remove]:
            entry = self._entries.pop(eid, None)
            if entry is not None:
                removed_ids.add(eid)
                # Чистим инвертированный индекс концептов
                for cid in entry.concept_ids:
                    lst = self._concept_to_entries.get(cid)
                    if lst:
                        self._concept_to_entries[cid] = [
                            x for x in lst if x not in removed_ids
                        ]

        if removed_ids:
            # Полный rebuild embedding матрицы после eviction
            self._index_dirty = True
            self._rebuild_entry_index()

        return len(removed_ids)
    
    # ===== PERSISTENCE =====
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Сохраняет память на диск.
        
        Сохраняет:
        - Все MemoryEntry (текст + embedding + metadata)
        - AssociationGraph (концепты + связи)
        - HierarchicalMemory state
        - Статистику
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "version": "0.19.0",
            "embedding_dim": self.embedding_dim,
            "max_entries": self.max_entries,
            
            # Entries
            "entries": {
                str(eid): {
                    "text": e.text,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                    "importance": e.importance,
                    "access_count": e.access_count,
                    "concept_ids": e.concept_ids,
                }
                for eid, e in self._entries.items()
            },
            "next_entry_id": self._next_entry_id,
            
            # Concepts
            "concepts": {
                str(nid): {
                    "name": c.name,
                    "embedding": c.embedding,
                    "metadata": c.metadata,
                    "created_at": c.created_at,
                    "last_accessed": c.last_accessed,
                    "access_count": c.access_count,
                    "strength": c.strength,
                }
                for nid, c in self.associations._id_to_concept.items()
            },
            "next_concept_id": self.associations._next_id,
            
            # Graph edges (dict-based, v0.20.0+)
            "graph_edges": {f"{a},{b}": w for (a, b), w in self.associations._edges.items()},
            
            # Hierarchical memory state
            "hierarchy_episodic_patterns": self.hierarchy.episodic._patterns[:self.hierarchy.episodic._num_stored],
            "hierarchy_episodic_strengths": self.hierarchy.episodic._strengths[:self.hierarchy.episodic._num_stored],
            "hierarchy_episodic_metadata": [
                self.hierarchy.episodic._metadata[i]
                for i in range(self.hierarchy.episodic._num_stored)
            ],
            
            # Session
            "session_id": self._session_id,
            
            # Stats
            "stats": self.stats,
        }
        
        torch.save(state, path)
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
    ) -> "NGTMemoryForLLM":
        """Загружает память с диска."""
        state = torch.load(path, map_location=device, weights_only=False)
        
        memory = cls(
            embedding_dim=state["embedding_dim"],
            max_entries=state["max_entries"],
            device=device,
        )
        
        # Восстанавливаем entries
        for eid_str, edata in state.get("entries", {}).items():
            eid = int(eid_str)
            entry = MemoryEntry(
                entry_id=eid,
                text=edata["text"],
                embedding=edata["embedding"],
                metadata=edata.get("metadata", {}),
                importance=edata.get("importance", 1.0),
                concept_ids=edata.get("concept_ids", []),
            )
            entry.timestamp = edata.get("timestamp", time.time())
            entry.access_count = edata.get("access_count", 0)
            memory._entries[eid] = entry
        memory._next_entry_id = state.get("next_entry_id", 0)
        # Восстанавливаем _entry_id_list и помечаем для rebuild
        memory._entry_id_list = list(memory._entries.keys())
        memory._emb_buffer = []
        memory._index_dirty = True

        # Восстанавливаем инвертированный индекс концептов
        for eid, entry in memory._entries.items():
            for cid in entry.concept_ids:
                if cid not in memory._concept_to_entries:
                    memory._concept_to_entries[cid] = []
                memory._concept_to_entries[cid].append(eid)

        # Восстанавливаем концепты
        for nid_str, cdata in state.get("concepts", {}).items():
            nid = int(nid_str)
            concept = ConceptNode(
                node_id=nid,
                name=cdata["name"],
                embedding=cdata["embedding"],
                metadata=cdata.get("metadata", {}),
            )
            concept.created_at = cdata.get("created_at", time.time())
            concept.last_accessed = cdata.get("last_accessed", time.time())
            concept.access_count = cdata.get("access_count", 1)
            concept.strength = cdata.get("strength", 1.0)
            
            memory.associations._id_to_concept[nid] = concept
            memory.associations._name_to_id[concept.name] = nid
            memory.associations._embeddings[nid] = concept.embedding
            memory.associations._active_ids.append(nid)
        memory.associations._next_id = state.get("next_concept_id", 0)
        memory.associations._emb_dirty = True

        # Восстанавливаем граф (новый dict формат v0.20.0+)
        if "graph_edges" in state:
            for key_str, w in state["graph_edges"].items():
                a_str, b_str = key_str.split(",")
                a, b = int(a_str), int(b_str)
                key = (a, b)
                memory.associations._edges[key] = w
                if a not in memory.associations._adj:
                    memory.associations._adj[a] = {}
                if b not in memory.associations._adj:
                    memory.associations._adj[b] = {}
                memory.associations._adj[a][b] = w
                memory.associations._adj[b][a] = w
        
        # Восстанавливаем episodic memory
        if "hierarchy_episodic_patterns" in state:
            ep = memory.hierarchy.episodic
            patterns = state["hierarchy_episodic_patterns"]
            strengths = state["hierarchy_episodic_strengths"]
            metadata_list = state.get("hierarchy_episodic_metadata", [])
            
            n = min(patterns.shape[0], ep.capacity)
            ep._patterns[:n] = patterns[:n]
            ep._strengths[:n] = strengths[:n]
            for i in range(min(n, len(metadata_list))):
                ep._metadata[i] = metadata_list[i]
            ep._num_stored = n
        
        # Восстанавливаем session и stats
        memory._session_id = state.get("session_id", 0)
        memory.stats = state.get("stats", memory.stats)
        
        return memory
    
    # ===== STATS & DEBUG =====
    
    def get_statistics(self) -> Dict:
        """Возвращает полную статистику памяти."""
        return {
            **self.stats,
            "entries_count": len(self._entries),
            "session_entries": len(self._session_entries),
            "current_session": self._session_id,
            "hierarchy": self.hierarchy.get_state(),
            "associations": self.associations.get_statistics(),
        }
    
    def __repr__(self) -> str:
        return (
            f"NGTMemoryForLLM("
            f"entries={len(self._entries)}, "
            f"concepts={self.associations.num_concepts}, "
            f"edges={self.associations.num_edges}, "
            f"episodic={self.hierarchy.episodic.size}, "
            f"semantic={self.hierarchy.semantic.size}, "
            f"dim={self.embedding_dim})"
        )

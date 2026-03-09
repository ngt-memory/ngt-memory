"""
AssociationGraph и ConceptNode — граф ассоциаций между концептами.

Лёгкая реализация без nn.Module overhead.
Связи формируются через Hebbian co-occurrence (dict-based граф).
"""

import math
import time
from typing import Optional, Dict, List, Tuple

import torch


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

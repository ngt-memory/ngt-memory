"""
Dynamic Graph - Динамический граф связей NGT

Реализует граф узлов, где связи формируются и изменяются
динамически во время обучения и инференса.

v0.3.1: Тензорное хранение рёбер (GPU-совместимо).
Вместо Python dict используются тензоры _edge_index, _edge_weight, _edge_activation.
Adjacency matrix кэшируется и инвалидируется при изменении графа.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from collections import deque


class DynamicGraph(nn.Module):
    """
    Динамический граф связей между нейронами.
    
    Хранение рёбер — тензорное (GPU-совместимо):
    - _edge_index: [2, E] — индексы (src, dst) рёбер
    - _edge_weight: [E] — веса рёбер
    - _edge_activation: [E] — счётчики активаций
    
    Связи (рёбра) могут:
    - Создаваться при совместной активации узлов
    - Усиливаться при повторной активации
    - Ослабевать при неиспользовании
    - Удаляться при падении веса ниже порога
    
    Attributes:
        num_nodes: количество узлов в графе
        hidden_dim: размерность скрытого состояния узла
        edge_threshold: минимальный вес связи для сохранения
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        edge_threshold: float = 0.01,
        max_edges_per_node: int = 100,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.edge_threshold = edge_threshold
        self.max_edges_per_node = max_edges_per_node
        self.device = device
        
        # Состояния узлов (node embeddings)
        self.node_states = nn.Parameter(
            torch.randn(num_nodes, hidden_dim, device=device) * 0.1
        )
        
        # Тензорное хранение рёбер (GPU-совместимо)
        self._edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        self._edge_weight = torch.zeros(0, dtype=torch.float, device=device)
        self._edge_activation = torch.zeros(0, dtype=torch.long, device=device)
        
        # Кэш adjacency matrix (инвалидируется при изменении графа)
        self._adj_cache: Optional[torch.Tensor] = None
        self._adj_dirty = True
        
        # История активаций узлов (для Hebbian learning, ограничена)
        self.node_activation_history: deque = deque(maxlen=100)
        
        # Статистика
        self.stats = {
            "edges_created": 0,
            "edges_removed": 0,
            "total_activations": 0,
        }
    
    def _invalidate_cache(self) -> None:
        """Инвалидирует кэш adjacency matrix."""
        self._adj_dirty = True
    
    def _find_edge(self, src: int, dst: int) -> int:
        """
        Находит индекс ребра (src, dst) в _edge_index.
        Возвращает -1 если ребро не найдено.
        """
        if self._edge_index.shape[1] == 0:
            return -1
        mask = (self._edge_index[0] == src) & (self._edge_index[1] == dst)
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            return -1
        return indices[0].item()
    
    def get_node_states(self) -> torch.Tensor:
        """Возвращает текущие состояния всех узлов."""
        return self.node_states
    
    def get_edge_count(self) -> int:
        """Возвращает количество активных связей."""
        return self._edge_index.shape[1]
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Возвращает матрицу смежности как dense tensor.
        Кэшируется — повторные вызовы без изменения графа O(1).
        """
        if not self._adj_dirty and self._adj_cache is not None:
            return self._adj_cache
        
        adj = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        if self._edge_index.shape[1] > 0:
            adj[self._edge_index[0], self._edge_index[1]] = self._edge_weight
        
        self._adj_cache = adj
        self._adj_dirty = False
        return adj
    
    def add_edge(self, src: int, dst: int, weight: float = 0.1) -> None:
        """Добавляет связь между узлами."""
        if src == dst:
            return  # Нет self-loops
        
        idx = self._find_edge(src, dst)
        if idx == -1:
            # Новое ребро — добавляем в тензоры
            new_edge = torch.tensor([[src], [dst]], dtype=torch.long, device=self.device)
            self._edge_index = torch.cat([self._edge_index, new_edge], dim=1)
            self._edge_weight = torch.cat([self._edge_weight, torch.tensor([weight], device=self.device)])
            self._edge_activation = torch.cat([self._edge_activation, torch.tensor([1], dtype=torch.long, device=self.device)])
            self.stats["edges_created"] += 1
        else:
            # Усиливаем существующую связь
            self._edge_weight[idx] = min(1.0, self._edge_weight[idx].item() + weight * 0.1)
            self._edge_activation[idx] += 1
        
        self._invalidate_cache()
    
    def update_edge(self, src: int, dst: int, delta: float) -> None:
        """Обновляет вес связи на delta."""
        idx = self._find_edge(src, dst)
        if idx >= 0:
            self._edge_weight[idx] += delta
            if self._edge_weight[idx].item() < self.edge_threshold:
                self.remove_edge(src, dst)
            else:
                self._invalidate_cache()
    
    def remove_edge(self, src: int, dst: int) -> None:
        """Удаляет связь между узлами."""
        idx = self._find_edge(src, dst)
        if idx >= 0:
            # Удаляем элемент по индексу
            mask = torch.ones(self._edge_index.shape[1], dtype=torch.bool, device=self.device)
            mask[idx] = False
            self._edge_index = self._edge_index[:, mask]
            self._edge_weight = self._edge_weight[mask]
            self._edge_activation = self._edge_activation[mask]
            self.stats["edges_removed"] += 1
            self._invalidate_cache()
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Возвращает соседей узла с весами связей."""
        if self._edge_index.shape[1] == 0:
            return []
        mask = self._edge_index[0] == node_id
        dst_nodes = self._edge_index[1, mask]
        weights = self._edge_weight[mask]
        # Сортируем по убыванию веса
        sorted_idx = weights.argsort(descending=True)
        return [(dst_nodes[i].item(), weights[i].item()) for i in sorted_idx]
    
    def activate_nodes(self, node_ids: List[int], activations: torch.Tensor) -> None:
        """
        Активирует указанные узлы и обновляет их состояния.
        
        Args:
            node_ids: индексы активируемых узлов
            activations: значения активации для каждого узла
        """
        self.stats["total_activations"] += len(node_ids)
        
        # Сохраняем историю активаций
        activation_vector = torch.zeros(self.num_nodes, device=self.device)
        ids_tensor = torch.tensor(node_ids, dtype=torch.long, device=self.device)
        activation_vector[ids_tensor] = activations.to(self.device)
        
        self.node_activation_history.append(activation_vector)
    
    def get_sparse_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает рёбра в формате PyTorch Geometric.
        Прямой доступ к тензорам — без конвертации dict → tensor.
        
        Returns:
            edge_index: [2, num_edges] - индексы рёбер
            edge_weight: [num_edges] - веса рёбер
        """
        return self._edge_index, self._edge_weight
    
    def get_edge_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает полные данные рёбер (index, weight, activation).
        
        Returns:
            edge_index: [2, E]
            edge_weight: [E]
            edge_activation: [E]
        """
        return self._edge_index, self._edge_weight, self._edge_activation
    
    # ========== Batch-операции (полностью тензорные, GPU-совместимые) ==========
    
    def batch_add_edges(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor
    ) -> int:
        """
        Добавляет batch рёбер за одну операцию (GPU-совместимо).
        Пропускает self-loops. Для существующих рёбер — усиливает вес.
        
        Args:
            src: [K] — индексы источников
            dst: [K] — индексы целей
            weights: [K] — веса новых рёбер
            
        Returns:
            Количество реально добавленных новых рёбер
        """
        # Фильтруем self-loops
        valid = src != dst
        src = src[valid]
        dst = dst[valid]
        weights = weights[valid]
        
        if len(src) == 0:
            return 0
        
        new_count = 0
        
        if self._edge_index.shape[1] == 0:
            # Граф пустой — добавляем все рёбра
            self._edge_index = torch.stack([src, dst], dim=0)
            self._edge_weight = weights.clone()
            self._edge_activation = torch.ones(len(src), dtype=torch.long, device=self.device)
            new_count = len(src)
        else:
            # Кодируем рёбра как уникальные ключи для быстрого поиска
            N = self.num_nodes
            existing_keys = self._edge_index[0] * N + self._edge_index[1]
            new_keys = src * N + dst
            
            # Находим какие рёбра уже существуют
            # Используем broadcasting: [K_new, 1] vs [1, E_existing]
            # Для больших графов используем searchsorted
            existing_sorted, sort_idx = existing_keys.sort()
            insert_pos = torch.searchsorted(existing_sorted, new_keys)
            insert_pos = insert_pos.clamp(max=len(existing_sorted) - 1)
            found_mask = existing_sorted[insert_pos] == new_keys
            
            # Обновляем существующие рёбра
            if found_mask.any():
                found_positions = sort_idx[insert_pos[found_mask]]
                self._edge_weight[found_positions] = torch.clamp(
                    self._edge_weight[found_positions] + weights[found_mask] * 0.1,
                    max=1.0
                )
                self._edge_activation[found_positions] += 1
            
            # Добавляем новые рёбра
            new_mask = ~found_mask
            if new_mask.any():
                new_src = src[new_mask]
                new_dst = dst[new_mask]
                new_w = weights[new_mask]
                new_count = new_mask.sum().item()
                
                self._edge_index = torch.cat([
                    self._edge_index,
                    torch.stack([new_src, new_dst], dim=0)
                ], dim=1)
                self._edge_weight = torch.cat([self._edge_weight, new_w])
                self._edge_activation = torch.cat([
                    self._edge_activation,
                    torch.ones(new_count, dtype=torch.long, device=self.device)
                ])
        
        self.stats["edges_created"] += new_count
        self._invalidate_cache()
        return new_count
    
    def batch_update_weights(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        deltas: torch.Tensor
    ) -> int:
        """
        Обновляет веса существующих рёбер batch-операцией.
        
        Args:
            src: [K] — индексы источников
            dst: [K] — индексы целей
            deltas: [K] — дельты весов
            
        Returns:
            Количество обновлённых рёбер
        """
        if self._edge_index.shape[1] == 0 or len(src) == 0:
            return 0
        
        N = self.num_nodes
        existing_keys = self._edge_index[0] * N + self._edge_index[1]
        update_keys = src * N + dst
        
        existing_sorted, sort_idx = existing_keys.sort()
        insert_pos = torch.searchsorted(existing_sorted, update_keys)
        insert_pos = insert_pos.clamp(max=len(existing_sorted) - 1)
        found_mask = existing_sorted[insert_pos] == update_keys
        
        updated = 0
        if found_mask.any():
            found_positions = sort_idx[insert_pos[found_mask]]
            self._edge_weight[found_positions] += deltas[found_mask]
            updated = found_mask.sum().item()
            self._invalidate_cache()
        
        return updated
    
    def apply_decay_tensor(self, decay_rate: float, min_weight: float = 0.0) -> int:
        """
        Применяет затухание ко всем рёбрам и удаляет слабые (тензорная операция).
        
        Args:
            decay_rate: множитель затухания (вес *= 1 - decay_rate)
            min_weight: минимальный вес для сохранения
            
        Returns:
            Количество удалённых рёбер
        """
        if self._edge_index.shape[1] == 0:
            return 0
        
        self._edge_weight *= (1 - decay_rate)
        
        # Удаляем слабые рёбра
        keep_mask = self._edge_weight >= min_weight
        removed = (~keep_mask).sum().item()
        
        if removed > 0:
            self._edge_index = self._edge_index[:, keep_mask]
            self._edge_weight = self._edge_weight[keep_mask]
            self._edge_activation = self._edge_activation[keep_mask]
            self.stats["edges_removed"] += removed
            self._invalidate_cache()
        
        return removed
    
    def prune_weak_edges(self, threshold: Optional[float] = None) -> int:
        """
        Удаляет слабые связи ниже порога (тензорная операция).
        
        Returns:
            Количество удалённых связей
        """
        threshold = threshold or self.edge_threshold
        
        if self._edge_index.shape[1] == 0:
            return 0
        
        keep_mask = self._edge_weight >= threshold
        removed = (~keep_mask).sum().item()
        
        if removed > 0:
            self._edge_index = self._edge_index[:, keep_mask]
            self._edge_weight = self._edge_weight[keep_mask]
            self._edge_activation = self._edge_activation[keep_mask]
            self.stats["edges_removed"] += removed
            self._invalidate_cache()
        
        return removed
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику графа."""
        E = self._edge_index.shape[1]
        if E > 0:
            w = self._edge_weight
            avg_w = w.mean().item()
            max_w = w.max().item()
            min_w = w.min().item()
        else:
            avg_w = max_w = min_w = 0
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": E,
            "avg_edge_weight": avg_w,
            "max_edge_weight": max_w,
            "min_edge_weight": min_w,
            "edges_created": self.stats["edges_created"],
            "edges_removed": self.stats["edges_removed"],
            "total_activations": self.stats["total_activations"],
            "sparsity": 1 - E / (self.num_nodes ** 2),
        }
    
    # Порог рёбер для chunked message passing (экономия VRAM)
    _EDGE_CHUNK_THRESHOLD = 50000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: агрегация сообщений от соседей.
        
        Поддерживает как 2D [num_nodes, hidden_dim], так и
        3D [batch, num_nodes, hidden_dim] входы.
        
        При большом количестве рёбер (>50K) используется chunked
        message passing для экономии GPU памяти.
        
        Args:
            x: входные признаки узлов [num_nodes, hidden_dim]
               или [batch, num_nodes, hidden_dim]
            
        Returns:
            Обновлённые состояния узлов (та же размерность что и вход)
        """
        if self._edge_index.shape[1] == 0:
            return x
        
        E = self._edge_index.shape[1]
        src_nodes = self._edge_index[0]  # [E]
        dst_nodes = self._edge_index[1]  # [E]
        
        # Detach + clone edge weights — они обучаются через Hebbian, не через backprop.
        # Clone необходим, т.к. Hebbian модифицирует _edge_weight in-place между forward вызовами.
        edge_w = self._edge_weight.detach().clone()
        
        if x.dim() == 3:
            # Батчевая обработка: [batch, num_nodes, hidden_dim]
            aggregated = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            
            if E <= self._EDGE_CHUNK_THRESHOLD:
                # Стандартный путь — все рёбра сразу
                src_features = x[:, src_nodes, :]  # [batch, E, hidden_dim]
                weighted_messages = src_features * edge_w.unsqueeze(0).unsqueeze(2)
                idx = dst_nodes.unsqueeze(0).unsqueeze(2).expand_as(weighted_messages)
                aggregated.scatter_add_(1, idx, weighted_messages)
            else:
                # Chunked message passing — обрабатываем рёбра блоками
                chunk_size = self._EDGE_CHUNK_THRESHOLD
                for start in range(0, E, chunk_size):
                    end = min(start + chunk_size, E)
                    src_chunk = src_nodes[start:end]
                    dst_chunk = dst_nodes[start:end]
                    ew_chunk = edge_w[start:end]
                    
                    src_features = x[:, src_chunk, :]  # [batch, chunk, hidden_dim]
                    weighted_messages = src_features * ew_chunk.unsqueeze(0).unsqueeze(2)
                    idx = dst_chunk.unsqueeze(0).unsqueeze(2).expand_as(weighted_messages)
                    aggregated.scatter_add_(1, idx, weighted_messages)
            
            return x + aggregated
        else:
            # 2D: [num_nodes, hidden_dim]
            src_features = x[src_nodes]  # [E, hidden_dim]
            weighted_messages = src_features * edge_w.unsqueeze(1)  # [E, hidden_dim]
            
            # Out-of-place scatter для совместимости с autograd
            aggregated = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            aggregated.scatter_add_(0, dst_nodes.unsqueeze(1).expand_as(weighted_messages), weighted_messages)
            return x + aggregated

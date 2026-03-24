"""
Hebbian Plasticity Core - Ядро Хеббовской пластичности

Реализует правило Хебба для обновления связей:
"Нейроны, которые активируются вместе - соединяются сильнее"

Δw = η × pre_activation × post_activation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from collections import deque
import numpy as np


class HebbianPlasticityCore(nn.Module):
    """
    Ядро Хеббовской пластичности для NGT.
    
    Реализует локальное обновление весов связей на основе
    корреляции активаций пре- и пост-синаптических нейронов.
    
    Правило обновления:
        Δw_ij = η × a_i × a_j  (для новых и существующих рёбер)
        
    Контроль sparsity:
        - Adaptive budget: новые рёбра создаются только если headroom > 0
        - Competitive weakening: рёбра с низкой корреляцией ослабляются
          пропорционально заполненности графа
        
    Attributes:
        learning_rate: скорость Хеббовского обучения
        decay_rate: базовая скорость конкурентного ослабления
        threshold: порог корреляции для создания/усиления связей
        target_edge_ratio: целевая доля рёбер от N² (контроль sparsity)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
        threshold: float = 0.1,
        max_weight: float = 1.0,
        normalize: bool = True,
        target_edge_ratio: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.max_weight = max_weight
        self.normalize = normalize
        self.target_edge_ratio = target_edge_ratio  # целевая доля рёбер от N²
        self.device = device
        
        # Статистика обучения
        self.stats = {
            "updates": 0,
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "new_connections": 0,
        }
        
        # История изменений весов для анализа (ограничена для предотвращения утечки памяти)
        self.weight_history: deque = deque(maxlen=1000)
    
    def compute_hebbian_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        current_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Вычисляет Хеббовское обновление весов.
        
        Args:
            pre_activation: активации пре-синаптических нейронов [N]
            post_activation: активации пост-синаптических нейронов [M]
            current_weight: текущие веса связей [N, M] (опционально)
            
        Returns:
            delta_weight: изменение весов [N, M]
        """
        # Внешнее произведение активаций: Δw = η × pre × post^T
        delta_weight = self.learning_rate * torch.outer(pre_activation, post_activation)
        
        # Применяем затухание к существующим весам
        if current_weight is not None:
            delta_weight = delta_weight - self.decay_rate * current_weight
        
        return delta_weight
    
    def update_graph_weights(
        self,
        graph,  # DynamicGraph
        activations: torch.Tensor,
        active_nodes: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Sparse Hebbian update: обновляет граф, создавая только top-k
        самых сильных новых связей за шаг.
        
        Архитектура:
        1. Вычисляем корреляции только для активных узлов
        2. Усиливаем существующие рёбра пропорционально корреляции
        3. Создаём только top-k новых рёбер с наибольшей корреляцией
        4. Конкурентное ослабление: существующие рёбра с низкой
           корреляцией ослабляются (competitive Hebbian learning)
        
        Это обеспечивает естественный контроль sparsity без
        внешнего enforce_max_edges.
        
        Args:
            graph: DynamicGraph для обновления
            activations: активации всех узлов [num_nodes]
            active_nodes: маска активных узлов (опционально)
            
        Returns:
            Статистика обновления
        """
        self.stats["updates"] += 1
        
        num_nodes = activations.shape[0]
        max_edges = int(num_nodes * num_nodes * self.target_edge_ratio)
        
        # Если не указаны активные узлы, считаем активными все с активацией > threshold
        if active_nodes is None:
            active_nodes = (activations > self.threshold).nonzero(as_tuple=True)[0]
        
        # Нормализуем активации если нужно
        if self.normalize and activations.max() > 0:
            activations = activations / activations.max()
        
        strengthened = 0
        weakened = 0
        new_connections = 0
        E = graph._edge_index.shape[1]
        
        if len(active_nodes) > 0:
            active_acts = activations[active_nodes]  # [k]
            k = len(active_nodes)
            
            # Матрица Хеббовских корреляций: delta[i,j] = η × a_i × a_j
            delta_matrix = self.learning_rate * torch.outer(active_acts, active_acts)  # [k, k]
            delta_matrix.fill_diagonal_(0.0)
            
            # === Шаг 1: Обновляем существующие рёбра ===
            if E > 0:
                src_e = graph._edge_index[0]  # [E]
                dst_e = graph._edge_index[1]  # [E]
                
                # Корреляция для каждого существующего ребра
                src_act = activations[src_e]  # [E]
                dst_act = activations[dst_e]  # [E]
                edge_corr = self.learning_rate * src_act * dst_act  # [E]
                
                # Усиливаем рёбра с высокой корреляцией
                high_corr = edge_corr > self.threshold * self.learning_rate * 0.5
                if high_corr.any():
                    graph._edge_weight[high_corr] += edge_corr[high_corr]
                    graph._edge_activation[high_corr] += 1
                    strengthened = high_corr.sum().item()
                
                # Конкурентное ослабление: рёбра с низкой корреляцией
                # Сила ослабления пропорциональна заполненности графа
                fill_ratio = E / max(max_edges, 1)
                competitive_decay = self.decay_rate * (1 + fill_ratio * 5)
                low_corr = ~high_corr
                if low_corr.any():
                    graph._edge_weight[low_corr] *= (1 - competitive_decay)
                    weakened = low_corr.sum().item()
                
                graph._edge_weight.clamp_(min=0.0, max=self.max_weight)
            
            # === Шаг 2: Адаптивный бюджет новых рёбер ===
            # Чем ближе к target — тем меньше бюджет
            headroom = max(max_edges - E, 0)
            budget = min(headroom, max(num_nodes // 2, 4))
            
            if budget > 0:
                # Генерируем все пары активных узлов
                ii_grid, jj_grid = torch.meshgrid(
                    torch.arange(k, device=activations.device),
                    torch.arange(k, device=activations.device),
                    indexing='ij'
                )
                off_diag = ii_grid != jj_grid
                ii_flat = ii_grid[off_diag]
                jj_flat = jj_grid[off_diag]
                
                src_nodes = active_nodes[ii_flat]
                dst_nodes = active_nodes[jj_flat]
                deltas = delta_matrix[ii_flat, jj_flat]
                
                # Порог для создания
                creation_threshold = self.threshold * self.learning_rate
                
                # Фильтруем кандидатов выше порога
                candidate_mask = deltas > creation_threshold
                if candidate_mask.any():
                    cand_src = src_nodes[candidate_mask]
                    cand_dst = dst_nodes[candidate_mask]
                    cand_delta = deltas[candidate_mask]
                    
                    # Исключаем уже существующие рёбра
                    if E > 0:
                        N = graph.num_nodes
                        existing_keys = graph._edge_index[0] * N + graph._edge_index[1]
                        candidate_keys = cand_src * N + cand_dst
                        
                        existing_sorted, _ = existing_keys.sort()
                        insert_pos = torch.searchsorted(existing_sorted, candidate_keys)
                        insert_pos = insert_pos.clamp(max=max(len(existing_sorted) - 1, 0))
                        already_exists = existing_sorted[insert_pos] == candidate_keys
                        
                        new_mask = ~already_exists
                        cand_src = cand_src[new_mask]
                        cand_dst = cand_dst[new_mask]
                        cand_delta = cand_delta[new_mask]
                    
                    # Top-k по силе корреляции (адаптивный бюджет)
                    if len(cand_src) > budget:
                        _, topk_idx = torch.topk(cand_delta, budget)
                        cand_src = cand_src[topk_idx]
                        cand_dst = cand_dst[topk_idx]
                        cand_delta = cand_delta[topk_idx]
                    
                    if len(cand_src) > 0:
                        new_connections = graph.batch_add_edges(
                            cand_src, cand_dst, cand_delta
                        )
        
        # Обновляем статистику
        self.stats["connections_strengthened"] += strengthened
        self.stats["connections_weakened"] += weakened
        self.stats["new_connections"] += new_connections
        
        update_stats = {
            "strengthened": strengthened,
            "weakened": weakened,
            "new_connections": new_connections,
            "active_nodes": len(active_nodes),
        }
        
        # Сохраняем историю
        self.weight_history.append({
            "step": self.stats["updates"],
            "total_edges": graph.get_edge_count(),
            **update_stats
        })
        
        return update_stats
    
    def compute_correlation_matrix(
        self,
        activation_history: list
    ) -> torch.Tensor:
        """
        Вычисляет матрицу корреляций активаций.
        
        Args:
            activation_history: список тензоров активаций [T, num_nodes]
            
        Returns:
            correlation_matrix: [num_nodes, num_nodes]
        """
        if not activation_history:
            return torch.zeros(1, 1)
        
        # Стекаем историю в матрицу [T, N]
        history = torch.stack(activation_history)
        
        # Центрируем
        history = history - history.mean(dim=0, keepdim=True)
        
        # Вычисляем корреляцию
        std = history.std(dim=0, keepdim=True) + 1e-8
        history_norm = history / std
        
        correlation = torch.mm(history_norm.t(), history_norm) / history.shape[0]
        
        return correlation
    
    def get_learning_signal(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        reward: float = 1.0
    ) -> torch.Tensor:
        """
        Вычисляет сигнал обучения с учётом награды (reward-modulated Hebbian).
        
        Это расширение классического правила Хебба:
        Δw = η × reward × pre × post
        
        Args:
            pre_activation: активации пре-синаптических нейронов
            post_activation: активации пост-синаптических нейронов
            reward: модулирующий сигнал награды
            
        Returns:
            learning_signal: сигнал для обновления весов
        """
        return reward * self.compute_hebbian_update(pre_activation, post_activation)
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику обучения."""
        return {
            **self.stats,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "threshold": self.threshold,
            "history_length": len(self.weight_history),
        }
    
    def reset_statistics(self) -> None:
        """Сбрасывает статистику."""
        self.stats = {
            "updates": 0,
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "new_connections": 0,
        }
        self.weight_history = deque(maxlen=1000)
    
    def forward(
        self,
        graph,
        activations: torch.Tensor
    ) -> Dict[str, float]:
        """
        Прямой проход: обновляет веса графа на основе активаций.
        
        Args:
            graph: DynamicGraph для обновления
            activations: активации узлов [num_nodes]
            
        Returns:
            Статистика обновления
        """
        return self.update_graph_weights(graph, activations)

"""
Structural Decay - Структурное затухание NGT

Реализует механизм ослабления и удаления неиспользуемых связей,
аналогичный синаптическому дрифту в биологическом мозге.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


class StructuralDecay(nn.Module):
    """
    Модуль структурного затухания для NGT.
    
    Обеспечивает:
    - Мягкое затухание неиспользуемых связей
    - Удаление связей ниже min_weight
    - Опциональный гомеостаз (по умолчанию выключен)
    
    Контроль sparsity (бюджет рёбер, конкурентное ослабление)
    реализован в HebbianPlasticityCore.
    
    Это аналог синаптического прунинга в мозге - процесса,
    при котором неиспользуемые синапсы удаляются.
    
    Attributes:
        decay_rate: скорость затухания
        min_weight: минимальный вес для сохранения связи
        target_sparsity: целевая разреженность графа
    """
    
    def __init__(
        self,
        decay_rate: float = 0.01,
        min_weight: float = 0.01,
        target_sparsity: float = 0.95,
        homeostasis_strength: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.target_sparsity = target_sparsity
        self.homeostasis_strength = homeostasis_strength
        self.device = device
        
        # История затухания для анализа (ограничена для предотвращения утечки памяти)
        self.decay_history: deque = deque(maxlen=1000)
        
        # Статистика
        self.stats = {
            "decay_steps": 0,
            "edges_decayed": 0,
            "edges_pruned": 0,
            "homeostasis_adjustments": 0,
        }
    
    def apply_decay(
        self,
        graph,  # DynamicGraph
        activity_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, int]:
        """
        Применяет затухание к связям графа.
        
        Args:
            graph: DynamicGraph для обновления
            activity_mask: маска активных связей (не затухают)
            
        Returns:
            Статистика затухания
        """
        self.stats["decay_steps"] += 1
        
        edges_decayed = 0
        edges_pruned = 0
        E = graph._edge_index.shape[1]
        
        if E > 0:
            if activity_mask is not None:
                # Определяем какие рёбра активны (оба конца активны)
                src_active = activity_mask[graph._edge_index[0]] > 0
                dst_active = activity_mask[graph._edge_index[1]] > 0
                edge_active = src_active & dst_active
                
                # Затухание только для неактивных рёбер
                inactive = ~edge_active
                graph._edge_weight[inactive] *= (1 - self.decay_rate)
                edges_decayed = inactive.sum().item()
            else:
                # Затухание для всех рёбер
                graph._edge_weight *= (1 - self.decay_rate)
                edges_decayed = E
            
            # Удаляем слабые рёбра
            keep_mask = graph._edge_weight >= self.min_weight
            edges_pruned = (~keep_mask).sum().item()
            
            if edges_pruned > 0:
                graph._edge_index = graph._edge_index[:, keep_mask]
                graph._edge_weight = graph._edge_weight[keep_mask]
                graph._edge_activation = graph._edge_activation[keep_mask]
                graph.stats["edges_removed"] += edges_pruned
            
            graph._invalidate_cache()
        
        self.stats["edges_decayed"] += edges_decayed
        self.stats["edges_pruned"] += edges_pruned
        
        # Сохраняем историю
        self.decay_history.append({
            "step": self.stats["decay_steps"],
            "edges_decayed": edges_decayed,
            "edges_pruned": edges_pruned,
            "total_edges": graph.get_edge_count(),
        })
        
        return {
            "edges_decayed": edges_decayed,
            "edges_pruned": edges_pruned,
        }
    
    def apply_homeostasis(
        self,
        graph,  # DynamicGraph
    ) -> Dict[str, float]:
        """
        Применяет гомеостаз для поддержания оптимальной структуры.
        
        Если связей слишком много - усиливает затухание.
        Если слишком мало - ослабляет затухание.
        
        Args:
            graph: DynamicGraph для анализа
            
        Returns:
            Параметры корректировки
        """
        num_nodes = graph.num_nodes
        num_edges = graph.get_edge_count()
        max_edges = num_nodes * num_nodes
        
        current_sparsity = 1 - (num_edges / max_edges)
        sparsity_error = current_sparsity - self.target_sparsity
        
        # Корректируем decay_rate
        adjustment = sparsity_error * self.homeostasis_strength
        
        if sparsity_error < 0:
            # Слишком много связей - усиливаем затухание
            self.decay_rate = min(0.5, self.decay_rate * (1 + abs(adjustment)))
            self.stats["homeostasis_adjustments"] += 1
        elif sparsity_error > 0.1:
            # Слишком мало связей - ослабляем затухание
            self.decay_rate = max(0.001, self.decay_rate * (1 - adjustment))
            self.stats["homeostasis_adjustments"] += 1
        
        return {
            "current_sparsity": current_sparsity,
            "target_sparsity": self.target_sparsity,
            "sparsity_error": sparsity_error,
            "adjusted_decay_rate": self.decay_rate,
        }
    
    def prune_by_importance(
        self,
        graph,  # DynamicGraph
        keep_ratio: float = 0.5
    ) -> int:
        """
        Удаляет наименее важные связи.
        
        Важность определяется как: weight × activation_count
        
        Args:
            graph: DynamicGraph для обрезки
            keep_ratio: доля связей для сохранения
            
        Returns:
            Количество удалённых связей
        """
        E = graph._edge_index.shape[1]
        if E == 0:
            return 0
        
        # Вычисляем важность: weight × log(1 + activation_count) — тензорная операция
        importance = graph._edge_weight * torch.log1p(graph._edge_activation.float())
        
        # Определяем сколько оставить
        num_to_keep = max(1, int(E * keep_ratio))
        
        if num_to_keep >= E:
            return 0
        
        # Оставляем top-k по важности
        _, keep_indices = torch.topk(importance, num_to_keep)
        keep_mask = torch.zeros(E, dtype=torch.bool, device=graph.device)
        keep_mask[keep_indices] = True
        
        removed = (~keep_mask).sum().item()
        
        graph._edge_index = graph._edge_index[:, keep_mask]
        graph._edge_weight = graph._edge_weight[keep_mask]
        graph._edge_activation = graph._edge_activation[keep_mask]
        graph.stats["edges_removed"] += removed
        graph._invalidate_cache()
        
        self.stats["edges_pruned"] += removed
        
        return removed
    
    def get_decay_analysis(self) -> Dict:
        """
        Анализирует историю затухания.
        
        Returns:
            Статистика и тренды затухания
        """
        if not self.decay_history:
            return {"status": "no_history"}
        
        recent = self.decay_history[-10:]
        
        avg_decayed = np.mean([h["edges_decayed"] for h in recent])
        avg_pruned = np.mean([h["edges_pruned"] for h in recent])
        
        # Тренд количества связей
        edge_counts = [h["total_edges"] for h in recent]
        if len(edge_counts) > 1:
            trend = (edge_counts[-1] - edge_counts[0]) / len(edge_counts)
        else:
            trend = 0
        
        return {
            "avg_edges_decayed": avg_decayed,
            "avg_edges_pruned": avg_pruned,
            "edge_count_trend": trend,
            "current_decay_rate": self.decay_rate,
            "history_length": len(self.decay_history),
        }
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику."""
        return {
            **self.stats,
            "decay_rate": self.decay_rate,
            "min_weight": self.min_weight,
            "target_sparsity": self.target_sparsity,
        }
    
    def forward(
        self,
        graph,  # DynamicGraph
        activity_mask: Optional[torch.Tensor] = None,
        apply_homeostasis: bool = False
    ) -> Dict:
        """
        Прямой проход: мягкое затухание + pruning слабых рёбер.
        
        Контроль sparsity теперь в Hebbian (бюджет новых рёбер +
        конкурентное ослабление). Decay отвечает только за:
        - Мягкое затухание всех/неактивных рёбер
        - Pruning рёбер ниже min_weight
        - Опциональный гомеостаз (по умолчанию выключен)
        
        Args:
            graph: DynamicGraph для обновления
            activity_mask: маска активных связей
            apply_homeostasis: применять ли гомеостаз
            
        Returns:
            Статистика обновления
        """
        # Применяем затухание
        decay_stats = self.apply_decay(graph, activity_mask)
        
        # Применяем гомеостаз (опционально)
        homeostasis_stats = {}
        if apply_homeostasis:
            homeostasis_stats = self.apply_homeostasis(graph)
        
        return {
            **decay_stats,
            **homeostasis_stats,
        }

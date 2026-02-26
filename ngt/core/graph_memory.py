"""
Graph-Enhanced Associative Memory

Объединяет TemporalMemory с DynamicGraph для улучшенного recall.
Граф хранит ассоциации между паттернами, что помогает при восстановлении.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from collections import deque
import numpy as np

from .memory import TemporalMemory
from .graph import DynamicGraph
from .hebbian import HebbianPlasticityCore


class GraphEnhancedMemory(nn.Module):
    """
    Память с графовым усилением.
    
    Комбинирует:
    1. TemporalMemory - хранение паттернов
    2. DynamicGraph - ассоциации между паттернами
    3. HebbianPlasticityCore - обучение ассоциаций
    
    При recall использует граф для "подсказки" связанных паттернов.
    """
    
    def __init__(
        self,
        memory_size: int = 100,
        pattern_dim: int = 64,
        similarity_threshold: float = 0.3,
        graph_influence: float = 0.3,
        hopfield_beta: float = 10.0,
        hopfield_iterations: int = 8,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.pattern_dim = pattern_dim
        self.graph_influence = graph_influence
        self.hopfield_beta = hopfield_beta
        self.hopfield_iterations = hopfield_iterations
        self.device = device
        
        # Основная память
        self.memory = TemporalMemory(
            memory_size=memory_size,
            pattern_dim=pattern_dim,
            similarity_threshold=similarity_threshold,
            device=device
        )
        
        # Граф ассоциаций между паттернами
        self.association_graph = DynamicGraph(
            num_nodes=memory_size,
            hidden_dim=pattern_dim,
            edge_threshold=0.05,
            device=device
        )
        
        # Хеббовское обучение для графа
        self.hebbian = HebbianPlasticityCore(
            learning_rate=0.1,
            decay_rate=0.01,
            threshold=0.2,
            target_edge_ratio=0.05,
            device=device
        )
        
        # История последних активаций для обучения ассоциаций
        self.recent_activations: deque = deque(maxlen=5)
        
        self.stats = {
            "stores": 0,
            "recalls": 0,
            "graph_assists": 0,
        }
    
    def store(
        self,
        pattern: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Сохраняет паттерн и обновляет граф ассоциаций.
        """
        # Сохраняем в память
        idx = self.memory.store_pattern(pattern, metadata)
        
        # Обновляем граф ассоциаций
        if len(self.recent_activations) > 0:
            # Создаём связи с недавно активными паттернами
            for prev_idx in self.recent_activations:
                if prev_idx != idx and prev_idx < self.memory_size:
                    # Сила связи зависит от близости во времени
                    weight = 0.5
                    self.association_graph.add_edge(prev_idx, idx, weight)
                    self.association_graph.add_edge(idx, prev_idx, weight)
        
        # Обновляем историю
        self.recent_activations.append(idx)
        
        self.stats["stores"] += 1
        return idx
    
    def recall(
        self,
        query: torch.Tensor,
        top_k: int = 1
    ) -> List[Tuple[int, float, torch.Tensor]]:
        """
        Извлекает паттерн с использованием графа для усиления.
        
        Алгоритм:
        1. Hopfield dynamics для уточнения запроса
        2. Находим ближайшие паттерны
        3. Используем граф для поиска связанных паттернов
        4. Комбинируем результаты
        """
        if self.memory.num_stored == 0:
            return []
        
        # Нормализуем запрос
        query = query.detach().clone()
        if query.norm() > 0:
            query = query / query.norm()
        
        # Шаг 1: Hopfield dynamics
        refined_query = self._hopfield_with_graph(query)
        
        # Шаг 2: Поиск в памяти
        results = self.memory.retrieve_pattern(
            refined_query, 
            top_k=top_k,
            use_hopfield=False  # Уже применили
        )
        
        self.stats["recalls"] += 1
        
        return results
    
    def _hopfield_with_graph(self, query: torch.Tensor) -> torch.Tensor:
        """
        Hopfield dynamics с учётом графа ассоциаций.
        """
        x = query.clone()
        memory = self.memory.long_term_memory[:self.memory.num_stored]
        num_patterns = self.memory.num_stored
        
        if num_patterns == 0:
            return x
        
        # Получаем веса графа
        graph_weights = self._get_graph_weights(num_patterns)
        
        for iteration in range(self.hopfield_iterations):
            # Стандартное Hopfield сходство
            similarities = torch.mv(memory, x)
            
            # Усиливаем с помощью графа
            if graph_weights is not None and iteration > 0:
                # Находим текущий лучший паттерн
                best_idx = similarities.argmax().item()
                
                # Получаем веса связей от лучшего паттерна
                neighbor_weights = graph_weights[best_idx]
                
                # Добавляем влияние графа
                similarities = similarities + self.graph_influence * neighbor_weights
                
                self.stats["graph_assists"] += 1
            
            # Softmax с температурой
            attention = torch.softmax(self.hopfield_beta * similarities, dim=0)
            
            # Взвешенная комбинация
            x_new = torch.mv(memory.T, attention)
            
            # Нормализуем
            if x_new.norm() > 0:
                x_new = x_new / x_new.norm()
            
            # Проверяем сходимость
            if torch.dot(x, x_new).item() > 0.9999:
                break
            
            x = x_new
        
        return x
    
    def _get_graph_weights(self, num_patterns: int) -> Optional[torch.Tensor]:
        """
        Получает матрицу весов графа для паттернов.
        Использует кэшированную adjacency matrix.
        """
        if self.association_graph.get_edge_count() == 0:
            return None
        
        adj = self.association_graph.get_adjacency_matrix()
        return adj[:num_patterns, :num_patterns]
    
    def train_associations(self, pattern_indices: List[int]) -> None:
        """
        Обучает ассоциации между паттернами.
        
        Args:
            pattern_indices: индексы паттернов, которые связаны
        """
        # Создаём вектор активаций
        activations = torch.zeros(self.memory_size, device=self.device)
        for idx in pattern_indices:
            if idx < self.memory_size:
                activations[idx] = 1.0
        
        # Применяем Хеббовское обучение
        self.hebbian(self.association_graph, activations)
    
    def get_state(self) -> Dict:
        """Возвращает состояние памяти."""
        return {
            "memory": self.memory.get_memory_state(),
            "graph_edges": self.association_graph.get_edge_count(),
            "stats": self.stats,
        }


def test_graph_enhanced_memory():
    """Тест GraphEnhancedMemory."""
    print("Testing GraphEnhancedMemory...")
    
    mem = GraphEnhancedMemory(
        memory_size=50,
        pattern_dim=64,
        similarity_threshold=0.3,
        graph_influence=0.3
    )
    
    # Создаём и сохраняем паттерны
    patterns = []
    for i in range(10):
        p = torch.randn(64)
        p = p / p.norm()
        patterns.append(p)
        mem.store(p)
    
    # Тестируем recall
    for noise_level in [0.1, 0.3, 0.5]:
        correct = 0
        for i, original in enumerate(patterns):
            # Добавляем шум
            noisy = original + torch.randn(64) * noise_level
            noisy = noisy / noisy.norm()
            
            # Recall
            results = mem.recall(noisy, top_k=1)
            if results:
                _, sim, retrieved = results[0]
                if torch.dot(original, retrieved).item() > 0.9:
                    correct += 1
        
        print(f"  Noise {noise_level:.0%}: {correct}/10 correct ({correct*10}%)")
    
    print(f"  State: {mem.get_state()}")


if __name__ == "__main__":
    test_graph_enhanced_memory()

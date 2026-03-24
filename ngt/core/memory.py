"""
Temporal Memory Layer - Слой временной памяти NGT

Реализует механизм хранения и извлечения паттернов активации
для долговременной ассоциативной памяти.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
import numpy as np


class TemporalMemory(nn.Module):
    """
    Слой временной памяти для NGT.
    
    Хранит паттерны активации и позволяет:
    - Запоминать новые паттерны
    - Извлекать похожие паттерны по частичному запросу
    - Консолидировать память (переносить из краткосрочной в долгосрочную)
    
    Это аналог гиппокампа в мозге - структуры, отвечающей за
    формирование и консолидацию памяти.
    
    Attributes:
        memory_size: максимальное количество паттернов
        pattern_dim: размерность паттерна
        similarity_threshold: порог схожести для извлечения
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        pattern_dim: int = 64,
        similarity_threshold: float = 0.7,
        consolidation_threshold: int = 5,
        decay_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.pattern_dim = pattern_dim
        self.similarity_threshold = similarity_threshold
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate
        self.device = device
        
        # Краткосрочная память (working memory)
        self.short_term_memory: List[Dict] = []
        self.short_term_capacity = 50
        
        # Долгосрочная память (consolidated patterns)
        self.long_term_memory = nn.Parameter(
            torch.zeros(memory_size, pattern_dim, device=device),
            requires_grad=False
        )
        self.memory_usage = torch.zeros(memory_size, device=device)
        self.memory_strength = torch.zeros(memory_size, device=device)
        self.memory_access_count = torch.zeros(memory_size, dtype=torch.long, device=device)
        
        # Индекс следующей свободной ячейки
        self.next_free_idx = 0
        self.num_stored = 0
        
        # Статистика
        self.stats = {
            "patterns_stored": 0,
            "patterns_retrieved": 0,
            "consolidations": 0,
            "forgettings": 0,
        }
    
    def store_pattern(
        self,
        pattern: torch.Tensor,
        metadata: Optional[Dict] = None,
        strength: float = 1.0,
        immediate_consolidate: bool = True
    ) -> int:
        """
        Сохраняет паттерн в память.
        
        Args:
            pattern: паттерн для сохранения [pattern_dim]
            metadata: дополнительные данные (метки, контекст)
            strength: начальная сила паттерна
            immediate_consolidate: сразу сохранять в долгосрочную память
            
        Returns:
            Индекс сохранённого паттерна
        """
        # Нормализуем паттерн
        pattern = pattern.detach().clone()
        if pattern.norm() > 0:
            pattern = pattern / pattern.norm()
        
        # Если immediate_consolidate - сразу в долгосрочную память
        if immediate_consolidate:
            return self._store_to_long_term(pattern, metadata, strength)
        
        # Проверяем, есть ли похожий паттерн
        similar_idx = self._find_similar_pattern(pattern)
        
        if similar_idx >= 0:
            # Усиливаем существующий паттерн
            self._strengthen_pattern(similar_idx, strength)
            return similar_idx
        
        # Добавляем в краткосрочную память
        entry = {
            "pattern": pattern,
            "metadata": metadata or {},
            "strength": strength,
            "access_count": 1,
            "timestamp": len(self.short_term_memory),
        }
        
        self.short_term_memory.append(entry)
        self.stats["patterns_stored"] += 1
        
        # Проверяем переполнение краткосрочной памяти
        if len(self.short_term_memory) > self.short_term_capacity:
            self._consolidate_memory()
        
        return len(self.short_term_memory) - 1
    
    def retrieve_pattern(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        use_hopfield: bool = True,
        hopfield_iterations: int = 5,
        hopfield_beta: float = 8.0
    ) -> List[Tuple[int, float, torch.Tensor]]:
        """
        Извлекает похожие паттерны по запросу с Hopfield-like уточнением.
        
        Args:
            query: запрос [pattern_dim]
            top_k: количество возвращаемых паттернов
            use_hopfield: использовать итеративное уточнение
            hopfield_iterations: количество итераций уточнения
            hopfield_beta: температура softmax (выше = более резкий выбор)
            
        Returns:
            Список кортежей (индекс, схожесть, паттерн)
        """
        # Нормализуем запрос
        query = query.detach().clone()
        if query.norm() > 0:
            query = query / query.norm()
        
        # Hopfield-like итеративное уточнение
        if use_hopfield and self.num_stored > 0:
            query = self._hopfield_retrieve(query, hopfield_iterations, hopfield_beta)
        
        results = []
        
        # Поиск в краткосрочной памяти
        for i, entry in enumerate(self.short_term_memory):
            similarity = torch.dot(query, entry["pattern"]).item()
            if similarity > self.similarity_threshold:
                results.append((f"stm_{i}", similarity, entry["pattern"]))
                entry["access_count"] += 1
        
        # Поиск в долгосрочной памяти
        if self.num_stored > 0:
            # Вычисляем схожесть со всеми сохранёнными паттернами
            similarities = torch.mv(
                self.long_term_memory[:self.num_stored], 
                query
            )
            
            # Находим top-k
            k = min(top_k, self.num_stored)
            top_similarities, top_indices = torch.topk(similarities, k)
            
            for sim, idx in zip(top_similarities, top_indices):
                if sim.item() > self.similarity_threshold:
                    results.append((
                        f"ltm_{idx.item()}", 
                        sim.item(), 
                        self.long_term_memory[idx]
                    ))
                    self.memory_access_count[idx] += 1
        
        # Сортируем по схожести
        results.sort(key=lambda x: -x[1])
        
        self.stats["patterns_retrieved"] += 1
        
        return results[:top_k]
    
    def _hopfield_retrieve(
        self,
        query: torch.Tensor,
        iterations: int = 5,
        beta: float = 8.0
    ) -> torch.Tensor:
        """
        Hopfield-like итеративное уточнение запроса.
        
        Использует Modern Hopfield Network формулу:
        x_new = softmax(beta * X^T @ x) @ X
        
        где X - матрица сохранённых паттернов.
        
        Args:
            query: начальный запрос
            iterations: количество итераций
            beta: температура (выше = резче)
            
        Returns:
            Уточнённый запрос
        """
        x = query.clone()
        memory = self.long_term_memory[:self.num_stored]  # [N, D]
        
        for _ in range(iterations):
            # Вычисляем сходство с каждым паттерном
            similarities = torch.mv(memory, x)  # [N]
            
            # Softmax с температурой
            attention = torch.softmax(beta * similarities, dim=0)  # [N]
            
            # Взвешенная комбинация паттернов
            x_new = torch.mv(memory.T, attention)  # [D]
            
            # Нормализуем
            if x_new.norm() > 0:
                x_new = x_new / x_new.norm()
            
            # Проверяем сходимость
            if torch.dot(x, x_new).item() > 0.9999:
                break
            
            x = x_new
        
        return x
    
    def _find_similar_pattern(self, pattern: torch.Tensor) -> int:
        """Ищет похожий паттерн в памяти."""
        # Поиск в краткосрочной памяти
        for i, entry in enumerate(self.short_term_memory):
            similarity = torch.dot(pattern, entry["pattern"]).item()
            if similarity > 0.95:  # Очень похожий
                return i
        
        # Поиск в долгосрочной памяти
        if self.num_stored > 0:
            similarities = torch.mv(self.long_term_memory[:self.num_stored], pattern)
            max_sim, max_idx = similarities.max(dim=0)
            if max_sim.item() > 0.95:
                return max_idx.item() + self.short_term_capacity
        
        return -1
    
    def _strengthen_pattern(self, idx: int, strength: float) -> None:
        """Усиливает паттерн."""
        if idx < len(self.short_term_memory):
            self.short_term_memory[idx]["strength"] += strength
            self.short_term_memory[idx]["access_count"] += 1
        else:
            ltm_idx = idx - self.short_term_capacity
            if ltm_idx < self.num_stored:
                self.memory_strength[ltm_idx] += strength
                self.memory_access_count[ltm_idx] += 1
    
    def _store_to_long_term(
        self,
        pattern: torch.Tensor,
        metadata: Optional[Dict] = None,
        strength: float = 1.0
    ) -> int:
        """Сохраняет паттерн напрямую в долгосрочную память."""
        # Находим место
        if self.num_stored < self.memory_size:
            ltm_idx = self.num_stored
            self.num_stored += 1
        else:
            # Заменяем самый слабый
            ltm_idx = self.memory_strength[:self.num_stored].argmin().item()
        
        # Сохраняем
        self.long_term_memory.data[ltm_idx] = pattern
        self.memory_strength[ltm_idx] = strength
        self.memory_access_count[ltm_idx] = 1
        self.memory_usage[ltm_idx] = 1.0
        
        self.stats["patterns_stored"] += 1
        
        return ltm_idx
    
    def _consolidate_memory(self) -> None:
        """
        Консолидирует память: переносит сильные паттерны
        из краткосрочной в долгосрочную память.
        """
        # Сортируем по силе и частоте доступа
        scored_entries = [
            (i, entry["strength"] * np.log1p(entry["access_count"]))
            for i, entry in enumerate(self.short_term_memory)
        ]
        scored_entries.sort(key=lambda x: -x[1])
        
        # Переносим сильные паттерны в долгосрочную память
        consolidated = 0
        for idx, score in scored_entries:
            if score < self.consolidation_threshold:
                break
                
            entry = self.short_term_memory[idx]
            
            # Находим место в долгосрочной памяти
            if self.num_stored < self.memory_size:
                ltm_idx = self.num_stored
                self.num_stored += 1
            else:
                # Заменяем самый слабый паттерн
                ltm_idx = self.memory_strength[:self.num_stored].argmin().item()
            
            # Сохраняем
            self.long_term_memory.data[ltm_idx] = entry["pattern"]
            self.memory_strength[ltm_idx] = entry["strength"]
            self.memory_access_count[ltm_idx] = entry["access_count"]
            self.memory_usage[ltm_idx] = 1.0
            
            consolidated += 1
        
        # Очищаем краткосрочную память
        self.short_term_memory = [
            entry for i, entry in enumerate(self.short_term_memory)
            if scored_entries[i][1] >= self.consolidation_threshold
        ][-10:]  # Оставляем последние 10
        
        self.stats["consolidations"] += consolidated
    
    def apply_decay(self) -> int:
        """
        Применяет затухание к памяти.
        
        Returns:
            Количество забытых паттернов
        """
        forgotten = 0
        
        # Затухание в краткосрочной памяти
        for entry in self.short_term_memory:
            entry["strength"] *= (1 - self.decay_rate)
        
        # Удаляем слабые паттерны
        self.short_term_memory = [
            entry for entry in self.short_term_memory
            if entry["strength"] > 0.1
        ]
        
        # Затухание в долгосрочной памяти
        self.memory_strength[:self.num_stored] *= (1 - self.decay_rate * 0.1)
        
        # Забываем очень слабые паттерны
        weak_mask = self.memory_strength[:self.num_stored] < 0.01
        forgotten = weak_mask.sum().item()
        
        if forgotten > 0:
            self.memory_usage[:self.num_stored][weak_mask] = 0
            self.stats["forgettings"] += forgotten
        
        return forgotten
    
    def get_replay_batch(self, batch_size: int = 16, noise_std: float = 0.05) -> Optional[torch.Tensor]:
        """
        Hippocampal Replay: извлекает паттерны из долгосрочной памяти
        для консолидации при обучении на новых задачах.
        
        Биологическая аналогия: во сне гиппокамп «проигрывает» дневной
        опыт неокортексу, предотвращая забывание.
        
        Args:
            batch_size: количество паттернов для replay
            noise_std: шум для аугментации (предотвращает overfitting на replay)
            
        Returns:
            replay_patterns: [batch_size, pattern_dim] или None если память пуста
        """
        if self.num_stored == 0:
            return None
        
        # Выбираем паттерны пропорционально их силе (важные чаще)
        strengths = self.memory_strength[:self.num_stored]
        if strengths.sum() <= 0:
            return None
        
        probs = strengths / strengths.sum()
        
        k = min(batch_size, self.num_stored)
        indices = torch.multinomial(probs, k, replacement=(k > self.num_stored))
        
        patterns = self.long_term_memory[indices].clone()  # [k, pattern_dim]
        
        # Добавляем небольшой шум (аугментация, как в мозге — replay не точная копия)
        if noise_std > 0:
            patterns = patterns + torch.randn_like(patterns) * noise_std
        
        # Обновляем счётчик доступа
        self.memory_access_count[indices] += 1
        
        return patterns

    def get_memory_state(self) -> Dict:
        """Возвращает состояние памяти."""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": self.num_stored,
            "total_capacity": self.memory_size,
            "utilization": self.num_stored / self.memory_size,
            "avg_strength": self.memory_strength[:self.num_stored].mean().item() if self.num_stored > 0 else 0,
        }
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику."""
        return {
            **self.stats,
            **self.get_memory_state(),
        }
    
    def forward(
        self,
        x: torch.Tensor,
        store: bool = True
    ) -> Tuple[torch.Tensor, List]:
        """
        Прямой проход: сохраняет паттерн и извлекает похожие.
        
        Args:
            x: входной паттерн [batch, pattern_dim] или [pattern_dim]
            store: сохранять ли паттерн
            
        Returns:
            retrieved: извлечённые паттерны
            indices: индексы извлечённых паттернов
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        all_retrieved = []
        
        for pattern in x:
            if store:
                self.store_pattern(pattern)
            
            retrieved = self.retrieve_pattern(pattern)
            all_retrieved.append(retrieved)
        
        return all_retrieved

"""
Hierarchical Memory System — Иерархическая система памяти NGT

Биологическая аналогия: мозг хранит информацию на нескольких уровнях,
от мгновенных сенсорных следов до долговременных семантических знаний.

4 уровня иерархии:
1. Sensory Buffer — мгновенные следы (~100ms в мозге), FIFO, без фильтрации
2. Working Memory — активная обработка (4-7 объектов), importance-gated
3. Episodic Memory — конкретные эпизоды обучения (гиппокамп), domain-tagged
4. Semantic Memory — обобщённые знания (неокортекс), прототипы классов/доменов

Механизмы:
- Importance-based promotion: важные паттерны поднимаются вверх
- Semantic compression: эпизоды → прототипы (centroid per class/domain)
- Level-dependent forgetting: нижние уровни забывают быстрее
- Dream-driven consolidation: episodic → semantic через Dream Phase
- Hierarchical replay: replay из разных уровней с разной частотой
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import deque
import math


class SensoryBuffer:
    """
    Уровень 1: Сенсорный буфер — мгновенные следы.
    
    FIFO очередь фиксированного размера. Все входящие данные
    попадают сюда без фильтрации. Быстрое затухание.
    
    Биоаналогия: иконическая/эхоическая память (~250ms).
    """
    
    def __init__(self, capacity: int = 64, pattern_dim: int = 64):
        self.capacity = capacity
        self.pattern_dim = pattern_dim
        self._buffer: deque = deque(maxlen=capacity)
    
    def store(self, pattern: torch.Tensor, metadata: Optional[Dict] = None) -> None:
        """Сохраняет паттерн в сенсорный буфер (FIFO)."""
        entry = {
            "pattern": pattern.detach().cpu(),
            "metadata": metadata or {},
            "strength": 1.0,
        }
        self._buffer.append(entry)
    
    def get_recent(self, n: int = 16) -> List[Dict]:
        """Возвращает n последних записей."""
        return list(self._buffer)[-n:]
    
    def get_strong(self, threshold: float = 0.5) -> List[Dict]:
        """Возвращает записи с силой выше порога (для promotion)."""
        return [e for e in self._buffer if e["strength"] >= threshold]
    
    def apply_decay(self, rate: float = 0.3) -> None:
        """Быстрое затухание — сенсорные следы исчезают быстро."""
        for entry in self._buffer:
            entry["strength"] *= (1.0 - rate)
    
    def clear(self) -> None:
        self._buffer.clear()
    
    @property
    def size(self) -> int:
        return len(self._buffer)


class WorkingMemory:
    """
    Уровень 2: Рабочая память — активная обработка.
    
    Ограниченная ёмкость (как 4-7 объектов в мозге).
    Importance-gated: только важные паттерны попадают сюда.
    Используется для текущей задачи.
    
    Биоаналогия: префронтальная кора, рабочая память Баддели.
    """
    
    def __init__(self, capacity: int = 32, pattern_dim: int = 64):
        self.capacity = capacity
        self.pattern_dim = pattern_dim
        self._slots: List[Dict] = []
    
    def store(
        self,
        pattern: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Сохраняет паттерн если он достаточно важен.
        
        Returns:
            True если паттерн был сохранён, False если отвергнут.
        """
        entry = {
            "pattern": pattern.detach().cpu(),
            "importance": importance,
            "access_count": 1,
            "metadata": metadata or {},
        }
        
        if len(self._slots) < self.capacity:
            self._slots.append(entry)
            return True
        
        # Заменяем наименее важный если новый важнее
        min_idx = min(range(len(self._slots)),
                      key=lambda i: self._slots[i]["importance"])
        if importance > self._slots[min_idx]["importance"]:
            self._slots[min_idx] = entry
            return True
        
        return False
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Извлекает top-k похожих паттернов."""
        if not self._slots:
            return []
        
        query_norm = query.detach().cpu()
        if query_norm.norm() > 0:
            query_norm = query_norm / query_norm.norm()
        
        results = []
        for entry in self._slots:
            p = entry["pattern"]
            if p.norm() > 0:
                p = p / p.norm()
            sim = torch.dot(query_norm.view(-1)[:p.shape[0]],
                          p.view(-1)[:query_norm.shape[0]]).item()
            results.append((sim, entry))
            entry["access_count"] += 1
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    def get_promotable(self, access_threshold: int = 3, importance_threshold: float = 0.5) -> List[Dict]:
        """Возвращает паттерны готовые к promotion в episodic memory."""
        return [
            e for e in self._slots
            if e["access_count"] >= access_threshold
            and e["importance"] >= importance_threshold
        ]
    
    def apply_decay(self, rate: float = 0.1) -> int:
        """Умеренное затухание. Возвращает количество забытых."""
        for entry in self._slots:
            entry["importance"] *= (1.0 - rate)
        
        before = len(self._slots)
        self._slots = [e for e in self._slots if e["importance"] > 0.05]
        return before - len(self._slots)
    
    def clear(self) -> None:
        self._slots.clear()
    
    @property
    def size(self) -> int:
        return len(self._slots)


class EpisodicMemory:
    """
    Уровень 3: Эпизодическая память — конкретные эпизоды обучения.
    
    Хранит паттерны с контекстом (задача, домен, метки).
    Основной источник для replay. Domain-tagged.
    
    Биоаналогия: гиппокамп — хранит «что, где, когда».
    """
    
    def __init__(self, capacity: int = 500, pattern_dim: int = 64):
        self.capacity = capacity
        self.pattern_dim = pattern_dim
        
        # Тензорное хранение для быстрого поиска
        self._patterns = torch.zeros(capacity, pattern_dim)
        self._strengths = torch.zeros(capacity)
        self._access_counts = torch.zeros(capacity, dtype=torch.long)
        self._metadata: List[Optional[Dict]] = [None] * capacity
        self._num_stored = 0
    
    def store(
        self,
        pattern: torch.Tensor,
        strength: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Сохраняет эпизод. Возвращает индекс."""
        p = pattern.detach().cpu().view(-1)
        if p.shape[0] != self.pattern_dim:
            # Pad или truncate
            if p.shape[0] < self.pattern_dim:
                p = F.pad(p, (0, self.pattern_dim - p.shape[0]))
            else:
                p = p[:self.pattern_dim]
        
        if p.norm() > 0:
            p = p / p.norm()
        
        if self._num_stored < self.capacity:
            idx = self._num_stored
            self._num_stored += 1
        else:
            # Заменяем самый слабый
            idx = self._strengths[:self._num_stored].argmin().item()
        
        self._patterns[idx] = p
        self._strengths[idx] = strength
        self._access_counts[idx] = 1
        self._metadata[idx] = metadata or {}
        
        return idx
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float, torch.Tensor, Dict]]:
        """Извлекает top-k похожих эпизодов."""
        if self._num_stored == 0:
            return []
        
        q = query.detach().cpu().view(-1)
        if q.shape[0] != self.pattern_dim:
            if q.shape[0] < self.pattern_dim:
                q = F.pad(q, (0, self.pattern_dim - q.shape[0]))
            else:
                q = q[:self.pattern_dim]
        if q.norm() > 0:
            q = q / q.norm()
        
        sims = torch.mv(self._patterns[:self._num_stored], q)
        k = min(top_k, self._num_stored)
        top_sims, top_idx = torch.topk(sims, k)
        
        results = []
        for sim, idx in zip(top_sims, top_idx):
            i = idx.item()
            self._access_counts[i] += 1
            results.append((i, sim.item(), self._patterns[i].clone(), self._metadata[i] or {}))
        
        return results
    
    def get_by_domain(self, domain: str) -> List[Tuple[int, torch.Tensor, Dict]]:
        """Возвращает все эпизоды определённого домена."""
        results = []
        for i in range(self._num_stored):
            meta = self._metadata[i]
            if meta and meta.get("domain") == domain:
                results.append((i, self._patterns[i].clone(), meta))
        return results
    
    def get_by_task(self, task: str) -> List[Tuple[int, torch.Tensor, Dict]]:
        """Возвращает все эпизоды определённой задачи."""
        results = []
        for i in range(self._num_stored):
            meta = self._metadata[i]
            if meta and meta.get("task") == task:
                results.append((i, self._patterns[i].clone(), meta))
        return results
    
    def get_consolidation_candidates(
        self,
        min_access: int = 2,
        min_strength: float = 0.3,
    ) -> List[Tuple[int, torch.Tensor, Dict]]:
        """Возвращает эпизоды готовые к консолидации в semantic memory."""
        results = []
        for i in range(self._num_stored):
            if (self._access_counts[i] >= min_access
                    and self._strengths[i] >= min_strength):
                results.append((i, self._patterns[i].clone(), self._metadata[i] or {}))
        return results
    
    def replay_batch(
        self,
        batch_size: int = 16,
        noise_std: float = 0.05,
        domain: Optional[str] = None,
    ) -> Optional[Tuple[torch.Tensor, List[Dict]]]:
        """
        Hippocampal replay: извлекает батч для replay.
        
        Returns:
            (patterns, metadata_list) или None
        """
        if self._num_stored == 0:
            return None
        
        # Фильтруем по домену если указан
        if domain is not None:
            valid_idx = [
                i for i in range(self._num_stored)
                if self._metadata[i] and self._metadata[i].get("domain") == domain
            ]
            if not valid_idx:
                valid_idx = list(range(self._num_stored))
        else:
            valid_idx = list(range(self._num_stored))
        
        # Strength-weighted sampling
        strengths = self._strengths[valid_idx]
        if strengths.sum() <= 0:
            return None
        
        probs = strengths / strengths.sum()
        k = min(batch_size, len(valid_idx))
        sampled = torch.multinomial(probs, k, replacement=(k > len(valid_idx)))
        
        indices = [valid_idx[s] for s in sampled]
        patterns = self._patterns[indices].clone()
        
        if noise_std > 0:
            patterns = patterns + torch.randn_like(patterns) * noise_std
        
        metadata_list = [self._metadata[i] or {} for i in indices]
        self._access_counts[indices] += 1
        
        return patterns, metadata_list
    
    def apply_decay(self, rate: float = 0.02) -> int:
        """Медленное затухание. Возвращает количество забытых."""
        if self._num_stored == 0:
            return 0
        
        self._strengths[:self._num_stored] *= (1.0 - rate)
        
        weak = self._strengths[:self._num_stored] < 0.01
        forgotten = weak.sum().item()
        
        return int(forgotten)
    
    @property
    def size(self) -> int:
        return self._num_stored


class SemanticMemory:
    """
    Уровень 4: Семантическая память — обобщённые знания.
    
    Хранит прототипы (centroids) для каждого класса/домена.
    Самый стабильный уровень — почти не забывает.
    Используется для обобщения и transfer learning.
    
    Биоаналогия: неокортекс — медленно обучается, долго помнит.
    """
    
    def __init__(self, max_prototypes: int = 200, pattern_dim: int = 64):
        self.max_prototypes = max_prototypes
        self.pattern_dim = pattern_dim
        
        # Прототипы: key = (domain, class_label)
        self._prototypes: Dict[Tuple[str, int], Dict] = {}
        
        # Тензорный кэш для быстрого поиска
        self._cache_valid = False
        self._cache_patterns: Optional[torch.Tensor] = None
        self._cache_keys: List[Tuple[str, int]] = []
    
    def update_prototype(
        self,
        domain: str,
        class_label: int,
        pattern: torch.Tensor,
        n_samples: int = 1,
    ) -> None:
        """
        Обновляет прототип класса (running mean).
        
        Если прототип уже существует — обновляет как running average.
        Если нет — создаёт новый.
        """
        key = (domain, class_label)
        p = pattern.detach().cpu().view(-1)
        if p.shape[0] != self.pattern_dim:
            if p.shape[0] < self.pattern_dim:
                p = F.pad(p, (0, self.pattern_dim - p.shape[0]))
            else:
                p = p[:self.pattern_dim]
        
        if key in self._prototypes:
            proto = self._prototypes[key]
            old_n = proto["n_samples"]
            new_n = old_n + n_samples
            # Running mean: new_mean = (old_mean * old_n + new_pattern * n_samples) / new_n
            proto["pattern"] = (proto["pattern"] * old_n + p * n_samples) / new_n
            proto["n_samples"] = new_n
            proto["access_count"] += 1
        else:
            if len(self._prototypes) >= self.max_prototypes:
                # Удаляем наименее используемый прототип
                min_key = min(self._prototypes, key=lambda k: self._prototypes[k]["access_count"])
                del self._prototypes[min_key]
            
            self._prototypes[key] = {
                "pattern": p.clone(),
                "n_samples": n_samples,
                "access_count": 1,
                "domain": domain,
                "class_label": class_label,
            }
        
        self._cache_valid = False
    
    def get_prototype(self, domain: str, class_label: int) -> Optional[torch.Tensor]:
        """Возвращает прототип для класса."""
        key = (domain, class_label)
        if key in self._prototypes:
            self._prototypes[key]["access_count"] += 1
            return self._prototypes[key]["pattern"].clone()
        return None
    
    def get_domain_prototypes(self, domain: str) -> List[Dict]:
        """Возвращает все прототипы домена."""
        results = []
        for key, proto in self._prototypes.items():
            if key[0] == domain:
                results.append({
                    "class_label": key[1],
                    "pattern": proto["pattern"].clone(),
                    "n_samples": proto["n_samples"],
                })
        return results
    
    def find_nearest_prototype(
        self,
        query: torch.Tensor,
        domain: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[Tuple[str, int], float, torch.Tensor]]:
        """Находит ближайшие прототипы к запросу."""
        self._rebuild_cache()
        
        if self._cache_patterns is None or len(self._cache_keys) == 0:
            return []
        
        q = query.detach().cpu().view(-1)
        if q.shape[0] != self.pattern_dim:
            if q.shape[0] < self.pattern_dim:
                q = F.pad(q, (0, self.pattern_dim - q.shape[0]))
            else:
                q = q[:self.pattern_dim]
        if q.norm() > 0:
            q = q / q.norm()
        
        # Фильтруем по домену
        if domain is not None:
            valid_mask = torch.tensor([k[0] == domain for k in self._cache_keys])
            if not valid_mask.any():
                valid_mask = torch.ones(len(self._cache_keys), dtype=torch.bool)
        else:
            valid_mask = torch.ones(len(self._cache_keys), dtype=torch.bool)
        
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return []
        
        patterns = self._cache_patterns[valid_idx]
        # Нормализуем прототипы
        norms = patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)
        patterns_norm = patterns / norms
        
        sims = torch.mv(patterns_norm, q)
        k = min(top_k, len(valid_idx))
        top_sims, top_local_idx = torch.topk(sims, k)
        
        results = []
        for sim, local_i in zip(top_sims, top_local_idx):
            global_i = valid_idx[local_i].item()
            key = self._cache_keys[global_i]
            results.append((key, sim.item(), self._prototypes[key]["pattern"].clone()))
        
        return results
    
    def _rebuild_cache(self) -> None:
        """Перестраивает тензорный кэш."""
        if self._cache_valid:
            return
        
        if not self._prototypes:
            self._cache_patterns = None
            self._cache_keys = []
            self._cache_valid = True
            return
        
        self._cache_keys = list(self._prototypes.keys())
        self._cache_patterns = torch.stack([
            self._prototypes[k]["pattern"] for k in self._cache_keys
        ])
        self._cache_valid = True
    
    def apply_decay(self, rate: float = 0.001) -> int:
        """Очень медленное затухание — семантическая память стабильна."""
        forgotten = 0
        keys_to_remove = []
        for key, proto in self._prototypes.items():
            proto["n_samples"] = max(1, proto["n_samples"] - 1)
            if proto["access_count"] <= 0 and proto["n_samples"] <= 1:
                keys_to_remove.append(key)
                forgotten += 1
        
        for key in keys_to_remove:
            del self._prototypes[key]
        
        if forgotten > 0:
            self._cache_valid = False
        
        return forgotten
    
    @property
    def size(self) -> int:
        return len(self._prototypes)
    
    @property
    def domains(self) -> List[str]:
        return list(set(k[0] for k in self._prototypes.keys()))


class HierarchicalMemory(nn.Module):
    """
    Иерархическая система памяти NGT.
    
    Оркестрирует 4 уровня памяти:
    1. Sensory Buffer (мгновенная, FIFO)
    2. Working Memory (активная, importance-gated)
    3. Episodic Memory (эпизоды, domain-tagged)
    4. Semantic Memory (обобщения, прототипы)
    
    Механизмы:
    - Automatic promotion: паттерны поднимаются по иерархии
    - Level-dependent forgetting: нижние уровни забывают быстрее
    - Hierarchical replay: replay из разных уровней
    - Dream consolidation: episodic → semantic
    
    Args:
        pattern_dim: размерность паттернов
        sensory_capacity: ёмкость сенсорного буфера
        working_capacity: ёмкость рабочей памяти
        episodic_capacity: ёмкость эпизодической памяти
        max_prototypes: максимум прототипов в семантической памяти
        promotion_threshold: порог importance для promotion
        consolidation_interval: интервал автоматической консолидации (в шагах)
    """
    
    def __init__(
        self,
        pattern_dim: int = 64,
        sensory_capacity: int = 64,
        working_capacity: int = 32,
        episodic_capacity: int = 500,
        max_prototypes: int = 200,
        promotion_threshold: float = 0.5,
        consolidation_interval: int = 50,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        self.promotion_threshold = promotion_threshold
        self.consolidation_interval = consolidation_interval
        
        # 4 уровня памяти
        self.sensory = SensoryBuffer(sensory_capacity, pattern_dim)
        self.working = WorkingMemory(working_capacity, pattern_dim)
        self.episodic = EpisodicMemory(episodic_capacity, pattern_dim)
        self.semantic = SemanticMemory(max_prototypes, pattern_dim)
        
        # Importance estimator: оценивает важность паттерна
        # на основе новизны (cosine distance от ближайшего прототипа)
        self._step_count = 0
        
        # Статистика
        self.stats = {
            "total_stored": 0,
            "promotions_sensory_to_working": 0,
            "promotions_working_to_episodic": 0,
            "consolidations_episodic_to_semantic": 0,
            "total_forgotten": 0,
            "total_replays": 0,
        }
    
    def store(
        self,
        pattern: torch.Tensor,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        class_label: Optional[int] = None,
        importance: Optional[float] = None,
    ) -> Dict:
        """
        Сохраняет паттерн в иерархическую память.
        
        Паттерн сначала попадает в sensory buffer, затем
        автоматически продвигается вверх по иерархии.
        
        Args:
            pattern: паттерн для сохранения
            domain: домен данных (vision, tabular, text, synthetic)
            task: имя задачи
            class_label: метка класса (для semantic memory)
            importance: явная важность (если None — вычисляется автоматически)
            
        Returns:
            stats: статистика сохранения
        """
        store_stats = {
            "level": "sensory",
            "importance": 0.0,
            "promoted": False,
        }
        
        metadata = {
            "domain": domain or "default",
            "task": task,
            "class_label": class_label,
            "step": self._step_count,
        }
        
        # 1. Всегда сохраняем в sensory buffer
        self.sensory.store(pattern, metadata)
        self.stats["total_stored"] += 1
        
        # 2. Вычисляем importance
        if importance is None:
            importance = self._estimate_importance(pattern, domain)
        store_stats["importance"] = importance
        
        # 3. Если важность выше порога — сразу в working memory
        if importance >= self.promotion_threshold:
            stored = self.working.store(pattern, importance, metadata)
            if stored:
                store_stats["level"] = "working"
                self.stats["promotions_sensory_to_working"] += 1
        
        # 4. Если есть class_label — обновляем semantic memory
        if class_label is not None and domain is not None:
            self.semantic.update_prototype(domain, class_label, pattern)
        
        # 5. Периодическая консолидация
        self._step_count += 1
        if self._step_count % self.consolidation_interval == 0:
            self._auto_consolidate()
        
        return store_stats
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        levels: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> List[Dict]:
        """
        Иерархический поиск: ищет во всех уровнях памяти.
        
        Args:
            query: запрос
            top_k: количество результатов
            levels: уровни для поиска (по умолчанию все)
            domain: фильтр по домену
            
        Returns:
            Список результатов с указанием уровня
        """
        if levels is None:
            levels = ["semantic", "episodic", "working"]
        
        results = []
        
        # Semantic memory — самый быстрый (прототипы)
        if "semantic" in levels:
            sem_results = self.semantic.find_nearest_prototype(
                query, domain=domain, top_k=top_k
            )
            for key, sim, pattern in sem_results:
                results.append({
                    "level": "semantic",
                    "similarity": sim,
                    "pattern": pattern,
                    "domain": key[0],
                    "class_label": key[1],
                })
        
        # Episodic memory — конкретные эпизоды
        if "episodic" in levels:
            ep_results = self.episodic.retrieve(query, top_k=top_k)
            for idx, sim, pattern, meta in ep_results:
                if domain is not None and meta.get("domain") != domain:
                    continue
                results.append({
                    "level": "episodic",
                    "similarity": sim,
                    "pattern": pattern,
                    "domain": meta.get("domain"),
                    "class_label": meta.get("class_label"),
                    "task": meta.get("task"),
                })
        
        # Working memory — текущий контекст
        if "working" in levels:
            wm_results = self.working.retrieve(query, top_k=top_k)
            for sim, entry in wm_results:
                meta = entry.get("metadata", {})
                if domain is not None and meta.get("domain") != domain:
                    continue
                results.append({
                    "level": "working",
                    "similarity": sim,
                    "pattern": entry["pattern"],
                    "domain": meta.get("domain"),
                    "class_label": meta.get("class_label"),
                })
        
        # Сортируем по similarity
        results.sort(key=lambda x: -x["similarity"])
        self.stats["total_replays"] += 1
        
        return results[:top_k]
    
    def hierarchical_replay(
        self,
        batch_size: int = 32,
        noise_std: float = 0.05,
        domain: Optional[str] = None,
        semantic_ratio: float = 0.3,
        episodic_ratio: float = 0.5,
        working_ratio: float = 0.2,
    ) -> Optional[Tuple[torch.Tensor, List[Dict]]]:
        """
        Иерархический replay: смешивает примеры из разных уровней.
        
        Semantic (30%): прототипы — стабильные обобщения
        Episodic (50%): конкретные эпизоды — основной replay
        Working (20%): свежие паттерны — текущий контекст
        
        Returns:
            (patterns, metadata_list) или None
        """
        all_patterns = []
        all_metadata = []
        
        # Semantic replay
        n_semantic = max(1, int(batch_size * semantic_ratio))
        sem_protos = self.semantic.get_domain_prototypes(domain or "default")
        if not sem_protos and domain:
            # Fallback: все домены
            for d in self.semantic.domains:
                sem_protos.extend(self.semantic.get_domain_prototypes(d))
        
        if sem_protos:
            indices = torch.randint(0, len(sem_protos), (min(n_semantic, len(sem_protos)),))
            for i in indices:
                proto = sem_protos[i.item()]
                p = proto["pattern"].clone()
                if noise_std > 0:
                    p = p + torch.randn_like(p) * noise_std
                all_patterns.append(p)
                all_metadata.append({
                    "level": "semantic",
                    "domain": domain or "default",
                    "class_label": proto["class_label"],
                })
        
        # Episodic replay
        n_episodic = max(1, int(batch_size * episodic_ratio))
        ep_result = self.episodic.replay_batch(n_episodic, noise_std, domain)
        if ep_result is not None:
            patterns, meta_list = ep_result
            for i in range(patterns.shape[0]):
                all_patterns.append(patterns[i])
                all_metadata.append({**meta_list[i], "level": "episodic"})
        
        # Working memory replay
        n_working = max(1, int(batch_size * working_ratio))
        if self.working.size > 0:
            wm_entries = self.working._slots[:n_working]
            for entry in wm_entries:
                p = entry["pattern"].clone()
                if noise_std > 0:
                    p = p + torch.randn_like(p) * noise_std
                all_patterns.append(p)
                meta = entry.get("metadata", {})
                all_metadata.append({**meta, "level": "working"})
        
        if not all_patterns:
            return None
        
        # Pad все паттерны до одинакового размера
        max_dim = max(p.shape[0] for p in all_patterns)
        padded = []
        for p in all_patterns:
            if p.shape[0] < max_dim:
                p = F.pad(p, (0, max_dim - p.shape[0]))
            padded.append(p)
        
        patterns_tensor = torch.stack(padded)
        self.stats["total_replays"] += 1
        
        return patterns_tensor, all_metadata
    
    def consolidate(
        self,
        min_access: int = 2,
        min_strength: float = 0.3,
    ) -> Dict:
        """
        Консолидация: переносит знания вверх по иерархии.
        
        1. Working → Episodic: часто используемые паттерны
        2. Episodic → Semantic: группировка в прототипы
        
        Returns:
            stats: статистика консолидации
        """
        consolidation_stats = {
            "working_to_episodic": 0,
            "episodic_to_semantic": 0,
        }
        
        # 1. Working → Episodic
        promotable = self.working.get_promotable(
            access_threshold=min_access,
            importance_threshold=min_strength,
        )
        for entry in promotable:
            meta = entry.get("metadata", {})
            self.episodic.store(
                entry["pattern"],
                strength=entry["importance"],
                metadata=meta,
            )
            consolidation_stats["working_to_episodic"] += 1
            self.stats["promotions_working_to_episodic"] += 1
        
        # 2. Episodic → Semantic (группировка по domain+class)
        candidates = self.episodic.get_consolidation_candidates(
            min_access=min_access,
            min_strength=min_strength,
        )
        
        # Группируем по (domain, class_label)
        groups: Dict[Tuple[str, int], List[torch.Tensor]] = {}
        for idx, pattern, meta in candidates:
            domain = meta.get("domain", "default")
            class_label = meta.get("class_label")
            if class_label is not None:
                key = (domain, class_label)
                if key not in groups:
                    groups[key] = []
                groups[key].append(pattern)
        
        # Обновляем прототипы
        for (domain, class_label), patterns in groups.items():
            centroid = torch.stack(patterns).mean(dim=0)
            self.semantic.update_prototype(
                domain, class_label, centroid, n_samples=len(patterns)
            )
            consolidation_stats["episodic_to_semantic"] += 1
            self.stats["consolidations_episodic_to_semantic"] += 1
        
        return consolidation_stats
    
    def dream_consolidate(self) -> Dict:
        """
        Dream-driven консолидация: агрессивная версия consolidate().
        
        Вызывается во время Dream Phase. Переносит больше знаний
        из episodic в semantic, снижает пороги.
        
        Returns:
            stats: статистика
        """
        # Агрессивная консолидация с низкими порогами
        stats = self.consolidate(min_access=1, min_strength=0.1)
        
        # Дополнительно: все эпизоды с class_label → semantic
        for i in range(self.episodic.size):
            meta = self.episodic._metadata[i]
            if meta and meta.get("class_label") is not None:
                domain = meta.get("domain", "default")
                self.semantic.update_prototype(
                    domain,
                    meta["class_label"],
                    self.episodic._patterns[i],
                    n_samples=1,
                )
        
        stats["dream"] = True
        return stats
    
    def apply_decay(self) -> Dict:
        """
        Применяет level-dependent forgetting.
        
        Нижние уровни забывают быстрее, верхние — медленнее.
        """
        decay_stats = {}
        
        # Sensory: быстрое затухание (30%)
        self.sensory.apply_decay(rate=0.3)
        decay_stats["sensory_size"] = self.sensory.size
        
        # Working: умеренное затухание (10%)
        wm_forgotten = self.working.apply_decay(rate=0.1)
        decay_stats["working_forgotten"] = wm_forgotten
        decay_stats["working_size"] = self.working.size
        
        # Episodic: медленное затухание (2%)
        ep_forgotten = self.episodic.apply_decay(rate=0.02)
        decay_stats["episodic_forgotten"] = ep_forgotten
        decay_stats["episodic_size"] = self.episodic.size
        
        # Semantic: очень медленное затухание (0.1%)
        sem_forgotten = self.semantic.apply_decay(rate=0.001)
        decay_stats["semantic_forgotten"] = sem_forgotten
        decay_stats["semantic_size"] = self.semantic.size
        
        total_forgotten = wm_forgotten + ep_forgotten + sem_forgotten
        self.stats["total_forgotten"] += total_forgotten
        decay_stats["total_forgotten"] = total_forgotten
        
        return decay_stats
    
    def _estimate_importance(
        self,
        pattern: torch.Tensor,
        domain: Optional[str] = None,
    ) -> float:
        """
        Оценивает важность паттерна на основе новизны.
        
        Новизна = 1 - max_similarity с существующими прототипами.
        Новые паттерны (далёкие от прототипов) более важны.
        """
        if self.semantic.size == 0:
            return 1.0  # Всё новое если нет прототипов
        
        nearest = self.semantic.find_nearest_prototype(
            pattern, domain=domain, top_k=1
        )
        
        if not nearest:
            return 1.0
        
        max_sim = nearest[0][1]
        # Новизна: чем дальше от прототипа, тем важнее
        novelty = 1.0 - max(0.0, max_sim)
        
        # Масштабируем в [0.1, 1.0]
        importance = 0.1 + 0.9 * novelty
        return importance
    
    def _auto_consolidate(self) -> None:
        """Автоматическая консолидация по интервалу."""
        # Promote из sensory в working
        strong = self.sensory.get_strong(threshold=0.5)
        for entry in strong:
            self.working.store(
                entry["pattern"],
                importance=entry["strength"],
                metadata=entry.get("metadata"),
            )
        
        # Consolidate working → episodic → semantic
        self.consolidate(min_access=2, min_strength=0.3)
        
        # Apply decay
        self.apply_decay()
    
    def get_state(self) -> Dict:
        """Возвращает полное состояние иерархической памяти."""
        return {
            "sensory_size": self.sensory.size,
            "working_size": self.working.size,
            "episodic_size": self.episodic.size,
            "semantic_size": self.semantic.size,
            "semantic_domains": self.semantic.domains,
            "total_stored": self.stats["total_stored"],
            "promotions": {
                "sensory_to_working": self.stats["promotions_sensory_to_working"],
                "working_to_episodic": self.stats["promotions_working_to_episodic"],
                "episodic_to_semantic": self.stats["consolidations_episodic_to_semantic"],
            },
            "total_forgotten": self.stats["total_forgotten"],
            "step": self._step_count,
        }
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику."""
        return {**self.stats, **self.get_state()}

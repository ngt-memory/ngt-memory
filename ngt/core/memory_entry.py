"""
MemoryEntry — запись в памяти LLM (один фрагмент текста с embedding).
"""

import time
from typing import Optional, Dict, List

import torch


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

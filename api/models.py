"""
Pydantic схемы для NGT Memory API.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="Сообщение пользователя", min_length=1, max_length=8000)
    session_id: str = Field(default="default", description="ID сессии (изолированная память)")
    use_memory: bool = Field(default=True, description="Использовать NGT Memory")
    stream: bool = Field(default=False, description="Стриминг ответа (не реализован в v1)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "My name is Anton and I'm allergic to penicillin.",
                "session_id": "user_42",
                "use_memory": True
            }
        }


class StoreRequest(BaseModel):
    text: str = Field(..., description="Текст для сохранения в память", min_length=1, max_length=8000)
    session_id: str = Field(default="default", description="ID сессии")
    concepts: Optional[List[str]] = Field(default=None, description="Концепты (авто если None)")
    domain: Optional[str] = Field(default="general", description="Домен знаний")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Произвольные метаданные")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient has penicillin allergy and takes lisinopril 10mg daily.",
                "session_id": "medical_session",
                "domain": "medical"
            }
        }


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос", min_length=1, max_length=8000)
    session_id: str = Field(default="default", description="ID сессии")
    top_k: int = Field(default=5, ge=1, le=50, description="Количество результатов")
    use_graph: bool = Field(default=True, description="Использовать graph retrieval")
    threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Минимальный cosine score")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What medications should I avoid?",
                "session_id": "medical_session",
                "top_k": 5
            }
        }


class NewSessionRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии для сброса")


# ── Response models ───────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    text: str
    score: float
    domain: Optional[str] = None
    concepts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    memories_used: List[MemoryItem]
    memories_count: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    memory_entries: int


class StoreResponse(BaseModel):
    success: bool
    session_id: str
    memory_entries: int
    message: str


class RetrieveResponse(BaseModel):
    results: List[MemoryItem]
    count: int
    session_id: str
    query: str


class SessionStatsResponse(BaseModel):
    session_id: str
    memory_entries: int
    graph_edges: int
    graph_concepts: int
    total_turns: int
    total_memories_used: int
    avg_memories_per_turn: float
    total_tokens_in: int
    total_tokens_out: int
    avg_embed_ms: float
    avg_retrieve_ms: float
    avg_chat_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    active_sessions: int
    model: str
    embedding_model: str


class ResetResponse(BaseModel):
    success: bool
    session_id: str
    message: str

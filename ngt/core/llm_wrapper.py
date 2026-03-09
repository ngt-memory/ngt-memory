"""
NGTMemoryLLMWrapper — обёртка NGT Memory для реального LLM (OpenAI API).

Использование:
    from ngt.core.llm_wrapper import NGTMemoryLLMWrapper

    wrapper = NGTMemoryLLMWrapper(
        openai_api_key="sk-...",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        memory_top_k=5,
    )

    response = wrapper.chat("My name is Anton and I'm allergic to penicillin.")
    response = wrapper.chat("What medications should I avoid?")
    # → ответ использует факт об аллергии из памяти

    # Статистика
    wrapper.print_stats()

    # Сравнение с no-memory режимом
    no_mem_response = wrapper.chat_no_memory("What medications should I avoid?")
"""

import os
import time
from typing import List, Dict, Optional, Tuple

import torch
from openai import OpenAI, AsyncOpenAI

from ngt.core.llm_memory import NGTMemoryForLLM

# Модели, требующие max_completion_tokens вместо max_tokens
_NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class NGTMemoryLLMWrapper:
    """
    Интегрирует NGTMemoryForLLM с OpenAI Chat API.

    Поток одного хода:
        1. Encode user message → embedding (text-embedding-3-small)
        2. Retrieve relevant memories → top_k фактов из прошлых ходов
        3. Inject memory context в system prompt
        4. Call LLM (gpt-4o-mini) → response
        5. Store user message + assistant response в NGT Memory

    Параметры:
        openai_api_key:   OpenAI API ключ (или из env OPENAI_API_KEY)
        model:            Chat модель (default: gpt-4o-mini)
        embedding_model:  Embedding модель (default: text-embedding-3-small)
        memory_top_k:     Сколько воспоминаний инжектить в контекст
        memory_threshold: Минимальный cosine score для включения воспоминания
        system_prompt:    Базовый system prompt
        use_graph:        Использовать ли graph retrieval (default: True)
        embedding_dim:    Размерность embedding (1536 для text-embedding-3-small)
        verbose:          Печатать ли debug информацию
    """

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant with persistent memory. "
        "When relevant memories are provided, use them to give accurate, "
        "personalized responses. Always prioritize information from your memory "
        "over general assumptions."
    )

    MEMORY_INJECTION_TEMPLATE = (
        "\n\n[MEMORY CONTEXT]\n"
        "Relevant facts from previous conversations:\n"
        "{memories}\n"
        "[END MEMORY CONTEXT]\n"
    )

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        memory_top_k: int = 5,
        memory_threshold: float = 0.25,
        system_prompt: Optional[str] = None,
        use_graph: bool = True,
        embedding_dim: int = 1536,
        verbose: bool = False,
    ):
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не установлен")

        self.client = OpenAI(api_key=api_key)
        self.aclient = AsyncOpenAI(api_key=api_key)
        self.model           = model
        self.embedding_model = embedding_model
        self.memory_top_k    = memory_top_k
        self.memory_threshold = memory_threshold
        self.system_prompt   = system_prompt or self.DEFAULT_SYSTEM
        self.use_graph       = use_graph
        self.embedding_dim   = embedding_dim
        self.verbose         = verbose

        # NGT Memory
        self.memory = NGTMemoryForLLM(
            embedding_dim=embedding_dim,
            max_entries=10000,
            max_concepts=3000,
            hebbian_lr=0.15,
            consolidation_interval=500,
            concept_extraction="regex",
            concept_top_k=6,
            device="cpu",
        )

        # История текущего разговора (для OpenAI messages format)
        self._chat_history: List[Dict[str, str]] = []

        # Статистика
        self._stats = {
            "total_turns":          0,
            "total_memories_used":  0,
            "total_tokens_in":      0,
            "total_tokens_out":     0,
            "total_embed_calls":    0,
            "total_chat_calls":     0,
            "latency_embed_ms":     [],
            "latency_chat_ms":      [],
            "latency_retrieve_ms":  [],
            "latency_store_ms":     [],
        }

    # ── LLM helpers ──────────────────────────────────────────────────

    def _chat_kwargs(self, max_tokens: int = 512) -> dict:
        """Возвращает kwargs для chat.completions.create с учётом модели.
        Reasoning-модели (gpt-5*, o1, o3, o4) используют токены на 'думание',
        поэтому нужен больший лимит и max_completion_tokens вместо max_tokens."""
        if any(self.model.startswith(p) for p in _NEW_API_PREFIXES):
            return {"max_completion_tokens": max(max_tokens, 4096)}
        return {"temperature": 0.3, "max_tokens": max_tokens}

    # ── Public embedding API ─────────────────────────────────────────────────

    def embed_text(self, text: str) -> torch.Tensor:
        """Получает embedding текста через OpenAI API (публичный sync API)."""
        return self._embed(text)

    async def aembed_text(self, text: str) -> torch.Tensor:
        """Получает embedding текста через OpenAI API (публичный async API)."""
        return await self._aembed(text)

    @property
    def memory_entries_count(self) -> int:
        """Количество записей в памяти (публичный доступ)."""
        return self.memory.num_entries

    def flush(self) -> int:
        """Применяет накопленные Hebbian обновления (публичный API)."""
        return self.memory.flush_hebbian()

    # ── Embedding (internal) ─────────────────────────────────────────────

    def _embed(self, text: str) -> torch.Tensor:
        """Получает embedding через OpenAI API."""
        t0 = time.perf_counter()
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000],  # лимит токенов
        )
        emb_list = response.data[0].embedding
        emb = torch.tensor(emb_list, dtype=torch.float32)
        emb = emb / emb.norm()

        ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_embed_ms"].append(ms)
        self._stats["total_embed_calls"] += 1
        return emb

    async def _aembed(self, text: str) -> torch.Tensor:
        """Async версия _embed — не блокирует event loop."""
        t0 = time.perf_counter()
        response = await self.aclient.embeddings.create(
            model=self.embedding_model,
            input=text[:8000],
        )
        emb_list = response.data[0].embedding
        emb = torch.tensor(emb_list, dtype=torch.float32)
        emb = emb / emb.norm()

        ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_embed_ms"].append(ms)
        self._stats["total_embed_calls"] += 1
        return emb

    # ── Retrieve memories ─────────────────────────────────────────────

    def _retrieve_memories(self, query_emb: torch.Tensor) -> List[Dict]:
        """Извлекает релевантные воспоминания из NGT Memory."""
        t0 = time.perf_counter()
        results = self.memory.retrieve(
            query_embedding=query_emb,
            top_k=self.memory_top_k,
            use_graph=self.use_graph,
        )
        ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_retrieve_ms"].append(ms)

        # Фильтруем по threshold
        filtered = [r for r in results if r.get("score", 0) >= self.memory_threshold]
        return filtered

    # ── Store memories ────────────────────────────────────────────────

    def _store(self, text: str, emb: torch.Tensor, role: str, turn: int):
        """Сохраняет ход в NGT Memory."""
        t0 = time.perf_counter()
        self.memory.store(
            embedding=emb,
            text=text,
            concepts=None,  # авто-извлечение
            metadata={"role": role, "turn": turn},
            domain="conversation",
        )
        ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_store_ms"].append(ms)

    # ── Format memory context ─────────────────────────────────────────

    def _format_memory_context(self, memories: List[Dict]) -> str:
        """Форматирует воспоминания для инжекции в system prompt."""
        if not memories:
            return ""
        lines = []
        for i, m in enumerate(memories, 1):
            score = m.get("score", 0)
            text  = m.get("text", "")
            lines.append(f"  {i}. [{score:.2f}] {text}")
        memory_block = "\n".join(lines)
        return self.MEMORY_INJECTION_TEMPLATE.format(memories=memory_block)

    # ── Main chat ────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        domain: Optional[str] = None,
    ) -> Dict:
        """
        Один ход диалога с NGT Memory.

        Returns:
            {
                "response": str,          # ответ LLM
                "memories_used": List,    # воспоминания которые были инжектированы
                "tokens_in": int,
                "tokens_out": int,
                "latency_ms": float,
            }
        """
        turn = self._stats["total_turns"]

        # 1. Embed user message
        user_emb = self._embed(user_message)

        # 2. Retrieve relevant memories
        memories = self._retrieve_memories(user_emb)

        if self.verbose and memories:
            print(f"\n  [Memory] {len(memories)} воспоминаний найдено:")
            for m in memories:
                print(f"    [{m.get('score', 0):.2f}] {m.get('text', '')[:80]}")

        # 3. Build system prompt с memory context
        memory_context = self._format_memory_context(memories)
        system_with_memory = self.system_prompt + memory_context

        # 4. Build messages
        messages = [{"role": "system", "content": system_with_memory}]
        messages += self._chat_history[-6:]  # последние 3 хода для краткосрочного контекста
        messages.append({"role": "user", "content": user_message})

        # 5. Call LLM
        t0 = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._chat_kwargs(512),
        )
        chat_ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_chat_ms"].append(chat_ms)
        self._stats["total_chat_calls"] += 1

        assistant_response = completion.choices[0].message.content
        tokens_in  = completion.usage.prompt_tokens
        tokens_out = completion.usage.completion_tokens

        # 6. Store user + assistant в память
        self._store(user_message, user_emb, role="user", turn=turn)

        asst_emb = self._embed(assistant_response)
        self._store(assistant_response, asst_emb, role="assistant", turn=turn)
        self.memory.flush_hebbian()

        # 7. Обновляем историю чата (краткосрочная)
        self._chat_history.append({"role": "user",      "content": user_message})
        self._chat_history.append({"role": "assistant", "content": assistant_response})

        # 8. Обновляем статистику
        self._stats["total_turns"]         += 1
        self._stats["total_memories_used"] += len(memories)
        self._stats["total_tokens_in"]     += tokens_in
        self._stats["total_tokens_out"]    += tokens_out

        return {
            "response":      assistant_response,
            "memories_used": memories,
            "tokens_in":     tokens_in,
            "tokens_out":    tokens_out,
            "latency_ms":    chat_ms,
        }

    async def achat(
        self,
        user_message: str,
        domain: Optional[str] = None,
    ) -> Dict:
        """Async версия chat — не блокирует event loop FastAPI."""
        turn = self._stats["total_turns"]

        # 1. Embed user message (async)
        user_emb = await self._aembed(user_message)

        # 2. Retrieve relevant memories (sync — CPU only, ~2ms)
        memories = self._retrieve_memories(user_emb)

        # 3. Build system prompt с memory context
        memory_context = self._format_memory_context(memories)
        system_with_memory = self.system_prompt + memory_context

        # 4. Build messages
        messages = [{"role": "system", "content": system_with_memory}]
        messages += self._chat_history[-6:]
        messages.append({"role": "user", "content": user_message})

        # 5. Call LLM (async — основной bottleneck)
        t0 = time.perf_counter()
        completion = await self.aclient.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._chat_kwargs(512),
        )
        chat_ms = (time.perf_counter() - t0) * 1000
        self._stats["latency_chat_ms"].append(chat_ms)
        self._stats["total_chat_calls"] += 1

        assistant_response = completion.choices[0].message.content or ""
        tokens_in  = completion.usage.prompt_tokens
        tokens_out = completion.usage.completion_tokens

        # 6. Store user + assistant в память (async embed)
        self._store(user_message, user_emb, role="user", turn=turn)

        asst_emb = await self._aembed(assistant_response)
        self._store(assistant_response, asst_emb, role="assistant", turn=turn)
        self.memory.flush_hebbian()

        # 7. Обновляем историю чата
        self._chat_history.append({"role": "user",      "content": user_message})
        self._chat_history.append({"role": "assistant", "content": assistant_response})

        # 8. Обновляем статистику
        self._stats["total_turns"]         += 1
        self._stats["total_memories_used"] += len(memories)
        self._stats["total_tokens_in"]     += tokens_in
        self._stats["total_tokens_out"]    += tokens_out

        return {
            "response":      assistant_response,
            "memories_used": memories,
            "tokens_in":     tokens_in,
            "tokens_out":    tokens_out,
            "latency_ms":    chat_ms,
        }

    async def achat_no_memory(self, user_message: str) -> Dict:
        """Async версия chat_no_memory."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self._chat_history[-6:]
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        completion = await self.aclient.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._chat_kwargs(512),
        )
        chat_ms = (time.perf_counter() - t0) * 1000

        return {
            "response":  completion.choices[0].message.content or "",
            "memories_used": [],
            "tokens_in":  completion.usage.prompt_tokens,
            "tokens_out": completion.usage.completion_tokens,
            "latency_ms": chat_ms,
        }

    def chat_no_memory(self, user_message: str) -> Dict:
        """
        Тот же вопрос БЕЗ NGT Memory — для сравнения.
        Использует только текущую краткосрочную историю чата.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self._chat_history[-6:]
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._chat_kwargs(512),
        )
        chat_ms = (time.perf_counter() - t0) * 1000

        return {
            "response":  completion.choices[0].message.content,
            "tokens_in":  completion.usage.prompt_tokens,
            "tokens_out": completion.usage.completion_tokens,
            "latency_ms": chat_ms,
        }

    # ── Session management ────────────────────────────────────────────

    def new_session(self):
        """Начинает новую сессию — очищает краткосрочную историю, сохраняет долгосрочную."""
        self._chat_history = []
        self.memory.new_session()

    # ── Stats ─────────────────────────────────────────────────────────

    def print_stats(self):
        s = self._stats
        def avg(lst): return sum(lst) / len(lst) if lst else 0

        print("\n── NGTMemoryLLMWrapper Stats ──────────────────")
        print(f"  Turns:            {s['total_turns']}")
        print(f"  Memories used:    {s['total_memories_used']} total ({s['total_memories_used']/max(s['total_turns'],1):.1f} avg/turn)")
        print(f"  Tokens in:        {s['total_tokens_in']}")
        print(f"  Tokens out:       {s['total_tokens_out']}")
        print(f"  Latency embed:    {avg(s['latency_embed_ms']):.0f} ms avg")
        print(f"  Latency retrieve: {avg(s['latency_retrieve_ms']):.0f} ms avg")
        print(f"  Latency store:    {avg(s['latency_store_ms']):.0f} ms avg")
        print(f"  Latency chat:     {avg(s['latency_chat_ms']):.0f} ms avg")
        print(f"  Graph edges:      {self.memory.associations.num_edges}")
        print(f"  Graph concepts:   {self.memory.associations.num_concepts}")
        print(f"  Memory entries:   {self.memory.num_entries}")
        print("──────────────────────────────────────────────")

    def get_stats(self) -> Dict:
        s = self._stats
        def avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0
        return {
            "total_turns":           s["total_turns"],
            "total_memories_used":   s["total_memories_used"],
            "avg_memories_per_turn": round(s["total_memories_used"] / max(s["total_turns"], 1), 2),
            "total_tokens_in":       s["total_tokens_in"],
            "total_tokens_out":      s["total_tokens_out"],
            "avg_embed_ms":          avg(s["latency_embed_ms"]),
            "avg_retrieve_ms":       avg(s["latency_retrieve_ms"]),
            "avg_store_ms":          avg(s["latency_store_ms"]),
            "avg_chat_ms":           avg(s["latency_chat_ms"]),
            "graph_edges":           self.memory.associations.num_edges,
            "graph_concepts":        self.memory.associations.num_concepts,
            "memory_entries":        self.memory.num_entries,
        }

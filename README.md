# NGT Memory API

> Persistent memory layer for LLM applications — drop-in REST API.

[![Version](https://img.shields.io/badge/version-0.23.0-blue)](https://github.com/ngt-memory/ngt-memory)
[![License](https://img.shields.io/badge/license-BSL%201.1-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

🇷🇺 [Документация на русском](README_RU.md)

---

## What is NGT Memory?

NGT Memory gives your LLM **persistent, cross-session memory** — like a hippocampus for AI.

Instead of forgetting everything between conversations, the LLM remembers:
- User preferences, allergies, constraints
- Past decisions and context
- Cross-domain facts (medical + dietary + travel all linked)

**Benchmark results (Exp 44, gpt-4o-mini):**

| Mode | Factual consistency (0-3) | Keyword hit |
|------|--------------------------|-------------|
| **NGT Memory (graph)** | **2.33 / 3** | **44%** |
| **NGT Memory (emb)** | **2.44 / 3** | **44%** |
| No memory | 1.22 / 3 | 27% |
| **Δ improvement** | **+1.22 (+100%)** | **+17pp** |

Real example — **without memory**:
> User: "What restaurants in Kyoto would you recommend?"
> LLM: *"Ippudo is great for ramen lovers"* ← recommended meat ramen to a vegetarian ❌

**With NGT Memory** (remembers the user is vegetarian):
> LLM: *"Shigetsu at Tenryu-ji Temple serves shojin ryori (Buddhist vegan cuisine)"* ✓

---

## Hosted API (ready to use)

Don't want to self-host? Use our hosted API at **[ngt-memory.ru](https://ngt-memory.ru)** — no setup required.

Pricing: from **1 990 ₽/month** → get an API key → use immediately.

---

## Self-hosted: Quick Start

### Option 1: Local / development

```bash
# 1. Clone
git clone https://github.com/ngt-memory/ngt-memory
cd ngt-memory

# 2. Install
pip install -r requirements.txt -r requirements-api.txt

# 3. Configure
cp .env.example .env
# Open .env, set: OPENAI_API_KEY=sk-...

# 4. Start
uvicorn api.main:app --host 0.0.0.0 --port 9190 --reload
```

API: http://localhost:9190  
Swagger UI: http://localhost:9190/docs

### Option 2: Docker (recommended)

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env

docker-compose up -d
```

> **Note:** single `uvicorn` worker by default — this is intentional.
> Sessions are in-memory, so one worker keeps `session_id` state consistent.
> For multi-worker setups, use sticky routing or move sessions to a shared backend.

### Option 3: VPS (Ubuntu/Debian)

```bash
git clone https://github.com/ngt-memory/ngt-memory
cd ngt-memory
cp .env.example .env
nano .env   # set OPENAI_API_KEY

docker-compose up -d
```

API will be available on port **9190**. Put nginx in front to add HTTPS.

---

## API Reference

### `POST /chat` — Chat with memory

Sends a message to the LLM. Automatically retrieves relevant memories and injects them into context. Saves the conversation to memory.

**Request:**
```json
{
  "message": "What medications should I avoid?",
  "session_id": "user_42",
  "use_memory": true
}
```

**Response:**
```json
{
  "response": "Based on your penicillin allergy and lisinopril prescription, you should avoid NSAIDs like ibuprofen...",
  "session_id": "user_42",
  "memories_used": [
    {"text": "Patient is allergic to penicillin", "score": 0.91},
    {"text": "Takes lisinopril 10mg daily", "score": 0.87}
  ],
  "memories_count": 2,
  "tokens_in": 312,
  "tokens_out": 89,
  "latency_ms": 1847.3,
  "memory_entries": 14
}
```

**cURL example:**
```bash
curl -X POST http://localhost:9190/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Anton and I am allergic to penicillin.", "session_id": "demo"}'
```

---

### `POST /store` — Store a fact directly

Pre-load facts into memory without going through chat (user profiles, documents, structured data).

**Request:**
```json
{
  "text": "Patient Anton, DOB 1990, allergic to penicillin, takes lisinopril 10mg.",
  "session_id": "medical_user_1",
  "domain": "medical"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "medical_user_1",
  "memory_entries": 1,
  "message": "Stored. Total memory entries: 1"
}
```

---

### `POST /retrieve` — Semantic search over memory

Direct semantic search without LLM call. Useful for building custom memory-augmented pipelines.

**Request:**
```json
{
  "query": "drug allergies",
  "session_id": "medical_user_1",
  "top_k": 5,
  "use_graph": true,
  "threshold": 0.2
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Patient is allergic to penicillin",
      "score": 0.91,
      "domain": "medical",
      "concepts": ["penicillin", "allergy", "patient"]
    }
  ],
  "count": 1,
  "session_id": "medical_user_1",
  "query": "drug allergies"
}
```

---

### `POST /session/reset` — Clear session memory

```json
{"session_id": "user_42"}
```

---

### `GET /session/{session_id}/stats` — Session statistics

```json
{
  "session_id": "user_42",
  "memory_entries": 24,
  "graph_edges": 47,
  "graph_concepts": 31,
  "total_turns": 12,
  "total_memories_used": 38,
  "avg_memories_per_turn": 3.8,
  "total_tokens_in": 4200,
  "total_tokens_out": 1890,
  "avg_embed_ms": 712.0,
  "avg_retrieve_ms": 2.1,
  "avg_chat_ms": 1543.0
}
```

---

### `GET /health` — Health check

```json
{
  "status": "ok",
  "version": "0.23.0",
  "active_sessions": 3,
  "model": "gpt-4.1-nano",
  "embedding_model": "text-embedding-3-small"
}
```

---

## Session model

Each `session_id` has **isolated memory**. Use different session IDs for different users:

```
session_id="user_42"   → memory of user 42
session_id="user_99"   → separate, isolated memory
session_id="default"   → shared (for single-user setups)
```

Sessions expire after `NGT_SESSION_TTL` seconds of inactivity (default: 1 hour).

Important deployment note:

- With the current in-memory `SessionStore`, Docker defaults to **1 uvicorn worker**.
- Multi-worker deployments require **sticky sessions** or a **shared session backend**.
- Without that, the same `session_id` can be split across processes.

---

## Python client example

```python
import httpx

BASE_URL = "http://localhost:9190"
SESSION = "my_user"

client = httpx.Client(base_url=BASE_URL)

# First conversation — introduce yourself
r = client.post("/chat", json={
    "message": "I'm vegetarian and I'm planning a trip to Kyoto next month.",
    "session_id": SESSION
})
print(r.json()["response"])

# Next day — new question, memory persists
r = client.post("/chat", json={
    "message": "What restaurants do you recommend in Kyoto?",
    "session_id": SESSION
})
# → Recommends vegetarian restaurants, remembers Kyoto trip
print(r.json()["response"])
print(f"Memories used: {r.json()['memories_count']}")
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `NGT_API_SECRET` | *(empty)* | Optional API protection key |
| `NGT_CHAT_MODEL` | `gpt-4.1-nano` | Any OpenAI chat model (gpt-4.1-nano, gpt-4o-mini, gpt-4o, etc.) |
| `NGT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `NGT_MEMORY_TOP_K` | `5` | Memories injected per turn |
| `NGT_MEMORY_THRESHOLD` | `0.25` | Min cosine score to include memory |
| `NGT_USE_GRAPH` | `true` | Enable graph-boosted retrieval |
| `NGT_SESSION_TTL` | `3600` | Session inactivity timeout (seconds) |
| `NGT_MAX_SESSIONS` | `100` | Max concurrent sessions |
| `NGT_CORS_ORIGINS` | `*` | Allowed CORS origins, comma-separated |
| `NGT_LOG_LEVEL` | `INFO` | Log level |
| `NGT_LOG_JSON` | `true` | Structured JSON logs |

---

## Architecture

```
User request
     ↓
[POST /chat]
     ↓
OpenAI Embeddings (text-embedding-3-small)  ~700ms
     ↓
NGT Memory Retrieve (cosine + graph boost)  ~2-3ms
     ↓
System prompt + [MEMORY CONTEXT] injection
     ↓
OpenAI Chat (gpt-4.1-nano)                 ~800-1500ms
     ↓
Store user + assistant → NGT Memory        ~1ms
     ↓
Response
```

**NGT overhead: ~2-3ms** (excluding API calls).
All OpenAI calls are fully async (`AsyncOpenAI`) — multiple requests processed concurrently.
Default Docker deployment: **1 uvicorn worker** for consistent in-memory sessions.
For multi-worker or multi-instance deployments, use sticky routing or a shared session backend.

Any OpenAI chat model is supported — set `NGT_CHAT_MODEL` in `.env`. Default: `gpt-4.1-nano` (fastest, cheapest).

---

## How NGT Memory works

NGT Memory combines **three retrieval mechanisms**:

1. **Cosine similarity** — finds semantically close facts
2. **Hebbian association graph** — links concepts that appear together  
   (e.g., "vegetarian" ↔ "restaurant" ↔ "Kyoto" get linked after first use)
3. **Hierarchical consolidation** — important facts promoted to semantic memory

This enables **cross-session recall**: ask about "restaurants" → retrieves "vegetarian preference" even if the words don't overlap.

---

## Performance

From Exp 40 benchmarks (NGT v0.23.0, CPU only, 5000 facts, dim=384):

| Operation | Throughput | Latency (p50) |
|-----------|-----------|--------|
| store() | 3,450 / sec | 0.29 ms |
| retrieve() | 150 q/sec | 6.3 ms |
| Memory footprint | — | ~0.8 MB / 1000 entries |

42× faster store and 6× faster retrieval vs v0.19.0 (Exp 39).

End-to-end retrieval (Exp 44, via API with OpenAI embeddings):

| Scenario | avg_retrieve_ms | avg_embed_ms |
|----------|----------------|-------------|
| Medical assistant | 3.5 ms | 1069 ms |
| Personal assistant | 1.8 ms | 867 ms |
| Tech support | 2.3 ms | 357 ms |
| **Average** | **2.5 ms** | **764 ms** |

Realistic profile A/B test (Exp 48, `gpt-4.1-nano`, local Docker, single worker):

| Metric | With memory | No memory |
|--------|-------------|-----------|
| Avg profile-aware score | **0.917** | **0.083** |
| Scenario wins | **5 / 6** | 0 / 6 |
| Retrieval success | **6 / 6** | — |

This experiment uses realistic user-profile scenarios (medical, travel, support, billing, fitness) and compares `use_memory=true` vs `use_memory=false` on the same prompts.

---

## License

**Business Source License 1.1 (BSL 1.1)**

- **Free** for personal projects, education, research, development, testing, and non-profits
- **Commercial production use** requires a paid license or hosted API subscription at [ngt-memory.ru](https://ngt-memory.ru)
- Automatically converts to **Apache 2.0** on 2030-03-24

See [LICENSE](LICENSE) for full details.

---

## Citation

If you use NGT Memory in research:

```bibtex
@software{ngt_memory_2026,
  title  = {NGT Memory: Neuroplastic Graph-based External Memory for LLMs},
  author = {Anton},
  year   = {2026},
  url    = {https://github.com/ngt-memory/ngt-memory},
  note   = {v0.23.0}
}
```

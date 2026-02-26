# NGT Memory API

> Persistent memory layer for LLM applications — drop-in REST API.

[![Version](https://img.shields.io/badge/version-0.23.0-blue)](https://github.com/ngt-memory/ngt-memory)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

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
| **NGT Memory** | **2.44 / 3** | **57%** |
| No memory | 1.33 / 3 | 27% |
| **Δ improvement** | **+1.11 (+83%)** | **+30pp** |

Real example — **without memory**:
> User: "What restaurants in Kyoto would you recommend?"
> LLM: *"Ippudo is great for ramen lovers"* ← recommended meat ramen to a vegetarian ❌

**With NGT Memory** (remembers the user is vegetarian):
> LLM: *"Shigetsu at Tenryu-ji Temple serves shojin ryori (Buddhist vegan cuisine)"* ✓

---

## Quick Start

### Option 1: Run locally

```bash
# 1. Clone
git clone https://github.com/ngt-memory/ngt-memory
cd ngt-memory

# 2. Install dependencies
pip install -r requirements.txt -r requirements-api.txt

# 3. Configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# 4. Start
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API доступен на: http://localhost:8000
Swagger UI: http://localhost:8000/docs

### Option 2: Docker (recommended for production)

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

docker-compose up -d
```

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
curl -X POST http://localhost:8000/chat \
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
  "avg_memories_per_turn": 3.8,
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
  "model": "gpt-4o-mini",
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

---

## Python client example

```python
import httpx

BASE_URL = "http://localhost:8000"
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
| `NGT_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `NGT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `NGT_MEMORY_TOP_K` | `5` | Memories injected per turn |
| `NGT_MEMORY_THRESHOLD` | `0.25` | Min cosine score to include memory |
| `NGT_USE_GRAPH` | `true` | Enable graph-boosted retrieval |
| `NGT_SESSION_TTL` | `3600` | Session inactivity timeout (seconds) |
| `NGT_MAX_SESSIONS` | `100` | Max concurrent sessions |

---

## Architecture

```
User request
     ↓
[POST /chat]
     ↓
OpenAI Embeddings (text-embedding-3-small)  ~700ms
     ↓
NGT Memory Retrieve (cosine + graph boost)  ~2ms
     ↓
System prompt + [MEMORY CONTEXT] injection
     ↓
OpenAI Chat (gpt-4o-mini)                  ~1500ms
     ↓
Store user + assistant → NGT Memory        ~1ms
     ↓
Response
```

**NGT overhead: ~2ms** (excluding embedding API).
The only latency cost is the OpenAI embedding call (~700ms), which can be eliminated with local embeddings.

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

From Exp 40 benchmarks (NGT v0.20.0):

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| store() | 42,000 / sec | 0.024 ms |
| retrieve() | 6,000 / sec | 0.17 ms |
| Memory footprint | — | ~2.4 MB / 1000 entries |

---

## Deploy to cloud

### Railway (easiest)
```bash
railway login
railway init
railway up
# Set OPENAI_API_KEY in Railway dashboard → Variables
```

### Fly.io
```bash
fly launch
fly secrets set OPENAI_API_KEY=sk-...
fly deploy
```

### Any VPS (Ubuntu)
```bash
git clone https://github.com/ngt-memory/ngt-memory
cd ngt-memory
cp .env.example .env && nano .env   # set OPENAI_API_KEY
docker-compose up -d
```

---

## License

MIT — free for commercial and personal use.

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

# ТЗ v3.0: Правки сайта ngt-memory.ru — устранение всех расхождений

**Дата:** 2026-03-05  
**Стек сайта:** Next.js (static export), Tailwind CSS, Lucide icons  
**Деплой:** IIS → `C:\Sites\ngt-memory.ru\wwwroot\out`  
**API:** `https://ngt-memory.ru/api` (Docker, порт 9190)  
**GitHub:** `https://github.com/ngt-memory/ngt-memory`

---

## Текущее состояние сайта (что уже правильно ✅)

- ✅ Hero subtitle: "~2-3ms retrieval. Zero infrastructure."
- ✅ Features карточка: "Ultra-fast Retrieval — ~2-3ms average"
- ✅ Compare секция: "sub-5ms retrieval"
- ✅ Code example: порт 9190
- ✅ API Docs ссылки → https://ngt-memory.ru/api/docs
- ✅ GitHub ссылки работают
- ✅ Footer: Apache 2.0 License, GitHub, Documentation
- ✅ "Try it now" секция присутствует (песочница)

---

## РАСХОЖДЕНИЯ — что нужно исправить

---

### 1. OG Description meta-тег

**Сейчас:**
```
Drop-in REST API that adds persistent cross-session memory to any LLM. 2ms retrieval, no vector DB required.
```

**Нужно:**
```
Drop-in REST API that adds persistent cross-session memory to any LLM. ~2-3ms retrieval, no vector DB required.
```

**Почему:** "2ms" → "~2-3ms" (среднее по Exp 44 = 2.5ms, максимум 3.5ms).

**Где искать:** `<meta property="og:description">` в `<head>` (index.html или _document / layout)

---

### 2. Pricing: модель gpt-5-nano → gpt-4.1-nano

**Сейчас на сайте (все 3 тарифа):**
```
Model: gpt-5-nano
```

**Нужно:**
```
Model: gpt-4.1-nano
```

**Почему:** Production API переключён на gpt-4.1-nano:
- Не reasoning модель — не тратит токены на "думание"
- В 30× быстрее (p50: 1.9с vs 58с)
- Дешевле: $0.10/$0.40 vs $0.15/$0.60 per 1M tokens
- Подтверждено Exp 47: 1.88 req/s, 0 ошибок

**Где искать:** Pricing секция — три карточки Free / Pro / Enterprise

---

### 3. Benchmark секция — нет конкретных цифр

**Сейчас:**
```
Proven results — not just promises
Benchmarked against baseline LLM with no memory layer
Benchmark results — Exp 44 (text-embedding-3-small)
```
→ Только заголовок + (вероятно) картинка-заглушка, без реальных данных.

**Нужно добавить реальные цифры** (одно из двух):

**Вариант A — Таблица:**

| Mode | Judge Score (0-3) | Keyword Hit |
|------|------------------|-------------|
| NGT Memory (graph) | 2.33 | 44% |
| NGT Memory (emb) | 2.44 | 44% |
| No memory baseline | 1.22 | 27% |
| **Δ improvement** | **+100%** | **+17pp** |

**Вариант B — Три метрики-карточки (проще визуально):**
- **+100%** factual accuracy vs baseline
- **2.44/3** judge score
- **~2-3ms** memory retrieval

**Источник данных:** `results/44_openai_memory_test.json`

---

### 4. Concurrency / Performance данные — новая информация

После оптимизаций (Exp 47) у нас есть данные о реальной нагрузке.
Можно добавить в секцию "Everything you need" или "How we compare":

| Метрика | Значение |
|---------|----------|
| Concurrent throughput | 1.88 req/s (5 parallel users) |
| Latency p50 | 1.9 сек (end-to-end) |
| Session mode | in-memory session store |
| Default Docker workers | 1 uvicorn worker |
| Async | Full async OpenAI pipeline |

**Не обязательно** — но усиливает доверие. Можно сделать как подпись к Compare секции.

Важно: не использовать формулировку про multi-worker memory isolation как production claim.
Эксперимент Exp 48 показал, что для in-memory `session_id` consistency нужен single-worker deployment или shared backend.

---

### 5. "How we compare" секция — проверить данные

**Сейчас:**
```
NGT Memory is the only solution that requires no external vector database and delivers sub-5ms retrieval
```

**Это корректно** ✅ — sub-5ms подтверждено (avg 2.5ms, max 3.5ms по Exp 44).

Но если есть таблица сравнения с конкурентами — проверить что не завышены цифры.

---

### 6. Use cases секция — проверить пример

**Сейчас (Personal AI Companion):**
```
Knows you're vegetarian, live in Berlin, and training for a marathon
```

**Это подтверждено Exp 46** ✅ — 100% recall всех 4 фактов (vegetarian, Berlin, nut allergy, marathon).

---

## СПРАВОЧНИК ПРОВЕРЕННЫХ ЦИФР

Все цифры ниже подтверждены экспериментами. Можно безопасно использовать на сайте.

### Exp 44 — Quality benchmark (gpt-4o-mini + text-embedding-3-small)

| Метрика | NGT Memory | No memory | Δ |
|---------|-----------|-----------|---|
| Judge Score (0-3) | 2.44 | 1.22 | +100% |
| Keyword Hit | 44% | 27% | +17pp |

Retrieval latency:

| Сценарий | retrieve_ms | embed_ms |
|----------|-----------|----------|
| Medical | 3.5 | 1069 |
| Personal | 1.8 | 867 |
| Tech | 2.3 | 357 |
| **Среднее** | **2.5** | **764** |

### Exp 46 — Live API memory test (gpt-5-nano → gpt-4.1-nano)

| Метрика | Результат |
|---------|-----------|
| Шаги пройдены | 4/4 |
| Персонализация | 100% (5/5 keywords) |
| Полный recall | Все 4 факта |

### Exp 47 — Concurrent load test (gpt-4.1-nano, 4 workers, async)

| Метрика | Значение |
|---------|----------|
| Concurrent users | 5 |
| Requests | 35/35 (0 ошибок) |
| Burst throughput | 1.88 req/s |
| Latency p50 | 1,908 ms |
| Latency p95 | 3,431 ms |
| Async pipeline | OK |

### Exp 48 — Realistic profile A/B test (gpt-4.1-nano, single worker)

| Метрика | With memory | No memory |
|---------|-------------|-----------|
| Avg profile-aware score | 0.917 | 0.083 |
| Scenario wins | 5/6 | 0/6 |
| Retrieval success | 6/6 | — |

Вывод для сайта: лучше продавать не только latency, но и product value памяти на реалистичных профилях пользователя.

### Exp 40 — Raw throughput (CPU, 5000 facts)

| Operation | Throughput | Latency (p50) |
|-----------|-----------|---------------|
| store() | 3,450 /sec | 0.29 ms |
| retrieve() | 150 q/sec | 6.3 ms |
| RAM | — | ~0.8 MB / 1000 entries |

---

## БЕЗОПАСНЫЕ ФОРМУЛИРОВКИ

| Заявление | OK? | Примечание |
|-----------|-----|-----------|
| "~2-3ms retrieval" | ✅ | avg 2.5ms, max 3.5ms (Exp 44) |
| "sub-5ms retrieval" | ✅ | max 3.5ms < 5ms |
| "+100% accuracy vs baseline" | ✅ | (2.44-1.22)/1.22 = 100% |
| "2x better than no-memory" | ✅ | 2.44/1.22 = 2.0x |
| "No vector DB required" | ✅ | всё in-memory, torch tensors |
| "Zero infrastructure" | ✅ | только Docker |
| "Under 5 minutes" | ✅ | docker-compose up -d |
| "1.88 req/s concurrent" | ✅ | Exp 47, 5 users |
| "Latency < 2 seconds" | ✅ | p50 = 1.9s end-to-end |
| "5/6 realistic wins vs no-memory baseline" | ✅ | Exp 48 |
| "Single-worker Docker by default" | ✅ | нужно для in-memory sessions |
| "gpt-4.1-nano" | ✅ | текущая production модель |

| Заявление | ❌ НЕ использовать | Почему |
|-----------|-------------------|--------|
| "2ms retrieval" | точная цифра — не всегда правда | avg 2.5ms |
| "sub-2ms" | max 3.5ms | |
| "42,000 store/sec" | ошибка в старом README | реально 3,450 |
| "gpt-5-nano" в pricing | устарело | теперь gpt-4.1-nano |
| "gpt-4o-mini" | устарело | |
| "4 workers by default" | устарело | сейчас 1 worker по умолчанию |
| "100% memory isolation in production" | нельзя обещать | без shared backend это некорректно |

---

## ПОРЯДОК РАБОТ

| # | Задача | Время | Приоритет |
|---|--------|-------|-----------|
| 1 | OG description: "2ms" → "~2-3ms" | 5 мин | 🔴 Критично |
| 2 | Pricing: gpt-5-nano → gpt-4.1-nano (3 карточки) | 10 мин | 🔴 Критично |
| 3 | Benchmark: добавить реальные цифры (таблица или карточки) | 30 мин | 🟡 Важно |
| 4 | Проверить все мета-теги на расхождения | 10 мин | 🟡 Важно |

**Общее время:** ~1 час  
**Всего правок:** 2 критичных + 1 важная + 1 проверка

# ТЗ: Актуализация сайта ngt-memory.ru без полной переделки

**Дата:** 2026-03-10
**Формат работ:** точечная актуализация контента и метаданных
**Что важно:** это не редизайн и не полная пересборка лендинга

---

## 1. Цель

Привести публичный сайт `ngt-memory.ru` в соответствие с текущим состоянием продукта, API и проверенных экспериментов.

Нужно обновить:
- фактические claims
- модель по умолчанию
- Docker / deployment messaging
- benchmark / proof sections
- тексты про практическую ценность памяти

Не нужно:
- менять визуальную концепцию
- менять структуру секций целиком
- переписывать весь копирайтинг с нуля
- делать новый дизайн hero / pricing / footer

---

## 2. Что остаётся без изменений

Сохранить как есть:
- текущую структуру лендинга
- текущую дизайн-систему
- текущую навигацию
- текущий стек сайта
- текущие CTA-кнопки, если ссылки корректны
- текущие API Docs / GitHub ссылки, если они рабочие

Допустимы только точечные изменения текста, цифр, подпунктов, подсказок и микро-блоков.

---

## 3. Главные изменения по смыслу

### 3.1. Что сайт должен сообщать после актуализации

Сайт должен транслировать четыре вещи:

1. `NGT Memory` даёт **persistent cross-session memory** для LLM
2. retrieval остаётся быстрым: **~2-3ms**
3. реальная продуктовая ценность подтверждена realistic experiment'ом:
   - **5/6 wins** против no-memory baseline
   - **0.917 vs 0.083** average profile-aware score
4. текущий Docker default — **single worker**, потому что session store пока in-memory

### 3.2. Что нельзя обещать

Нельзя писать или подразумевать:
- что Docker по умолчанию использует `4 workers`
- что в production уже есть корректный multi-worker shared session state
- что есть `100% memory isolation in production` без shared backend
- что retrieval всегда `2ms` ровно

---

## 4. Контентные правки по блокам

### 4.1. Hero

**Оставить структуру блока.**

**Обновить подзаголовок на один из вариантов:**

Вариант A:
`Drop-in REST API that adds persistent cross-session memory to any LLM. ~2-3ms retrieval. Zero extra infrastructure.`

Вариант B:
`Persistent memory layer for LLM apps. ~2-3ms retrieval, realistic profile recall, simple Docker deployment.`

**Задача блока:**
- не перегрузить
- не обещать multi-worker scalability
- сразу дать триггер: memory + speed + simple deploy

### 4.2. Meta description / OG description

**Заменить все варианты с `2ms retrieval` на `~2-3ms retrieval`.**

Рекомендуемый текст:
`Drop-in REST API that adds persistent cross-session memory to any LLM. ~2-3ms retrieval, no vector DB required, open source.`

### 4.3. Pricing / plan cards

Если в тарифах или карточках фигурирует модель по умолчанию, обновить:

- было: `gpt-5-nano`
- должно стать: `gpt-4.1-nano`

Если указана benefit-строка, можно использовать:
- `Fast default model: gpt-4.1-nano`
- `Optimized for low-latency memory-aware chat`

### 4.4. Benchmark / proof section

**Не переделывать секцию целиком.**
Нужно заменить заглушки и общие фразы на реальные цифры.

#### Обязательные цифры

**Latency proof:**
- `~2-3ms retrieval`
- `sub-5ms retrieval`

**Realistic value proof (новое, обязательно):**
- `5/6 realistic profile scenarios won vs no-memory baseline`
- `0.917 avg profile-aware score with memory`
- `0.083 avg profile-aware score without memory`

**Качество из Exp 44:**
- `2.44 / 3 judge score`
- `+100% vs no-memory baseline`

#### Рекомендуемый формат

Либо 3 карточки:
- `~2-3ms retrieval`
- `5/6 realistic wins`
- `+100% quality vs baseline`

Либо таблица + 1 строка ниже:
`In a realistic A/B profile test, memory won 5 out of 6 scenarios against the same model without memory.`

### 4.5. How it works / Architecture

**Визуальную схему не менять.**
Нужно только поправить подписи и пояснение под схемой.

Использовать:
- chat model: `gpt-4.1-nano`
- retrieval: `~2-3ms`
- deployment note below diagram:
  `Default Docker deployment runs a single API worker to keep in-memory sessions consistent. Multi-worker mode requires sticky routing or a shared session backend.`

### 4.6. Quick Start / Docker tab

Нужно оставить current code example, но добавить маленький note под Docker-командой:

`Default Docker deployment uses 1 worker because session state is currently stored in memory.`

Если есть блок “production-ready”, заменить на более корректный:
- было: `production-ready Docker setup`
- стало: `simple Docker deployment`

### 4.7. Features

Не менять сетку.

Нужно только аккуратно поправить тексты:
- `Persistent Memory`
- `~2-3ms Retrieval`
- `Isolated Sessions`
- `Async OpenAI Pipeline`
- `API Key Auth`

Если сейчас есть что-то про multi-worker scaling — убрать или переформулировать нейтрально.

### 4.8. Comparison section

Можно оставить общий layout.

Нужно проверить, чтобы claim был только такой:
- `No vector DB required`
- `sub-5ms retrieval`
- `REST API`

Не добавлять неподтверждённые claims против конкурентов.

### 4.9. Use cases

Секцию не перестраивать.

Нужно усилить формулировки конкретными фактами:

- **Medical AI Assistant**
  `Remembers allergies, medications, and prior reactions across sessions.`

- **Personal AI Companion**
  `Keeps track of preferences, travel constraints, and personal plans.`

- **Customer Support Bot**
  `Recalls prior issues, refund preferences, and user-specific constraints.`

### 4.10. Footer / small print

Если есть блок с мелким пояснением или docs link, добавить ссылку на API docs и GitHub без изменения композиции.

---

## 5. Факты, которые можно использовать на сайте

### Разрешённые claims

- `~2-3ms retrieval`
- `sub-5ms retrieval`
- `No vector DB required`
- `Open source`
- `gpt-4.1-nano default model`
- `5/6 realistic scenario wins vs no-memory baseline`
- `0.917 avg profile-aware score with memory`
- `0.083 avg profile-aware score without memory`
- `+100% quality vs baseline`
- `Single-worker Docker by default for in-memory session consistency`

### Запрещённые claims

- `2ms retrieval`
- `4 workers by default`
- `100% memory isolation in production`
- `shared backend already implemented`
- `fully horizontally scalable out of the box`

---

## 6. Обязательные текстовые замены

### Замена 1
- Было: `2ms retrieval`
- Стало: `~2-3ms retrieval`

### Замена 2
- Было: `gpt-5-nano`
- Стало: `gpt-4.1-nano`

### Замена 3
- Было: `4 workers`
- Стало: `single-worker Docker default`

### Замена 4
- Было: общие фразы вида `proven performance`
- Стало: конкретные метрики `5/6 wins`, `0.917 score`, `~2-3ms retrieval`

---

## 7. Что не нужно менять руками без причины

Не трогать без необходимости:
- палитру
- иконки
- spacing
- layout секций
- animations
- порядок блоков на странице

Если правка требует редизайна, её нужно отложить, а не делать в рамках этой задачи.

---

## 8. Приёмка

Задача считается выполненной, если:

- на сайте больше нет устаревшего `gpt-5-nano`
- на сайте больше нет claim'ов про `4 workers by default`
- все упоминания `2ms retrieval` заменены на `~2-3ms retrieval`
- добавлен хотя бы один блок с результатом Exp 48
- Docker messaging отражает single-worker default
- обновления выглядят как аккуратная актуализация существующего сайта, а не как новый сайт

---

## 9. Короткий приоритет работ

1. Meta / Hero / Pricing
2. Benchmark / Proof section
3. Architecture / Docker note
4. Use cases / support copy
5. Финальная факт-проверка всех public claims

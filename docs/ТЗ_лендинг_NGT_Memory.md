# Техническое задание: Лендинг NGT Memory

Актуальный документ для точечных правок существующего сайта: `docs/SITE_UPDATE_TZ_2026-03.md`.
Этот файл сохраняется как базовое ТЗ лендинга, но цифры и технические факты ниже приведены к текущему состоянию проекта.

## 1. Общие сведения

- **Проект:** NGT Memory — слой долгосрочной памяти для LLM-приложений
- **Тип сайта:** одностраничный лендинг (Landing Page)
- **Цель:** привлечение разработчиков, знакомство с продуктом, конверсия в GitHub Stars / установки
- **Целевая аудитория:** разработчики Python/AI, строящие приложения на LLM (chatbots, AI-агенты, RAG-системы)
- **Язык контента:** английский (основной), русский (опционально)
- **URL:** ngt-memory.ru / ngt-memory.io (регистрируется отдельно)

---

## 2. Технические требования

| Параметр | Требование |
|---|---|
| Фреймворк | Next.js 14+ или Astro (статическая генерация) |
| Стилизация | TailwindCSS |
| Компоненты | shadcn/ui |
| Иконки | Lucide Icons |
| Анимации | Framer Motion (легкие) |
| Адаптивность | Mobile-first, breakpoints: 375 / 768 / 1280 / 1440px |
| Производительность | Lighthouse Score ≥ 90 по всем метрикам |
| Деплой | Vercel / Netlify (статика) |
| Аналитика | Plausible или Umami (privacy-first) |

---

## 3. Структура страницы (секции по порядку)

### 3.1 Hero-секция

**Цель:** захватить внимание, донести суть продукта за 5 секунд.

**Контент:**
- Логотип / название: **NGT Memory**
- Headline: `Give your LLM a memory it never forgets`
- Subheadline: `Drop-in REST API that adds persistent, cross-session memory to any LLM application. 2ms retrieval. Zero infrastructure.`
- CTA-кнопки:
  - Основная: `Get Started` → ведёт на секцию Quick Start
  - Вторичная: `View on GitHub` → `https://github.com/ngt-memory/ngt-memory`
- Бейджи под кнопками: версия `v0.23.0`, лицензия `Apache 2.0`, Python `3.10+`
- **Визуал:** анимированная схема: `User message → NGT Memory → LLM → Response` с подсветкой блока памяти

---

### 3.2 Социальное доказательство / бенчмарк

**Цель:** сразу показать измеримый результат.

**Контент:**
- Заголовок: `Proven results — not just promises`
- Три карточки метрик:
  - `+100%` — Quality improvement vs no-memory baseline
  - `~2-3ms` — Memory retrieval latency
  - `5/6` — Realistic scenario wins with memory
- Таблица сравнения (из Exp 44):

| Mode | Factual score | Keyword hit |
|---|---|---|
| **NGT Memory** | **2.44 / 3** | **44%** |
| No memory | 1.22 / 3 | 27% |

---

### 3.3 Проблема → Решение

**Цель:** установить эмоциональный контакт с болью разработчика.

**Контент (две колонки: ❌ Без памяти / ✅ С NGT Memory):**

| Без памяти | С NGT Memory |
|---|---|
| LLM рекомендует мясо вегетарианцу | Помнит предпочтения пользователя |
| Просит повторить контекст каждый раз | Сохраняет факты между сессиями |
| Generic советы без учёта истории | Персонализированные ответы |
| Опасные советы в medical/финансах | Учитывает аллергии, препараты, ограничения |

**Реальный пример** (из README):
> ❌ *"Ippudo is great for ramen lovers"* — рекомендация мяса вегетарианцу
>
> ✅ *"Shigetsu at Tenryu-ji serves shojin ryori (Buddhist vegan cuisine)"* — с памятью

---

### 3.4 Как это работает (Architecture)

**Цель:** объяснить архитектуру просто, без перегрузки деталями.

**Контент:**
- Заголовок: `How NGT Memory works`
- Визуальная схема пайплайна (горизонтальная на desktop, вертикальная на mobile):

```
POST /chat
    ↓
Embed (text-embedding-3-small)  ~700ms
    ↓
NGT Retrieve (cosine + graph)   ~2-3ms  ← Your advantage
    ↓
[MEMORY CONTEXT] → LLM prompt
    ↓
gpt-4.1-nano response           ~800-1500ms
    ↓
Store to NGT Memory             ~1ms
```

- Три механизма памяти (иконки + краткое описание):
  1. **Cosine similarity** — семантически близкие факты
  2. **Hebbian graph** — ассоциативные связи между концептами
  3. **Hierarchical consolidation** — важные факты в долгосрочную память

---

### 3.5 Quick Start / Code Demo

**Цель:** показать простоту интеграции — разработчик должен увидеть себя использующим это через 5 минут.

**Контент:**
- Заголовок: `Up and running in 5 minutes`
- Табы: `Docker` / `Local` / `Python Client`

**Docker:**
```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker-compose up -d
```

**Python Client:**
```python
import httpx

client = httpx.Client(base_url="http://localhost:9190")

# Store context
client.post("/store", json={
    "text": "User is vegetarian, allergic to penicillin",
    "session_id": "user_42"
})

# Chat with memory
r = client.post("/chat", json={
    "message": "What restaurants in Kyoto?",
    "session_id": "user_42"
})
# → Recommends vegetarian restaurants automatically
```

- Кнопка: `View full API docs →`

---

### 3.6 Возможности (Features)

**Цель:** полный список функциональных возможностей.

**Формат:** сетка 3×3 карточек с иконкой, заголовком и одной строкой описания.

| Иконка | Заголовок | Описание |
|---|---|---|
| 🧠 | Persistent Memory | Сохраняет факты между сессиями |
| ⚡ | ~2-3ms Retrieval | Граф + косинусный поиск без внешней БД |
| 🔌 | Drop-in REST API | Интегрируется за 5 минут |
| 👥 | Multi-session | Изолированная память на пользователя |
| 🐳 | Docker Ready | Один команда — полный деплой |
| 🔒 | Local-first | Работает без облака |
| 🕸️ | Hebbian Graph | Ассоциативные связи как в мозге |
| 📊 | Built-in Analytics | Метрики и статистика сессии |
| 🔑 | API Key Auth | Опциональная защита эндпоинтов |

---

### 3.7 Сравнение с аналогами

**Цель:** позиционирование против конкурентов.

**Контент:** таблица сравнения:

| | **NGT Memory** | Mem0 | Zep | LangChain Memory |
|---|---|---|---|---|
| Self-hosted | ✅ | ✅ | ✅ | ✅ |
| No vector DB required | ✅ | ❌ | ❌ | ❌ |
| Hebbian graph | ✅ | ❌ | ❌ | ❌ |
| Retrieval latency | **~2-3ms** | ~50ms | ~100ms | ~30ms |
| Open source | ✅ | ✅ | ✅ | ✅ |
| REST API | ✅ | ✅ | ✅ | ❌ |

---

### 3.8 Use Cases

**Цель:** показать применимость в реальных сценариях.

**Контент:** три карточки:

1. **Medical AI Assistant**
   > Помнит аллергии, препараты, анамнез пациента между сессиями

2. **Personal AI Companion**
   > Запоминает предпочтения, планы, важные события пользователя

3. **Customer Support Bot**
   > Помнит историю обращений, предпочтения, предыдущие решения

---

### 3.9 Конфигурация (опционально — для технической аудитории)

**Контент:** таблица переменных окружения из README (свёрнутая по умолчанию / accordion).

---

### 3.10 CTA-секция (финальная)

**Заголовок:** `Ready to give your LLM a memory?`
**Кнопки:**
- `Star on GitHub` → `https://github.com/ngt-memory/ngt-memory`
- `Read the Docs` → ссылка на документацию

---

### 3.11 Footer

- Логотип + название
- Ссылки: GitHub · Документация · Лицензия (Apache 2.0) · Реестр российского ПО
- Copyright: `© 2026 NGT Memory. Apache 2.0 License.`

---

## 4. Дизайн-требования

### Цветовая схема
- **Тема:** тёмная (dark mode по умолчанию, light mode опционально)
- **Акцентный цвет:** фиолетовый / индиго (`#6366f1` — Indigo 500) или бирюзовый (`#06b6d4` — Cyan 500)
- **Фон:** `#0f0f0f` или `#0a0a0a`
- **Текст:** `#f8fafc`
- **Карточки:** `#1e1e2e` с `border: 1px solid #2e2e3e`

### Типографика
- **Заголовки:** Inter или Geist (bold 700-800)
- **Код:** JetBrains Mono или Fira Code
- **Тело:** Inter Regular 400/500

### Визуальный стиль
- Минималистичный, технический, "developer-first"
- Подсветка синтаксиса кода (Shiki или Prism)
- Subtle gradient glow на hero-секции
- Нет лишних иллюстраций — данные и код говорят сами

---

## 5. SEO и мета-теги

```html
<title>NGT Memory — Persistent Memory Layer for LLM Applications</title>
<meta name="description" content="Drop-in REST API that adds persistent cross-session memory to any LLM. ~2-3ms retrieval, no vector DB required. Open source." />
<meta property="og:title" content="NGT Memory — Give your LLM a memory" />
<meta property="og:image" content="/og-image.png" />
```

**Ключевые слова:** LLM memory, RAG, persistent memory, AI memory layer, LangChain alternative, Mem0 alternative

---

## 6. Требования к производительности

- Размер бандла JS: < 100KB gzipped
- LCP (Largest Contentful Paint): < 1.5s
- Нет внешних шрифтов без self-hosting
- Изображения: WebP + lazy loading

---

## 7. Что НЕ входит в лендинг

- Личный кабинет / авторизация
- Платёжная система
- Блог
- Форма обратной связи (опционально — простая форма через Formspree)

---

## 8. Приёмочные критерии

- [ ] Все 11 секций реализованы
- [ ] Lighthouse Score ≥ 90 (Performance, Accessibility, SEO)
- [ ] Адаптивная вёрстка на 375 / 768 / 1280 / 1440px
- [ ] Рабочие ссылки на GitHub
- [ ] Подсветка синтаксиса в code-блоках
- [ ] Тёмная тема по умолчанию
- [ ] Deployed на Vercel/Netlify

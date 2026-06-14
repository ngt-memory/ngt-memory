# NGT Memory API — Production Hardening (v0.24.0)

Этот раздел описывает изменения версии 0.24.0, нацеленные на эксплуатацию
под нагрузкой. Все правки обратно совместимы: дефолтная конфигурация
(`NGT_SESSION_BACKEND=memory`, без rate limit) ведёт себя как 0.23.x,
но теперь корректно работает под конкурентной нагрузкой и не блокирует
event loop на дисковом I/O.

## Что изменилось

### 1. Конкурентность: лок на чтение + дисковый I/O вне event loop
`/chat`, `/store` и `/retrieve` выполняются под per-session async-локом.
Раньше `/retrieve` был без лока — параллельный запрос мог поймать
полупересобранный индекс памяти. Восстановление сессии с диска вынесено
в пул потоков (`asyncio.to_thread`), поэтому подъём крупной сессии не
останавливает обработку остальных запросов.

### 2. Глобальный бюджет памяти
Кроме лимита на число сессий, теперь есть суммарный лимит по всем сессиям —
по числу записей и по оценочному объёму RAM. При превышении вытесняются
старейшие незанятые сессии (с сохранением на диск, если включена
персистентность). Сессия, обрабатывающая запрос прямо сейчас, не вытесняется.

| Переменная | Назначение | Дефолт |
|---|---|---|
| `NGT_MAX_TOTAL_ENTRIES` | Суммарный лимит записей по всем сессиям | `200000` |
| `NGT_MAX_TOTAL_MB` | Суммарный бюджет RAM сессий, МБ | `2048` |

### 3. Безопасная персистентность (safetensors + JSON вместо pickle)
Состояние сессии сохраняется в трёх файлах:

```
{session}.memory.safetensors   # тензоры (embeddings)
{session}.memory.json          # entries / concepts / рёбра графа
{session}.session.json         # профиль, история, статистика
```

Формат версионирован (`format_version`), запись атомарна (tmp→rename).
Замена `torch.save(..., weights_only=False)` устраняет возможность
исполнения произвольного кода при загрузке файла сессии.

**Миграция.** Старые `.memory.pt`/`.session.pt` читаются автоматически
(с предупреждением в лог) и при следующем сохранении конвертируются в
новый формат. Отдельных действий не требуется — просто разверните 0.24.0
поверх существующего `NGT_PERSIST_DIR`. После прогрева можно удалить
оставшиеся `.pt` (если все сессии успели пересохраниться).

> ⚠️ Чтение легаси `.pt` использует небезопасный pickle. Применяйте только
> к файлам, созданным вашей же инсталляцией.

### 4. Redis backend для multi-worker
`NGT_SESSION_BACKEND=redis` хранит состояние сессий в Redis с TTL и
версионным счётчиком. Любой воркер может обслужить любой `session_id` —
sticky sessions не нужны. Per-worker hot-cache сверяет версию перед
использованием; конкурентный доступ к одной сессии сериализуется
распределённым локом (`SET NX PX` + безопасный release через Lua).

| Переменная | Назначение | Дефолт |
|---|---|---|
| `NGT_SESSION_BACKEND` | `memory` или `redis` | `memory` |
| `NGT_REDIS_URL` | URL Redis (в compose — `redis://redis:6379/0`) | `redis://localhost:6379/0` |
| `NGT_WORKERS` | Число uvicorn-воркеров (>1 только с redis) | `1` |

При `NGT_WORKERS>1` и `backend=memory` число воркеров принудительно
сбрасывается в 1 (сессии не шарятся между процессами) с предупреждением.

### 5. Rate limiting + Prometheus
Token-bucket лимитер на ключ (API-ключ, либо IP с учётом `X-Forwarded-For`).
При исчерпании — `429` с заголовком `Retry-After`. Метрики Prometheus
доступны на `GET /metrics`.

| Переменная | Назначение | Дефолт |
|---|---|---|
| `NGT_RATE_LIMIT_RPS` | Устойчивая частота, запросов/сек (0 = выкл.) | `0` |
| `NGT_RATE_LIMIT_BURST` | Размер всплеска (ёмкость ведра) | `20` |
| `NGT_METRICS_ENABLED` | Включить `/metrics` | `true` |

Экспортируемые метрики: `ngt_requests_total`, `ngt_request_duration_seconds`,
`ngt_active_sessions`, `ngt_memory_entries_total`, `ngt_llm_tokens_total`,
`ngt_memories_used`.

> In-process лимитер при multi-worker делит лимит между процессами.
> Для строгого глобального лимита используйте `nginx limit_req` или
> лимитер на уровне reverse-proxy.

## Новые зависимости

```
safetensors>=0.4        # правка 3 (обязательно для записи состояния)
prometheus-client>=0.20 # правка 5 (опционально — без неё /metrics → 501)
redis>=5.0              # правка 4 (нужна только при backend=redis)
```

Полный список — в `requirements-api.txt`.

## Профили деплоя

| Сценарий | Конфигурация | Готовность |
|---|---|---|
| Self-hosted, один пользователь | `backend=memory`, persist on, rate limit off | высокая |
| Команда за nginx | `backend=memory`, 1 воркер, rate limit on, metrics on | высокая |
| Публичный multi-tenant | `backend=redis`, N воркеров, rate limit on, metrics on | базовая — нужен ещё биллинг/квоты |

## Быстрый старт

```bash
# Self-hosted (1 воркер, in-memory + диск)
cp .env.example .env        # впишите OPENAI_API_KEY
docker compose up

# Multi-worker через Redis
echo "NGT_SESSION_BACKEND=redis"            >> .env
echo "NGT_REDIS_URL=redis://redis:6379/0"   >> .env
echo "NGT_WORKERS=4"                        >> .env
docker compose --profile redis up

# + Prometheus
docker compose --profile redis --profile metrics up
```

# NGT Memory — глубокая оценка качества памяти

Реальные end-to-end замеры на настоящем LLM (YandexGPT): не одна метрика, а
**профиль качества** памяти по семи осям — фактическая точность, устойчивость
к шуму, коррекция противоречий, отсутствие «ложной памяти», качество
retrieval, стоимость и персистентность.

> Старый одно-метричный отчёт (только recall, 8 сценариев) заменён этим
> расширенным. Простой бенч остаётся в `bench_memory_vs_baseline.py`,
> глубокий — в `deep_eval.py`.

Сырые данные: [`deep_latest.json`](./deep_latest.json).
Скрипты: [`deep_eval.py`](./deep_eval.py), [`bench_memory_vs_baseline.py`](./bench_memory_vs_baseline.py).
Датасет: [`bench_dataset.jsonl`](./bench_dataset.jsonl) (17 типизированных сценариев).

## Конфигурация прогона

| Параметр | Значение |
|---|---|
| Провайдер | YandexGPT (OpenAI-совместимый endpoint) |
| Chat-модель | `yandexgpt/latest` |
| Embedding | `text-search-doc/latest` (dim 256) |
| `memory_top_k` | 5 |
| Filler-ходов | 6 |
| Сценариев (e2e) | 17 |
| Дата | 2026-06-16 (UTC) |

## Главное

| Показатель | Без памяти | С памятью |
|---|---|---|
| **Фактическая точность** (recall+distractor+contradiction) | **0.308** | **0.923** |
| Прирост (lift) | — | **+0.615** |
| **Precision** (нет ложной памяти, контроль) | — | **1.000** |
| Латентность, мс (avg) | 1159 | 1456 |
| Токенов на вход (avg) | 267 | 403 |
| Токенов на выход (avg) | 84 | 107 |

Память поднимает фактическую точность с 31% до 92% и при этом **не подмешивает
неверные факты** на нерелевантных вопросах (precision 1.0). Цена — +26% к
латентности и +51% входных токенов (инъекция найденных фактов в промпт).

## По категориям

| Категория | n | Без памяти | С памятью | Примечание |
|---|---|---|---|---|
| **recall** (вспомнить факт) | 8 | 0.12 | **1.00** | главный сценарий |
| **distractor** (1 факт среди ~10 шумов) | 2 | 0.50 | 0.50 | см. честный промах ниже |
| **contradiction** (исправить пользователя) | 3 | 0.67 | **1.00** | |
| **control** (вопрос не по теме) | 4 | 1.00 | 1.00 | leak-rate 0%, precision 1.00 |

## Retrieval (граф вкл vs выкл)

Реальные эмбеддинги, набор `dataset.jsonl` (8 сценариев), `top_k=5`:

| Режим | recall@5 | precision@5 | MRR | hit@1 |
|---|---|---|---|---|
| graph_on | 1.000 | 0.375 | 1.000 | 1.000 |
| graph_off | 1.000 | 0.375 | 1.000 | 1.000 |

На этом наборе прямой векторный поиск уже идеален (recall@5 = 1.0, факт всегда
#1), поэтому граф ассоциаций не даёт дополнительного прироста — честный
результат: вклад графа проявляется на более сложных, многошаговых запросах,
которых в текущем наборе нет.

## Персистентность

| Проверка | Результат |
|---|---|
| `load_state` после `save_state` | OK |
| Записи сохранены (entries 2 → 2) | да |
| Факт восстановлен после перезагрузки | да (retrieve нашёл «penicillin») |
| **Итог** | **PASS** |

## Примеры ответов

**Коррекция противоречия (`contradiction_sober`)** — «What wine should I pair with my dinner?»
- Без памяти: *"To suggest the best wine, I'd need to know what you're having for dinner…"*
- С памятью: *"I'm sorry, but you've mentioned that you've been completely sober for five years and don't drink any alcohol, so wine might not be suitable for you…"* — память исправляет пользователя.

**Нет ложной памяти (`control_recipe`)** — «How do I bake a simple banana bread?» (в памяти лежат факты про Rust/PostgreSQL)
- С памятью: даёт обычный рецепт бананового хлеба, **не упоминая** Rust/Postgres. Leak = no.

**Честный промах (`distractor_mobility`)** — «Recommend a fun weekend outing.» (факт: пользователь в инвалидном кресле, + 10 дистракторов)
- С памятью: *"Since you're a fan of board games, you might enjoy visiting a local game cafe…"* — retrieval поднял **дистрактор** (board games) вместо ключевого факта о доступности. Промах обоих режимов (base MISS, mem MISS) показывает реальный предел при расплывчатом запросе и большом шуме — тест не подкручен.

## Воспроизведение

```bash
# Полная глубокая оценка (конфиг из .env)
python -m eval.deep_eval --out eval/results

# Только e2e (без retrieval/persistence)
python -m eval.deep_eval --skip-retrieval --skip-persistence

# Машиночитаемо
python -m eval.deep_eval --json
```

## Оговорки

- **Оценка по ключевым словам** — воспроизводима и без ручной разметки, но
  грубая: иногда baseline «угадывает» ключевое слово в общем ответе (отсюда
  ненулевая точность baseline в recall/contradiction).
- **Небольшой набор** (17 сценариев) — это профиль поведения, а не
  статистически репрезентативная выборка. Расширяется через `bench_dataset.jsonl`.
- **distractor n=2** — мало; промах `distractor_mobility` показателен как
  направление для улучшения retrieval при сильном шуме.

---

## Приложение: простой бенч «с памятью vs без памяти»

Реальный end-to-end замер: помогает ли долговременная память NGT настоящему
LLM правильно отвечать на вопросы, ответ на которые зависит от фактов,
сказанных ранее в диалоге.

Сырые данные: [`bench_latest.json`](./bench_latest.json).
Скрипт: [`bench_memory_vs_baseline.py`](./bench_memory_vs_baseline.py).

## Конфигурация прогона

| Параметр | Значение |
|---|---|
| Провайдер | YandexGPT (OpenAI-совместимый endpoint) |
| Chat-модель | `yandexgpt/latest` |
| Embedding | `text-search-doc/latest` (dim 256) |
| `memory_top_k` | 5 |
| Graph retrieval | включён |
| Сценариев | 8 |
| Filler-ходов | 6 |
| Дата | 2026-06-16 (UTC) |

## Методика

Для каждого сценария из [`dataset.jsonl`](./dataset.jsonl):

1. Факты пользователя кладутся в **долговременную память** NGT (`store` +
   профиль), без вызова LLM — только эмбеддинги.
2. История чата заполняется нейтральным small-talk (filler), чтобы факты
   гарантированно выпали из окна последних 6 сообщений. Так baseline
   физически не видит фактов в краткосрочной истории — единственный способ их
   «вспомнить» это память NGT.
3. Один и тот же вопрос задаётся **дважды** с идентичным начальным состоянием:
   - **baseline** — `chat_no_memory()`: только последние 6 сообщений (filler);
   - **memory** — `chat()`: retrieval из памяти + профиль + история.
4. Ответ считается верным, если содержит любое из `expect_keywords` сценария.

Это честное разделение: факты вне окна, поэтому baseline принципиально не
имеет к ним доступа — ровно та ситуация, ради которой нужна внешняя память.

## Итог

| Метрика | Без памяти | С памятью |
|---|---|---|
| **Accuracy** (доля верных ответов) | **0.000** (0/8) | **1.000** (8/8) |
| Прирост (lift) | — | **+1.000** |
| Средняя латентность, мс | 957 | 1756 |
| Средне токенов на вход | 266 | 406 |

Память даёт **+100 п.п. точности** на этом наборе. Цена — выше латентность и
расход входных токенов (инъекция найденных фактов и профиля в промпт), что
ожидаемо и приемлемо.

## По сценариям

| Сценарий | Вопрос | Без памяти | С памятью |
|---|---|---|---|
| allergy | What antibiotics should I avoid? | MISS | HIT |
| diet | Can you suggest a protein source for me? | MISS | HIT |
| location | What should I consider when buying a car for my climate? | MISS | HIT |
| profession | What kind of schedule advice fits my job? | MISS | HIT |
| family | Any recommendations for a family vacation? | MISS | HIT |
| pet | What should I keep in mind for my pet's exercise? | MISS | HIT |
| tech | What ORM or query library fits my stack? | MISS | HIT |
| fitness | How should I plan my nutrition? | MISS | HIT |

## Примеры ответов

**allergy** — «What antibiotics should I avoid?»
- Без памяти: *"I'm not a medical professional, so I can't provide personalized medical advice. It's best to consult with a healthcare provider…"* — общий ответ, факт об аллергии потерян.
- С памятью: *"Given that you're allergic to penicillin and had a bad reaction to amoxicillin, it's advisable to avoid penicillin-based antibiotics…"* — использует сохранённый факт.

**tech** — «What ORM or query library fits my stack?»
- Без памяти: *"…I'd need to know more about the technologies you're using. Could you please tell me which programming language and database…"*
- С памятью: *"Since you're using PostgreSQL as your database and building a backend in Rust with the Axum framework, you might consider…"*

**pet** — «What should I keep in mind for my pet's exercise?»
- Без памяти: *"…consider their breed, age, and health condition. Different pets have different needs…"*
- С памятью: *"Considering that Max has arthritis in his hind legs, it's important to focus on low-impact exercises that won't strain his joints…"*

## Воспроизведение

```bash
# конфиг берётся из .env (тот же провайдер, что у сервера)
python -m eval.bench_memory_vs_baseline --out eval/results

# машиночитаемый вывод
python -m eval.bench_memory_vs_baseline --json
```

## Оговорки

- **Оценка по ключевым словам** — грубая, но воспроизводимая и без ручной
  разметки. Возможны редкие ложные срабатывания, если baseline случайно
  угадает ключевое слово; в данном прогоне этого не произошло (0/8).
- **Маленький набор** (n=8) — это демонстрация различия, а не статистически
  репрезентативная выборка. Для усиления — расширить `dataset.jsonl`.
- **baseline = 0%** объясняется методикой: факты намеренно вынесены за окно
  последних сообщений. Это показывает ценность именно долговременной памяти,
  а не разницу в «уме» модели.

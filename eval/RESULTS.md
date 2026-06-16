# NGT Memory — результаты бенчмарка «с памятью vs без памяти»

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

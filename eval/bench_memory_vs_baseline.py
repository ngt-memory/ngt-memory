"""
NGT Memory — end-to-end бенчмарк «с памятью vs без памяти».

ЗАЧЕМ. run_eval.py меряет retrieval (всплывает ли факт в топ-K). Этот скрипт
идёт дальше: меряет, помогает ли память РЕАЛЬНОМУ LLM правильно ответить.

МЕТОДИКА (на каждый сценарий dataset.jsonl):
  1. Факты пользователя кладутся в ДОЛГОСРОЧНУЮ память (store + profile),
     без вызова LLM — только embeddings.
  2. История чата заполняется филлером (small-talk), чтобы факты гарантированно
     выпали из окна последних 6 сообщений. Так baseline физически не видит
     фактов в краткосрочной истории — единственный способ их «вспомнить» —
     это память NGT.
  3. Один и тот же вопрос задаётся ДВАЖДЫ с идентичным начальным состоянием:
       - baseline:  chat_no_memory()  — только последние 6 сообщений (филлер)
       - memory:    chat()            — retrieval из памяти + профиль + история
  4. Ответ считается верным, если содержит любое из expect_keywords сценария.

МЕТРИКИ:
  accuracy_baseline — доля верных ответов БЕЗ памяти
  accuracy_memory   — доля верных ответов С памятью
  lift              — accuracy_memory - accuracy_baseline (главное число)
  + средняя латентность и токены каждого режима

Запуск (конфиг берётся из .env через api.config.settings — тот же провайдер,
что и у сервера, напр. YandexGPT):
  python -m eval.bench_memory_vs_baseline
  python -m eval.bench_memory_vs_baseline --json
  python -m eval.bench_memory_vs_baseline --out eval/results
  python -m eval.bench_memory_vs_baseline --filler 8 --limit 4

ВАЖНО: расходует токены провайдера (по 2 chat-вызова на сценарий).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).parent
_DATASET = _HERE / "dataset.jsonl"

# Нейтральный small-talk, чтобы вытеснить факты из окна последних сообщений.
_FILLER_TURNS = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("What's the weather like in general?", "I can't check live weather, but I'm happy to chat."),
    ("Tell me a fun fact.", "Honey never spoils — archaeologists found edible honey in ancient tombs."),
    ("Nice. Do you like puzzles?", "I enjoy helping with them! Got one in mind?"),
    ("Maybe later.", "Sounds good — just let me know."),
    ("What can you do?", "I can answer questions, brainstorm, and help with tasks."),
    ("Cool.", "Glad to hear it!"),
    ("Let's continue.", "Of course — what would you like to talk about?"),
]


def _is_correct(response: str, keywords: List[str]) -> bool:
    low = (response or "").lower()
    return any(kw.lower() in low for kw in keywords)


def _build_wrapper():
    """Создаёт обёртку из той же конфигурации, что и сервер (api.config.settings)."""
    from api.config import settings
    from ngt.core.llm_wrapper import NGTMemoryLLMWrapper

    return NGTMemoryLLMWrapper(
        openai_api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url or None,
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        memory_top_k=settings.memory_top_k,
        memory_threshold=settings.memory_threshold,
        use_graph=settings.use_graph,
    )


def teach_and_fill(wrapper, facts: List[str], distractors: List[str], filler: int) -> List[Dict]:
    """Учит факты+дистракторы в долгосрочную память и строит историю чата,
    где факты вынесены за окно последних 6 сообщений филлером.

    Возвращает построенную историю (для восстановления между прогонами).
    """
    # 1. Факты → долгосрочная память + профиль (без вызова LLM).
    for i, fact in enumerate(facts):
        emb = wrapper._embed(fact)
        wrapper._store(fact, emb, role="user", turn=i)
        wrapper.profile.extract_and_update(fact, confidence=1.0, source="user_explicit")
    # Дистракторы тоже в память — реалистичный шум.
    for j, dtext in enumerate(distractors):
        emb = wrapper._embed(dtext)
        wrapper._store(dtext, emb, role="user", turn=100 + j)
    wrapper.memory.flush_hebbian()

    # 2. История: факты как «давние» реплики, затем филлер.
    history: List[Dict] = []
    for fact in facts:
        history.append({"role": "user", "content": fact})
        history.append({"role": "assistant", "content": "Got it, I'll remember that."})
    for u, a in _FILLER_TURNS[:max(filler, 4)]:
        history.append({"role": "user", "content": u})
        history.append({"role": "assistant", "content": a})
    wrapper._chat_history = list(history)
    return history


def eval_scenario(scenario: Dict, filler: int, verbose: bool) -> Dict:
    """Прогоняет один сценарий: учит факты в память, вытесняет их филлером,
    задаёт вопрос с памятью и без, сравнивает."""
    wrapper = _build_wrapper()
    history = teach_and_fill(wrapper, scenario["facts"], scenario.get("distractors", []), filler)

    query = scenario["query"]
    keywords = scenario["expect_keywords"]

    # 3a. BASELINE — без памяти (read-only, не мутирует состояние).
    base = wrapper.chat_no_memory(query)
    base_resp = base.get("response", "") or ""
    base_hit = _is_correct(base_resp, keywords)

    # 3b. MEMORY — с памятью (восстанавливаем то же начальное состояние истории).
    wrapper._chat_history = list(history)
    mem = wrapper.chat(query)
    mem_resp = mem.get("response", "") or ""
    mem_hit = _is_correct(mem_resp, keywords)

    if verbose:
        b = "HIT " if base_hit else "MISS"
        m = "HIT " if mem_hit else "MISS"
        print(f"  {scenario['id']:12s}  baseline {b}   memory {m}   "
              f"(mem used {len(mem.get('memories_used', []))})")

    return {
        "id": scenario["id"],
        "query": query,
        "expect_keywords": keywords,
        "baseline": {
            "hit": base_hit,
            "response": base_resp,
            "tokens_in": base.get("tokens_in", 0),
            "tokens_out": base.get("tokens_out", 0),
            "latency_ms": round(base.get("latency_ms", 0.0), 1),
        },
        "memory": {
            "hit": mem_hit,
            "response": mem_resp,
            "memories_used": len(mem.get("memories_used", [])),
            "tokens_in": mem.get("tokens_in", 0),
            "tokens_out": mem.get("tokens_out", 0),
            "latency_ms": round(mem.get("latency_ms", 0.0), 1),
        },
    }


def run(filler: int = 6, limit: Optional[int] = None,
        dataset_path: Path = _DATASET, verbose: bool = True) -> Dict:
    scenarios = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        scenarios = scenarios[:limit]

    per: List[Dict] = []
    for sc in scenarios:
        per.append(eval_scenario(sc, filler=filler, verbose=verbose))

    n = len(per)
    acc_base = sum(1 for r in per if r["baseline"]["hit"]) / n
    acc_mem = sum(1 for r in per if r["memory"]["hit"]) / n

    def _avg(path_a, path_b):
        return round(sum(r[path_a][path_b] for r in per) / n, 1)

    agg = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": n,
        "filler_turns": max(filler, 4),
        "accuracy_baseline": round(acc_base, 3),
        "accuracy_memory": round(acc_mem, 3),
        "lift": round(acc_mem - acc_base, 3),
        "avg_latency_ms": {
            "baseline": _avg("baseline", "latency_ms"),
            "memory": _avg("memory", "latency_ms"),
        },
        "avg_tokens_in": {
            "baseline": _avg("baseline", "tokens_in"),
            "memory": _avg("memory", "tokens_in"),
        },
        "avg_tokens_out": {
            "baseline": _avg("baseline", "tokens_out"),
            "memory": _avg("memory", "tokens_out"),
        },
        "results": per,
    }
    return agg


def _model_info() -> Dict:
    from api.config import settings
    provider = "OpenAI"
    if settings.openai_base_url and "yandex" in settings.openai_base_url.lower():
        provider = "YandexGPT"
    elif settings.openai_base_url:
        provider = settings.openai_base_url
    return {
        "provider": provider,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "memory_top_k": settings.memory_top_k,
        "memory_threshold": settings.memory_threshold,
        "use_graph": settings.use_graph,
    }


def main():
    parser = argparse.ArgumentParser(description="NGT Memory end-to-end: memory vs baseline")
    parser.add_argument("--filler", type=int, default=6,
                        help="Сколько филлер-ходов вставить (>=4), чтобы вытеснить факты из окна")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число сценариев")
    parser.add_argument("--json", action="store_true", help="Машиночитаемый вывод")
    parser.add_argument("--out", type=str, default=None,
                        help="Папка для сохранения JSON-результата с таймстампом")
    args = parser.parse_args()

    info = _model_info()
    if not args.json:
        print(f"Provider: {info['provider']} | chat={info['chat_model']} | "
              f"emb={info['embedding_model']} (dim {info['embedding_dim']}) | "
              f"top_k={info['memory_top_k']} graph={info['use_graph']}\n")

    agg = run(filler=args.filler, limit=args.limit, verbose=not args.json)
    agg["model"] = info

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"bench_{stamp}.json"
        out_file.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nСохранено: {out_file}", file=sys.stderr)

    if args.json:
        print(json.dumps(agg, ensure_ascii=False, indent=2))
    else:
        print(f"\n── Итог ({agg['scenarios']} сценариев, filler={agg['filler_turns']}) ─────────────")
        print(f"  accuracy без памяти:  {agg['accuracy_baseline']:.3f}")
        print(f"  accuracy с памятью:   {agg['accuracy_memory']:.3f}")
        print(f"  прирост (lift):       {agg['lift']:+.3f}")
        print(f"  латентность ms:       baseline {agg['avg_latency_ms']['baseline']} | "
              f"memory {agg['avg_latency_ms']['memory']}")
        print(f"  tokens_in (avg):      baseline {agg['avg_tokens_in']['baseline']} | "
              f"memory {agg['avg_tokens_in']['memory']}")
    return agg


if __name__ == "__main__":
    main()

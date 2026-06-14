"""
NGT Memory — eval-харнесс для retrieval.

ЗАЧЕМ. До этого не было способа ответить «помогает ли память?». Демо
memory-vs-no-memory качественное, а не измеримое. Этот харнесс превращает
вопрос в числа: на наборе сценариев (факты + отвлекающие записи + запрос
+ ожидаемые ключевые слова) меряет, всплывает ли нужный факт в топ-K
retrieval, не утонув среди дистракторов.

МЕТРИКИ:
  recall@k   — доля сценариев, где хотя бы один релевантный факт попал в топ-K
  precision@k— доля релевантных среди возвращённых
  MRR        — средний обратный ранг первого релевантного факта (1.0 = всегда #1)
  hit@1      — доля сценариев, где релевантный факт оказался самым первым

Запуск:
  # с реальным OpenAI (полноценный eval на настоящих embeddings):
  OPENAI_API_KEY=sk-... python -m eval.run_eval

  # без ключа (детерминированный псевдо-embedding — проверка пайплайна и графа,
  # НЕ качества реальных embeddings):
  python -m eval.run_eval --fake-embeddings

  # сравнить два набора порогов:
  python -m eval.run_eval --top-k 3 --threshold 0.2

Эталон считается по ключевым словам: факт «релевантен», если содержит любое
из expect_keywords сценария. Это грубо, но воспроизводимо и не требует
ручной разметки рангов.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

# Путь к датасету рядом с этим файлом
_DATASET = Path(__file__).parent / "dataset.jsonl"


# ── Embedding-провайдеры ─────────────────────────────────────────────

def make_openai_embedder(model: str = "text-embedding-3-small", dim: int = 1536) -> Callable[[str], torch.Tensor]:
    """Реальный OpenAI-embedder. Требует OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI()

    def embed(text: str) -> torch.Tensor:
        resp = client.embeddings.create(model=model, input=text[:8000], encoding_format="float")
        v = torch.tensor(resp.data[0].embedding, dtype=torch.float32)
        return v / v.norm()

    return embed


def make_fake_embedder(dim: int = 256) -> Callable[[str], torch.Tensor]:
    """Детерминированный псевдо-embedding для проверки пайплайна без сети.

    Кодирует пересечение по словам: вектор строится из хэшей токенов, так что
    тексты с общими словами оказываются ближе по косинусу. Это НЕ семантика —
    «doctor» и «physician» останутся далёкими, — но достаточно, чтобы проверить,
    что retrieval/граф поднимают записи с лексическим пересечением и топят
    непересекающиеся дистракторы.
    """
    def _tokens(text: str) -> List[str]:
        return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2]

    def embed(text: str) -> torch.Tensor:
        v = torch.zeros(dim)
        toks = _tokens(text)
        if not toks:
            return v + 1e-6
        for tok in toks:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            # каждый токен зажигает несколько координат (разреженный bag-of-words)
            for k in range(3):
                idx = (h >> (k * 16)) % dim
                v[idx] += 1.0
        n = v.norm()
        return v / n if n > 0 else v + 1e-6

    return embed


# ── Eval core ────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hit_at_1: bool
    relevant_ranks: List[int] = field(default_factory=list)
    retrieved_texts: List[str] = field(default_factory=list)


def _is_relevant(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def eval_scenario(scenario: Dict, embed: Callable[[str], torch.Tensor],
                  embedding_dim: int, top_k: int, threshold: float,
                  use_graph: bool) -> ScenarioResult:
    """Прогоняет один сценарий: грузит факты+дистракторы в свежую память,
    делает retrieval по запросу, считает метрики против expect_keywords."""
    from ngt.core.llm_memory import NGTMemoryForLLM

    memory = NGTMemoryForLLM(embedding_dim=embedding_dim, max_entries=1000)

    # Загружаем факты и дистракторы вперемешку (порядок не должен решать)
    items = [("fact", t) for t in scenario["facts"]] + \
            [("distractor", t) for t in scenario.get("distractors", [])]
    for _kind, text in items:
        emb = embed(text)
        memory.store(embedding=emb, text=text, domain="eval")
    memory.flush_hebbian()

    # Retrieval по запросу
    q = embed(scenario["query"])
    results = memory.retrieve(q, top_k=top_k, use_graph=use_graph)
    results = [r for r in results if r.get("score", 0) >= threshold]

    keywords = scenario["expect_keywords"]
    retrieved_texts = [r.get("text", "") for r in results]
    relevant_ranks = [i + 1 for i, t in enumerate(retrieved_texts) if _is_relevant(t, keywords)]

    n_relevant_total = sum(1 for t in scenario["facts"] if _is_relevant(t, keywords))
    n_relevant_total = max(n_relevant_total, 1)

    recall = (1.0 if relevant_ranks else 0.0)  # «хотя бы один факт всплыл»
    precision = (len(relevant_ranks) / len(results)) if results else 0.0
    rr = (1.0 / relevant_ranks[0]) if relevant_ranks else 0.0
    hit1 = bool(relevant_ranks and relevant_ranks[0] == 1)

    return ScenarioResult(
        id=scenario["id"], recall_at_k=recall, precision_at_k=precision,
        reciprocal_rank=rr, hit_at_1=hit1, relevant_ranks=relevant_ranks,
        retrieved_texts=retrieved_texts,
    )


def run_eval(embed: Callable[[str], torch.Tensor], embedding_dim: int,
             top_k: int = 5, threshold: float = 0.0, use_graph: bool = True,
             dataset_path: Path = _DATASET, verbose: bool = True) -> Dict:
    scenarios = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: List[ScenarioResult] = []
    for sc in scenarios:
        res = eval_scenario(sc, embed, embedding_dim, top_k, threshold, use_graph)
        results.append(res)
        if verbose:
            status = "✓" if res.recall_at_k > 0 else "✗"
            rank = f"rank={res.relevant_ranks[0]}" if res.relevant_ranks else "MISS"
            print(f"  {status} {res.id:12s} {rank:8s} precision={res.precision_at_k:.2f} rr={res.reciprocal_rank:.2f}")

    n = len(results)
    agg = {
        "scenarios": n,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
        "precision_at_k": sum(r.precision_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "hit_at_1": sum(1 for r in results if r.hit_at_1) / n,
        "top_k": top_k, "threshold": threshold, "use_graph": use_graph,
    }
    return agg


def main():
    parser = argparse.ArgumentParser(description="NGT Memory retrieval eval")
    parser.add_argument("--fake-embeddings", action="store_true",
                        help="Детерминированный псевдо-embedding (без сети)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-graph", action="store_true", help="Отключить graph retrieval")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--json", action="store_true", help="Вывести результат как JSON")
    args = parser.parse_args()

    use_real = not args.fake_embeddings and os.environ.get("OPENAI_API_KEY")
    if use_real:
        embed = make_openai_embedder(model=args.embedding_model)
        embedding_dim = 1536
        mode = f"OpenAI ({args.embedding_model})"
    else:
        if not args.fake_embeddings:
            print("OPENAI_API_KEY не задан → fake-embeddings (проверка пайплайна, не качества)\n", file=sys.stderr)
        embed = make_fake_embedder(dim=256)
        embedding_dim = 256
        mode = "fake (lexical, deterministic)"

    print(f"Embeddings: {mode} | top_k={args.top_k} | threshold={args.threshold} | graph={not args.no_graph}\n",
          file=sys.stderr if args.json else sys.stdout)
    agg = run_eval(embed, embedding_dim, top_k=args.top_k, threshold=args.threshold,
                   use_graph=not args.no_graph, verbose=not args.json)

    if args.json:
        print(json.dumps(agg, ensure_ascii=False, indent=2))
    else:
        print(f"\n── Итог ({agg['scenarios']} сценариев) ─────────────")
        print(f"  recall@{args.top_k}:    {agg['recall_at_k']:.3f}   (хотя бы один факт всплыл)")
        print(f"  precision@{args.top_k}: {agg['precision_at_k']:.3f}   (доля релевантных в выдаче)")
        print(f"  MRR:         {agg['mrr']:.3f}   (1.0 = факт всегда первый)")
        print(f"  hit@1:       {agg['hit_at_1']:.3f}   (факт оказался #1)")
    return agg


if __name__ == "__main__":
    main()

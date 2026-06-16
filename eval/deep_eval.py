"""
NGT Memory — глубокая оценка по нескольким осям + единый отчёт.

Считает не одну метрику, а профиль качества памяти:

  1. recall        — фактическая точность (нужный факт в ответе) с/без памяти
  2. distractor    — устойчивость к шуму (1 факт среди ~10 дистракторов)
  3. contradiction — коррекция: пользователь утверждает обратное факту,
                     исправляет ли модель, опираясь на память
  4. control       — precision/«нет ложной памяти»: вопрос НЕ по теме фактов;
                     память не должна подмешивать неверные факты (leak), но
                     должна нормально ответить на сам вопрос
  5. retrieval     — recall@k / precision@k / MRR / hit@1 (граф вкл vs выкл)
  6. cost          — латентность и токены: overhead памяти vs baseline
  7. persistence   — факты переживают save/load (рестарт сессии)

Конфиг берётся из .env через api.config.settings (тот же провайдер, что у
сервера). Запуск:

  python -m eval.deep_eval                 # полный отчёт + JSON
  python -m eval.deep_eval --out eval/results
  python -m eval.deep_eval --skip-retrieval --skip-persistence

ВАЖНО: расходует токены (по 2 chat-вызова на e2e-сценарий + эмбеддинги).
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from eval.bench_memory_vs_baseline import (
    _build_wrapper, _is_correct, teach_and_fill, _model_info,
)

_HERE = Path(__file__).parent
_BENCH_DATASET = _HERE / "bench_dataset.jsonl"
_RETRIEVAL_DATASET = _HERE / "dataset.jsonl"


def _run_e2e(scenario: Dict, filler: int) -> Dict:
    """Один e2e-сценарий: baseline (без памяти) и memory (с памятью)."""
    wrapper = _build_wrapper()
    history = teach_and_fill(wrapper, scenario["facts"], scenario.get("distractors", []), filler)
    query = scenario["query"]

    base = wrapper.chat_no_memory(query)
    base_resp = base.get("response", "") or ""

    wrapper._chat_history = list(history)
    mem = wrapper.chat(query)
    mem_resp = mem.get("response", "") or ""

    expect = scenario.get("expect_keywords", [])
    forbid = scenario.get("forbid_keywords", [])

    return {
        "id": scenario["id"],
        "type": scenario["type"],
        "query": query,
        "expect_keywords": expect,
        "forbid_keywords": forbid,
        "baseline": {
            "response": base_resp,
            "hit": _is_correct(base_resp, expect),
            "leak": _is_correct(base_resp, forbid) if forbid else False,
            "tokens_in": base.get("tokens_in", 0),
            "tokens_out": base.get("tokens_out", 0),
            "latency_ms": round(base.get("latency_ms", 0.0), 1),
        },
        "memory": {
            "response": mem_resp,
            "hit": _is_correct(mem_resp, expect),
            "leak": _is_correct(mem_resp, forbid) if forbid else False,
            "memories_used": len(mem.get("memories_used", [])),
            "tokens_in": mem.get("tokens_in", 0),
            "tokens_out": mem.get("tokens_out", 0),
            "latency_ms": round(mem.get("latency_ms", 0.0), 1),
        },
    }


def _section_e2e(filler: int, verbose: bool) -> Dict:
    scenarios = [json.loads(l) for l in _BENCH_DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    per = []
    for sc in scenarios:
        r = _run_e2e(sc, filler)
        per.append(r)
        if verbose:
            b = "HIT " if r["baseline"]["hit"] else "MISS"
            m = "HIT " if r["memory"]["hit"] else "MISS"
            extra = ""
            if r["type"] == "control":
                extra = f"  leak={'YES' if r['memory']['leak'] else 'no'}"
            print(f"  [{r['type']:13s}] {r['id']:20s} base {b}  mem {m}{extra}")

    def _by(types):
        return [r for r in per if r["type"] in types]

    def _rate(rows, side, key):
        return round(sum(1 for r in rows if r[side][key]) / len(rows), 3) if rows else None

    factual = _by({"recall", "distractor", "contradiction"})
    control = _by({"control"})

    by_type = {}
    for t in ["recall", "distractor", "contradiction", "control"]:
        rows = _by({t})
        if not rows:
            continue
        entry = {
            "n": len(rows),
            "accuracy_baseline": _rate(rows, "baseline", "hit"),
            "accuracy_memory": _rate(rows, "memory", "hit"),
        }
        if t == "control":
            entry["leak_rate_memory"] = _rate(rows, "memory", "leak")
            entry["no_false_memory_precision"] = round(1.0 - (entry["leak_rate_memory"] or 0.0), 3)
        by_type[t] = entry

    n_all = len(per)

    def _avg(side, key):
        return round(sum(r[side][key] for r in per) / n_all, 1)

    summary = {
        "scenarios": n_all,
        "filler_turns": max(filler, 4),
        "factual_accuracy_baseline": _rate(factual, "baseline", "hit"),
        "factual_accuracy_memory": _rate(factual, "memory", "hit"),
        "factual_lift": round((_rate(factual, "memory", "hit") or 0) - (_rate(factual, "baseline", "hit") or 0), 3),
        "no_false_memory_precision": round(1.0 - (_rate(control, "memory", "leak") or 0.0), 3) if control else None,
        "by_type": by_type,
        "cost": {
            "avg_latency_ms": {"baseline": _avg("baseline", "latency_ms"), "memory": _avg("memory", "latency_ms")},
            "avg_tokens_in": {"baseline": _avg("baseline", "tokens_in"), "memory": _avg("memory", "tokens_in")},
            "avg_tokens_out": {"baseline": _avg("baseline", "tokens_out"), "memory": _avg("memory", "tokens_out")},
        },
        "results": per,
    }
    return summary


def _section_retrieval(verbose: bool) -> Dict:
    """recall@k / precision@k / MRR / hit@1 на реальных эмбеддингах, граф вкл/выкл."""
    from api.config import settings
    from eval.run_eval import run_eval

    wrapper = _build_wrapper()
    embed = wrapper._embed  # корректный клиент (в т.ч. YandexGPT)
    dim = settings.embedding_dim
    top_k = settings.memory_top_k

    out = {}
    for use_graph in (True, False):
        agg = run_eval(embed, dim, top_k=top_k, threshold=settings.memory_threshold,
                       use_graph=use_graph, dataset_path=_RETRIEVAL_DATASET, verbose=False)
        key = "graph_on" if use_graph else "graph_off"
        out[key] = {
            "recall_at_k": agg["recall_at_k"],
            "precision_at_k": agg["precision_at_k"],
            "mrr": agg["mrr"],
            "hit_at_1": agg["hit_at_1"],
        }
        if verbose:
            print(f"  retrieval [{key:9s}] recall@{top_k}={agg['recall_at_k']:.3f} "
                  f"mrr={agg['mrr']:.3f} hit@1={agg['hit_at_1']:.3f}")
    out["top_k"] = top_k
    return out


def _section_persistence(verbose: bool) -> Dict:
    """Факты сохраняются и восстанавливаются после save/load (рестарт сессии)."""
    facts = [
        "My name is Anton and I am allergic to penicillin",
        "I live in Novosibirsk and I am a vegetarian",
    ]
    w1 = _build_wrapper()
    for i, f in enumerate(facts):
        w1._store(f, w1._embed(f), role="user", turn=i)
    w1.memory.flush_hebbian()
    before = w1.memory.num_entries

    tmp = Path(tempfile.mkdtemp(prefix="ngt_persist_"))
    base_path = tmp / "sess"
    w1.save_state(base_path)

    w2 = _build_wrapper()
    ok = w2.load_state(base_path)
    after = w2.memory.num_entries

    # Проверяем смысловое восстановление: запрос по сохранённому факту.
    q = w2._embed("What am I allergic to?")
    hits = w2.memory.retrieve(q, top_k=3)
    recovered = any("penicillin" in (h.get("text", "").lower()) for h in hits)

    result = {
        "load_ok": bool(ok),
        "entries_before": before,
        "entries_after": after,
        "entries_preserved": before == after,
        "fact_recovered_after_reload": bool(recovered),
        "passed": bool(ok and before == after and recovered),
    }
    if verbose:
        print(f"  persistence: load_ok={result['load_ok']} "
              f"entries {before}->{after} recovered={result['fact_recovered_after_reload']}")
    return result


def run(filler: int = 6, skip_retrieval: bool = False, skip_persistence: bool = False,
        verbose: bool = True) -> Dict:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": _model_info(),
    }
    if verbose:
        print("\n[1-4] End-to-end (recall / distractor / contradiction / control)")
    report["e2e"] = _section_e2e(filler, verbose)

    if not skip_retrieval:
        if verbose:
            print("\n[5] Retrieval quality (graph on vs off)")
        report["retrieval"] = _section_retrieval(verbose)

    if not skip_persistence:
        if verbose:
            print("\n[7] Persistence round-trip")
        report["persistence"] = _section_persistence(verbose)

    return report


def _print_summary(rep: Dict):
    e = rep["e2e"]
    print("\n" + "=" * 60)
    print("ГЛУБОКАЯ ОЦЕНКА — СВОДКА")
    print("=" * 60)
    print(f"Провайдер: {rep['model']['provider']} | chat={rep['model']['chat_model']} "
          f"| emb dim {rep['model']['embedding_dim']}")
    print(f"\nФактическая точность (recall+distractor+contradiction):")
    print(f"  без памяти: {e['factual_accuracy_baseline']:.3f}")
    print(f"  с памятью:  {e['factual_accuracy_memory']:.3f}   (lift {e['factual_lift']:+.3f})")
    print(f"\nПо категориям:")
    for t, v in e["by_type"].items():
        line = f"  {t:13s} n={v['n']}  base={v['accuracy_baseline']:.2f}  mem={v['accuracy_memory']:.2f}"
        if t == "control":
            line += f"  no-false-memory={v['no_false_memory_precision']:.2f}"
        print(line)
    if e.get("no_false_memory_precision") is not None:
        print(f"\nPrecision (нет ложной памяти на контроле): {e['no_false_memory_precision']:.3f}")
    c = e["cost"]
    print(f"\nСтоимость (avg):")
    print(f"  латентность ms:  baseline {c['avg_latency_ms']['baseline']} | memory {c['avg_latency_ms']['memory']}")
    print(f"  tokens_in:       baseline {c['avg_tokens_in']['baseline']} | memory {c['avg_tokens_in']['memory']}")
    print(f"  tokens_out:      baseline {c['avg_tokens_out']['baseline']} | memory {c['avg_tokens_out']['memory']}")
    if "retrieval" in rep:
        r = rep["retrieval"]
        print(f"\nRetrieval (top_k={r['top_k']}):")
        for key in ("graph_on", "graph_off"):
            m = r[key]
            print(f"  {key:9s}  recall@k={m['recall_at_k']:.3f}  precision@k={m['precision_at_k']:.3f}  "
                  f"mrr={m['mrr']:.3f}  hit@1={m['hit_at_1']:.3f}")
    if "persistence" in rep:
        p = rep["persistence"]
        print(f"\nPersistence: {'PASS' if p['passed'] else 'FAIL'} "
              f"(entries {p['entries_before']}->{p['entries_after']}, "
              f"recovered={p['fact_recovered_after_reload']})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NGT Memory deep evaluation")
    parser.add_argument("--filler", type=int, default=6)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-persistence", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rep = run(filler=args.filler, skip_retrieval=args.skip_retrieval,
              skip_persistence=args.skip_persistence, verbose=not args.json)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"deep_{stamp}.json"
        out_file.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nСохранено: {out_file}", file=sys.stderr)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_summary(rep)
    return rep


if __name__ == "__main__":
    main()

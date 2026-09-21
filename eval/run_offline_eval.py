import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.intent_classifier import classify_intent  # noqa: E402
from rag.evaluable_retriever import EvaluableHybridRetriever  # noqa: E402


def evaluate() -> dict:
    cases = json.loads((ROOT / "eval" / "fitness_golden.json").read_text(encoding="utf-8"))
    retriever = EvaluableHybridRetriever.from_directory(ROOT / "data")
    intent_correct = 0
    recalls = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    ndcg = []
    latencies = []

    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case["query"], top_k=5)
        latencies.append((time.perf_counter() - started) * 1000)
        intent_correct += classify_intent(case["query"]) == case["intent"]
        sources = [hit.source for hit in hits]
        rank = sources.index(case["source"]) + 1 if case["source"] in sources else 0
        for k in recalls:
            recalls[k] += bool(rank and rank <= k)
        reciprocal_ranks.append(1 / rank if rank else 0)
        ndcg.append(1 / math.log2(rank + 1) if rank else 0)

    total = len(cases)
    result = {
        "dataset_size": total,
        "intent_accuracy": round(intent_correct / total, 4),
        "recall_at_1": round(recalls[1] / total, 4),
        "recall_at_3": round(recalls[3] / total, 4),
        "recall_at_5": round(recalls[5] / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "ndcg_at_5": round(sum(ndcg) / total, 4),
        "latency_p95_ms": round(sorted(latencies)[math.ceil(total * 0.95) - 1], 3),
    }
    (ROOT / "eval" / "latest_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))

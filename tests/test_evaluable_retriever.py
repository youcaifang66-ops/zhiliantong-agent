from pathlib import Path

from rag.evaluable_retriever import EvaluableHybridRetriever


DATA = Path(__file__).resolve().parents[1] / "data"


def test_hybrid_retrieval_returns_citations():
    hits = EvaluableHybridRetriever.from_directory(DATA).search("深蹲后膝盖不舒服", top_k=3)
    assert hits
    assert hits[0].source in {"训练安全.txt", "动作知识.txt"}
    assert hits[0].channels


def test_plan_query_retrieves_training_principles():
    hits = EvaluableHybridRetriever.from_directory(DATA).search("新手每周练几次", top_k=3)
    assert any(hit.source == "训练计划原则.txt" for hit in hits)

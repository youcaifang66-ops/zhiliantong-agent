from pathlib import Path

from agent.fitness_workflow import build_fitness_workflow
from rag.evaluable_retriever import EvaluableHybridRetriever


def workflow():
    data = Path(__file__).resolve().parents[1] / "data"
    return build_fitness_workflow(EvaluableHybridRetriever.from_directory(data))


def test_workflow_emits_trace_and_citations():
    result = workflow().invoke(
        {"query": "新手每周练几次"}, config={"configurable": {"thread_id": "normal-1"}}
    )
    assert result["intent"] == "plan"
    assert result["steps"] == ["route_intent", "safety_guard", "hybrid_retrieval", "compose_response"]
    assert result["citations"]


def test_safety_branch_skips_retrieval():
    result = workflow().invoke(
        {"query": "训练时胸痛和呼吸困难"}, config={"configurable": {"thread_id": "risk-1"}}
    )
    assert result["safe"] is False
    assert "citations" not in result
    assert "医疗" in result["response"]

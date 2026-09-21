import json
from pathlib import Path


def test_golden_dataset_contains_50_labeled_queries():
    path = Path(__file__).resolve().parents[1] / "eval" / "fitness_golden.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 50
    assert all({"query", "intent", "source"} == set(case) for case in cases)

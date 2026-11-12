import json
from pathlib import Path

from rag_eval_kit import cli

ROOT = Path(__file__).resolve().parents[1]


def test_example_dataset_is_valid():
    assert cli.main(["validate", str(ROOT / "examples" / "questions.json")]) == 0


def test_example_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "ret.json"
    code = cli.main([
        "retrieval",
        "--adapter", "examples.keyword_adapter:KeywordAdapter",
        "--dataset", "examples/questions.json",
        "--out", str(out),
    ])
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert len(rows) == 4
    assert sum(r["metrics"]["mrr"] > 0 for r in rows) >= 3

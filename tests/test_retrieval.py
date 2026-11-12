import json

import pytest

from rag_eval_kit import retrieval

ITEM_A = {
    "id": "Q1",
    "category": "normal",
    "split": "dev",
    "question": "q?",
    "expected_outcome": "answer",
    "relevant": [{"source": "a.pdf", "evidence": "quick brown fox"}],
}
CHUNKS = [
    {"text": "nothing here", "source": "a.pdf"},
    {"text": "The quick, brown fox jumps.", "source": "a.pdf"},
    {"text": "quick brown fox", "source": "b.pdf"},
]


class Stub:
    version = "stub-v1"
    corpus_id = "c1"

    def __init__(self, ranked):
        self.ranked = ranked

    def retrieve(self, query):
        return self.ranked


class Raises:
    version = "boom"

    def retrieve(self, query):
        raise ConnectionError("index down")


def test_score_matches_harness_selfcheck():
    # same values as the original harness self-check, so the scoring cannot drift
    item = {
        **ITEM_A,
        "relevant": [
            {"source": "a.pdf", "evidence": "quick brown fox"},
            {"source": "a.pdf", "evidence": "absent span"},
        ],
    }
    result = retrieval.score(item, CHUNKS, (1, 2))
    assert result["recall@1"] == 0 and result["recall@2"] == 0.5
    assert result["success@2"] == 0 and result["mrr"] == 0.5
    assert retrieval.score({**item, "any_of": True}, CHUNKS, (2,))["recall@2"] == 1


def test_source_must_match_not_just_text():
    ranked = [CHUNKS[2]]  # right text, wrong document
    assert retrieval.score(ITEM_A, ranked, (1,))["recall@1"] == 0.0


def test_full_path_source_matches():
    ranked = [
        {"text": "the quick brown fox", "source": r"C:\docs\a.pdf"},
        {"text": "x", "source": "/srv/b.pdf"},
    ]
    assert retrieval.score(ITEM_A, ranked, (1,))["recall@1"] == 1.0
    ranked[0]["source"] = "/srv/a.pdf"
    assert retrieval.score(ITEM_A, ranked, (1,))["recall@1"] == 1.0


def test_empty_ranked_scores_zero():
    assert retrieval.score(ITEM_A, [], (1, 3)) == {
        "recall@1": 0.0,
        "success@1": 0.0,
        "recall@3": 0.0,
        "success@3": 0.0,
        "mrr": 0.0,
    }


def test_missing_source_names_item():
    with pytest.raises(ValueError, match="Q1.*source"):
        retrieval.run(Stub([{"text": "t"}]), [ITEM_A], (1,))


def test_adapter_error_is_recorded_and_run_continues():
    rows = retrieval.run(Raises(), [ITEM_A, {**ITEM_A, "id": "Q2"}], (1,))
    assert [r["id"] for r in rows] == ["Q1", "Q2"]
    assert all("index down" in r["error"] for r in rows)


def test_run_skips_abstain_items_and_records_ids():
    abstain = {**ITEM_A, "id": "Q9", "expected_outcome": "abstain", "relevant": []}
    rows = retrieval.run(Stub(CHUNKS), [ITEM_A, abstain], (1, 3), dataset_id="d123")
    assert [r["id"] for r in rows] == ["Q1"]
    row = rows[0]
    assert (row["dataset"], row["corpus"], row["version"]) == ("d123", "c1", "stub-v1")
    assert row["metrics"]["recall@3"] == 1.0 and row["metrics"]["mrr"] == 0.5


def test_invalid_dataset_stops_with_the_reason(tmp_path, capsys):
    from rag_eval_kit import cli

    data = tmp_path / "d.json"
    data.write_text(json.dumps([{**ITEM_A, "difficulty": "easy", "origin": "manual", "ci": False,
                                 "ground_truth": "g", "relevant": []}]))
    with pytest.raises(SystemExit):
        cli.main(["retrieval", "--adapter", "x:y", "--dataset", str(data)])
    assert "answer items need at least one relevant span" in capsys.readouterr().out


def test_retrieval_cli(tmp_path, monkeypatch):
    from rag_eval_kit import cli

    (tmp_path / "my_rag.py").write_text(
        "class A:\n"
        "    version = 'v'\n"
        "    def retrieve(self, q):\n"
        "        return [{'text': 'quick brown fox', 'source': 'a.pdf'}]\n"
    )
    data = tmp_path / "d.json"
    data.write_text(json.dumps([{**ITEM_A, "difficulty": "easy", "origin": "manual",
                                 "ci": False, "ground_truth": "g"}]))
    out = tmp_path / "out.json"
    monkeypatch.chdir(tmp_path)
    assert cli.main(["retrieval", "--adapter", "my_rag:A", "--dataset", str(data), "--out", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert rows[0]["metrics"]["recall@1"] == 1.0

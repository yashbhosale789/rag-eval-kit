import json

import pytest

from rag_eval_kit import judge, store

ITEMS = [
    {"id": "faq-3", "question": "q3", "ground_truth": "t3", "expected_outcome": "answer"},
    {"id": "faq-1", "question": "q1", "ground_truth": "t1", "expected_outcome": "abstain"},
]


def fake(replies):
    calls = []

    def call(prompt):
        calls.append(prompt)
        return replies.pop(0) if replies else '{"total": 1, "supported": 1}'

    return call, calls


def gens_file(tmp_path):
    path = tmp_path / "generations.jsonl"
    for item in ITEMS:
        store.append(path, {
            "key": store.key("baseline", item["id"], "v1"), "variant": "baseline", "qid": item["id"],
            "version": "v1", "answer": "a", "contexts": ["c"], "refused": False,
        })
    return path


def test_faithfulness_counts(monkeypatch):
    call, _ = fake(['```json\n{"total": 4, "supported": 3}\n```'])
    monkeypatch.setattr(judge, "judge_call", call)
    assert judge.run_metric("faithfulness", ITEMS[0], "a", ["c"]) == (0.75, {"total": 4, "supported": 3})


def test_bad_counts_give_up_after_two_tries(monkeypatch):
    bad = '{"total": 0, "covered": 0, "contradicts": false}'
    call, calls = fake([bad, bad])
    monkeypatch.setattr(judge, "judge_call", call)
    assert judge.run_metric("correctness", ITEMS[0], "a", ["c"]) == (None, {"error": "unparsable"})
    assert len(calls) == 2


def test_pending_follows_dataset_order(tmp_path):
    todo = judge.pending(ITEMS, gens_file(tmp_path), tmp_path / "j.jsonl", ["baseline"], "v1")
    assert [(g["qid"], m) for g, m in todo] == [
        ("faq-3", "faithfulness"), ("faq-3", "correctness"), ("faq-3", "relevancy"),
        ("faq-1", "faithfulness"), ("faq-1", "relevancy"),
    ]


def test_nothing_pending_makes_no_call(tmp_path, monkeypatch):
    gens, judged = gens_file(tmp_path), tmp_path / "j.jsonl"
    for g, m in judge.pending(ITEMS, gens, judged, ["baseline"], "v1"):
        store.append(judged, {"key": f"{g['key']}|{m}", "score": 1.0})
    call, calls = fake([])
    monkeypatch.setattr(judge, "judge_call", call)
    judge.judge_generations(ITEMS, gens, judged, ["baseline"], "v1")
    assert calls == []


def test_judge_records_carry_ids(tmp_path, monkeypatch):
    gens, judged = gens_file(tmp_path), tmp_path / "j.jsonl"
    call, _ = fake([])
    monkeypatch.setattr(judge, "judge_call", call)
    monkeypatch.setitem(judge.METRICS, "relevancy", lambda *a: (0.9, {}))
    judge.judge_generations(ITEMS, gens, judged, ["baseline"], "v1", limit=1)
    (record,) = store.load(judged).values()
    assert record["key"] == "baseline|faq-3|v1|faithfulness" and record["score"] == 1.0


def test_missing_settings_explain_themselves(monkeypatch):
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="JUDGE_BASE_URL"):
        judge.settings()


def test_auc():
    assert judge.auc([1.0, 1.0], [0.0, 0.0]) == 1.0 and judge.auc([0.5], [0.5]) == 0.5


def test_cli_judge_with_nothing_to_judge_needs_no_endpoint(tmp_path, monkeypatch, capsys):
    from rag_eval_kit import cli

    data = tmp_path / "d.json"
    data.write_text(json.dumps([{**ITEMS[0], "split": "dev", "ci": True, "category": "normal",
                                 "difficulty": "easy", "origin": "manual",
                                 "relevant": [{"source": "a.txt", "evidence": "e"}]}]))
    for var in ("JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["judge", "--dataset", str(data), "--version", "v1", "--runs", str(tmp_path)]) == 0
    assert "0 judge calls" in capsys.readouterr().out

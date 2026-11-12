import json

import pytest

from rag_eval_kit import compare, store


def test_stats_match_harness_selfcheck():
    # same values as the original harness self-check, so the maths cannot drift
    assert abs(compare.sign_test(8, 1) - 0.0390625) < 1e-9 and compare.sign_test(0, 0) == 1.0
    assert compare.bootstrap_ci([0.5] * 10, 0.05) == (0.5, 0.5)
    assert compare.verdict([0.0] * 10, 1.0, 0.0, 0.0, 0.05, 0.0).startswith("no evidence")
    assert compare.verdict([1.0] * 10, compare.sign_test(10, 0), 1.0, 1.0, 0.05, 0.0) == "A higher"
    assert "smaller" in compare.verdict([0.1] * 10, compare.sign_test(10, 0), 0.1, 0.1, 0.05, 0.5)


def ret_file(tmp_path, name, mrrs):
    rows = [{"id": qid, "metrics": {"mrr": m}} for qid, m in mrrs.items()]
    rows.append({"id": "err", "error": "boom"})
    path = tmp_path / name
    path.write_text(json.dumps(rows))
    return path


ITEMS = {q: {"id": q, "split": "dev", "expected_outcome": "answer"} for q in ("faq-3", "faq-1", "faq-2")}


def test_identical_retrieval_runs_show_no_difference(tmp_path):
    a = compare.load_ret(ret_file(tmp_path, "a.json", {"faq-3": 1.0, "faq-1": 0.5, "faq-2": 0.0}))
    assert "err" not in a
    result = compare.compare(a, a, ITEMS, "mrr")
    assert result["pairs"] == 3 and result["mean_diff"] == 0.0
    assert result["verdict"] == "no evidence of a difference at this sample size"


def test_pairs_follow_dataset_order_for_any_id_shape(tmp_path):
    a = compare.load_ret(ret_file(tmp_path, "a.json", {"faq-3": 1.0, "faq-1": 0.0, "faq-2": 0.5}))
    b = compare.load_ret(ret_file(tmp_path, "b.json", {"faq-3": 0.0, "faq-1": 0.0, "faq-2": 0.5}))
    result = compare.compare(a, b, ITEMS, "mrr")
    assert result["ids"] == ["faq-3", "faq-1", "faq-2"]
    assert result["wins"] == 1 and result["losses"] == 0


def test_too_few_pairs_is_an_error(tmp_path):
    a = compare.load_ret(ret_file(tmp_path, "a.json", {"faq-3": 1.0}))
    with pytest.raises(SystemExit, match="only 1 paired"):
        compare.compare(a, a, ITEMS, "mrr")


def test_load_gen_scores_behavior_and_judged_metrics(tmp_path):
    gens, judged = tmp_path / "g.jsonl", tmp_path / "j.jsonl"
    k = store.key("baseline", "faq-1", "v1")
    store.append(gens, {"key": k, "qid": "faq-1", "refused": False, "latency_ms": 10})
    store.append(judged, {"key": f"{k}|faithfulness", "score": 0.9})
    rows = compare.load_gen(gens, judged, "baseline", "v1", ITEMS)
    assert rows == {"faq-1": {"behavior": 1.0, "latency_ms": 10.0, "faithfulness": 0.9}}

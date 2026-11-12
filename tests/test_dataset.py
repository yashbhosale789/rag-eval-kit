from rag_eval_kit import dataset

EVIDENCE = "the spare tyre is under the boot floor"
VALID_ANSWER = {
    "id": "Q1",
    "category": "normal",
    "difficulty": "easy",
    "origin": "manual",
    "split": "dev",
    "ci": True,
    "question": "Where is the spare tyre?",
    "ground_truth": "Under the boot floor.",
    "expected_outcome": "answer",
    "relevant": [{"source": "a.pdf", "evidence": EVIDENCE}],
}
VALID_ABSTAIN = {
    **VALID_ANSWER,
    "id": "Q2",
    "category": "unanswerable",
    "expected_outcome": "abstain",
    "relevant": [],
}


def test_categories_include_new():
    assert {"multi_hop", "image", "table"} <= dataset.CATEGORIES


def test_check_rejects_abstain_with_spans():
    item = {**VALID_ABSTAIN, "relevant": [{"source": "a.pdf", "evidence": "x" * 30}]}
    errors, _ = dataset.check([item], None)
    assert any("abstain items must have relevant=[]" in e for e in errors)


def test_check_without_chunks_skips_span_check():
    assert dataset.check([VALID_ANSWER, VALID_ABSTAIN], None) == ([], [])


def test_check_with_chunks_finds_missing_evidence():
    chunks = {"a.pdf": [dataset.norm("something else entirely")]}
    errors, _ = dataset.check([VALID_ANSWER], chunks)
    assert any("evidence matches no chunk" in e for e in errors)


def test_duplicate_id_is_an_error():
    errors, _ = dataset.check([VALID_ANSWER, VALID_ANSWER], None)
    assert any("duplicate id" in e for e in errors)


def test_norm_and_basename():
    assert dataset.norm("Boot  Floor!") == "bootfloor"
    assert dataset.basename(r"C:\docs\a.pdf") == dataset.basename("/srv/a.pdf") == "a.pdf"


def test_validate_cli(tmp_path):
    import json

    from rag_eval_kit import cli

    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps([VALID_ANSWER, VALID_ABSTAIN]))
    bad.write_text(json.dumps([{**VALID_ANSWER, "split": "train"}]))
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(bad)]) == 1

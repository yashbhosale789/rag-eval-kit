import pytest

from rag_eval_kit import cli, gate, generate, store

PHRASE = "don't have enough information"
ITEMS = [
    {"id": "faq-3", "question": "q3", "split": "dev", "ci": True, "expected_outcome": "answer"},
    {"id": "faq-1", "question": "q1", "split": "dev", "ci": True, "expected_outcome": "abstain"},
]


class Bot:
    version = "bot-v1"
    corpus_id = "c"

    def __init__(self):
        self.calls = []

    def answer(self, query):
        self.calls.append(query)
        if query == "q1":
            return {"answer": "I don\u2019t have enough information.", "contexts": ["c"]}
        return {"answer": "Under the boot floor.", "contexts": ["ctx"]}


def test_is_refusal_matches_harness_selfcheck():
    assert generate.is_refusal({"outcome": "abstain", "contexts": ["c"], "answer": "x"}, PHRASE)
    assert generate.is_refusal(
        {"outcome": "answer", "contexts": ["c"], "answer": "I don\u2019t have enough information."}, PHRASE
    )
    assert not generate.is_refusal({"outcome": "answer", "contexts": ["c"], "answer": "Under the boot floor."}, PHRASE)
    assert generate.is_refusal({"answer": "anything", "contexts": []}, PHRASE)


def test_records_in_dataset_order_and_resume(tmp_path):
    out, bot = tmp_path / "g.jsonl", Bot()
    assert generate.run(bot, ITEMS, out) is False
    records = list(store.load(out).values())
    assert [r["qid"] for r in records] == ["faq-3", "faq-1"]
    assert [r["refused"] for r in records] == [False, True]
    first = records[0]
    assert first["key"] == "baseline|faq-3|bot-v1"
    assert (first["version"], first["corpus"], first["context_hash"]) == ("bot-v1", "c", store.context_hash(["ctx"]))
    generate.run(bot, ITEMS, out)
    assert bot.calls == ["q3", "q1"]  # rerun made no new calls


def test_answer_error_is_recorded_and_retried_next_run(tmp_path):
    class Flaky(Bot):
        def answer(self, query):
            raise TimeoutError("slow")

    out = tmp_path / "g.jsonl"
    assert generate.run(Flaky(), ITEMS, out) is True
    assert all("slow" in r["error"] for r in store.load(out).values())
    assert len(generate.pending(Bot(), ITEMS, out, "baseline")) == 2


def test_cli_refuses_to_call_without_yes(tmp_path, monkeypatch, capsys):
    import json

    (tmp_path / "bot.py").write_text(
        "class B:\n    version = 'v'\n    def answer(self, q):\n        raise AssertionError('called')\n"
    )
    data = tmp_path / "d.json"
    data.write_text(json.dumps([{**ITEMS[0], "category": "normal", "difficulty": "easy", "origin": "manual",
                                 "ground_truth": "g", "relevant": [{"source": "a.txt", "evidence": "e"}]}]))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["generate", "--adapter", "bot:B", "--dataset", str(data)]) == 2
    assert "1 adapter.answer calls" in capsys.readouterr().out


def test_gate_matches_harness_selfcheck():
    items = [{"id": "A", "expected_outcome": "answer"}, {"id": "B", "expected_outcome": "abstain"}]
    gens = {"v|A|f": {"key": "v|A|f", "refused": False}, "v|B|f": {"key": "v|B|f", "refused": True}}
    judged = {
        f"v|A|f|{m}": {"score": s}
        for m, s in (("faithfulness", 1.0), ("relevancy", 0.9), ("correctness", 1.0))
    }
    assert gate.evaluate(items, gens, judged, "v", "f", 0.85, 0.75)[:2] == ([], [])
    assert gate.evaluate(items, gens, judged, "v", "f", 1.01, 0.75)[1]
    refused = {**gens, "v|A|f": {"key": "v|A|f", "refused": True}}
    assert gate.evaluate(items, refused, judged, "v", "f", 0.85, 0.75)[1]
    assert gate.evaluate(items, {}, judged, "v", "f", 0.85, 0.75)[0]

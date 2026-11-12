"""CI gate: right refusal behaviour on every item, and judged quality above the floors."""

from rag_eval_kit import store

FAITHFULNESS_FLOOR = 0.85
RELEVANCY_FLOOR = 0.75


def evaluate(items, gens, judged, variant, version, faith_floor=FAITHFULNESS_FLOOR, rel_floor=RELEVANCY_FLOOR):
    problems: list[str] = []
    failures: list[str] = []
    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "relevancy": [],
        "correctness": [],
    }
    for item in items:
        gen = gens.get(store.key(variant, item["id"], version))
        if gen is None or "error" in gen:
            problems.append(
                f"{item['id']}: no generation for variant {variant!r} at version {version}"
            )
            continue

        wants_refusal = item["expected_outcome"] == "abstain"
        if gen["refused"] != wants_refusal:
            expected = "a refusal" if wants_refusal else "an answer"
            failures.append(
                f"{item['id']}: expected {expected} but the pipeline {'refused' if gen['refused'] else 'answered'}"
            )
        if gen["refused"]:
            continue

        for metric, values in scores.items():
            if metric == "correctness" and item["expected_outcome"] != "answer":
                continue
            record = judged.get(f"{gen['key']}|{metric}")
            if record is None or record["score"] is None:
                problems.append(f"{item['id']}: {metric} not judged")
            else:
                values.append(record["score"])

    means = {m: sum(v) / len(v) for m, v in scores.items() if v}
    for metric, floor in (("faithfulness", faith_floor), ("relevancy", rel_floor)):
        if metric in means and means[metric] < floor:
            failures.append(
                f"{metric} mean {means[metric]:.3f} is below the floor {floor}"
            )
    if not problems and "faithfulness" not in means:
        failures.append("no answered item was scored")
    return problems, failures, means

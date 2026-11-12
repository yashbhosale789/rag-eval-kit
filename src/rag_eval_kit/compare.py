"""Paired comparison of two runs on the same questions: sign test + bootstrap CI.

A run is `ret:<retrieval file>` or `gen:<variant>@<version>` (generation + judgments).
"""

import json
import random
from math import comb
from pathlib import Path

from rag_eval_kit import store


def load_ret(path: Path) -> dict[str, dict[str, float]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r["metrics"] for r in rows if "metrics" in r}


def load_gen(
    gen_path: Path, judge_path: Path, variant: str, version: str, items: dict
) -> dict[str, dict[str, float]]:
    gens, judged = store.load(gen_path), store.load(judge_path)
    rows: dict[str, dict[str, float]] = {}
    for qid, item in items.items():
        gen = gens.get(store.key(variant, qid, version))
        if gen is None or "error" in gen:
            continue
        refusal_expected = item["expected_outcome"] == "abstain"
        row = {
            "behavior": float(gen["refused"] == refusal_expected),
            "latency_ms": float(gen["latency_ms"]),
        }
        for metric in ("faithfulness", "correctness", "relevancy"):
            record = judged.get(f"{gen['key']}|{metric}")
            if record is not None and record["score"] is not None:
                row[metric] = record["score"]
        rows[qid] = row
    return rows


def load(spec: str, items: dict, runs: Path) -> dict[str, dict[str, float]]:
    kind, _, rest = spec.partition(":")
    if kind == "ret" and rest:
        return load_ret(Path(rest))
    variant, _, version = rest.partition("@")
    if kind == "gen" and variant and version:
        return load_gen(runs / "generations.jsonl", runs / "judgments.jsonl", variant, version, items)
    raise SystemExit(f"bad run {spec!r}: use ret:<file> or gen:<variant>@<version>")


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n if n else 1.0
    return min(1.0, 2 * tail)


def bootstrap_ci(
    diffs: list[float], alpha: float, reps: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)  # nosec B311
    means = sorted(
        sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps)
    )
    return means[int(alpha / 2 * reps)], means[int((1 - alpha / 2) * reps) - 1]


def verdict(
    diffs: list[float],
    p: float,
    low: float,
    high: float,
    alpha: float,
    min_effect: float,
) -> str:
    mean = sum(diffs) / len(diffs)
    if p >= alpha or low <= 0 <= high:
        return "no evidence of a difference at this sample size"
    if abs(mean) < min_effect:
        return f"significant but smaller than the minimum useful effect ({min_effect})"
    return "A higher" if mean > 0 else "B higher"


def compare(
    a: dict,
    b: dict,
    items: dict,
    metric: str,
    metric_b: str | None = None,
    split: str = "all",
    family: int = 1,
    min_effect: float = 0.0,
) -> dict:
    mb = metric_b or metric
    order = {qid: n for n, qid in enumerate(items)}  # dataset order, whatever the id shape
    in_split = [q for q in items if split in ("all", items[q].get("split"))]
    ids = [q for q in in_split if q in a and q in b and metric in a[q] and mb in b[q]]
    ids.sort(key=order.__getitem__)
    if len(ids) < 2:
        raise SystemExit(
            f"only {len(ids)} paired questions with metric {metric!r}; nothing to compare"
        )
    diffs = [a[q][metric] - b[q][mb] for q in ids]
    alpha = 0.05 / family
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    p = sign_test(wins, losses)
    low, high = bootstrap_ci(diffs, alpha)
    return {
        "ids": ids,
        "diffs": diffs,
        "pairs": len(ids),
        "mean_a": sum(a[q][metric] for q in ids) / len(ids),
        "mean_b": sum(b[q][mb] for q in ids) / len(ids),
        "mean_diff": sum(diffs) / len(diffs),
        "alpha": alpha,
        "ci": (low, high),
        "wins": wins,
        "losses": losses,
        "p": p,
        "verdict": verdict(diffs, p, low, high, alpha, min_effect),
    }

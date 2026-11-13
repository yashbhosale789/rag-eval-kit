"""Run adapter.answer() over a dataset and store one record per (variant, question, version).

Records already stored without an error are skipped, so an interrupted run resumes.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_kit import store

REFUSAL_PHRASE = "don't have enough information"


def is_refusal(out: dict, phrase: str = REFUSAL_PHRASE) -> bool:
    answer = " ".join(out["answer"].lower().replace("’", "'").split())
    return (
        out.get("outcome") == "abstain"
        or not out.get("contexts")
        or phrase in answer
    )


def pending(adapter, items: list[dict], out_path: Path, variant: str) -> list[dict]:
    done = {k for k, r in store.load(out_path).items() if "error" not in r}
    return [i for i in items if store.key(variant, i["id"], adapter.version) not in done]


def run(
    adapter,
    items: list[dict],
    out_path: Path,
    variant: str = "baseline",
    limit: int | None = None,
    phrase: str = REFUSAL_PHRASE,
    dataset_id: str = "",
    gap: float = 0.0,
) -> bool:
    """Returns True when some question failed (the run is incomplete)."""
    todo = pending(adapter, items, out_path, variant)
    todo = todo[:limit] if limit else todo
    corpus = getattr(adapter, "corpus_id", "")
    incomplete = False
    for n, item in enumerate(todo, 1):
        base = {
            "key": store.key(variant, item["id"], adapter.version),
            "variant": variant,
            "qid": item["id"],
            "version": adapter.version,
            "dataset": dataset_id,
            "corpus": corpus,
        }
        started = time.monotonic()
        try:
            out = adapter.answer(item["question"])
            if not isinstance(out, dict) or "answer" not in out:
                raise TypeError("answer() must return a dict with an 'answer' key")
        except Exception as e:  # one failing question must not lose the run
            incomplete = True
            store.append(out_path, {**base, "error": f"{type(e).__name__}: {e}"})
            print(f"[{n}/{len(todo)}] {item['id']} ERROR {str(e)[:100]}")
            continue
        latency_ms = int((time.monotonic() - started) * 1000)
        contexts = out.get("contexts", [])
        record = {
            **base,
            "question": item["question"],
            "answer": out["answer"],
            "contexts": contexts,
            "context_hash": store.context_hash(contexts),
            "outcome": out.get("outcome"),
            "refused": is_refusal(out, phrase),
            "latency_ms": latency_ms,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        store.append(out_path, record)
        print(f"[{n}/{len(todo)}] {item['id']} refused={record['refused']} {latency_ms}ms")
        time.sleep(max(0.0, gap - (time.monotonic() - started)))
    return incomplete


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(items: list[dict], out_path: Path, variant: str, version: str) -> None:
    records = store.load(out_path)
    rows = [(i, records.get(store.key(variant, i["id"], version))) for i in items]
    rows = [(i, r) for i, r in rows if r and "error" not in r]
    if not rows:
        return
    latencies = sorted(r["latency_ms"] for _, r in rows)
    print(f"\n{variant} @ {version}: n={len(rows)}")
    print(f"  abstention correct : {rate([r['refused'] == (i['expected_outcome'] == 'abstain') for i, r in rows]):.2f}")
    print(f"  false-abstain rate : {rate([r['refused'] for i, r in rows if i['expected_outcome'] == 'answer']):.2f}")
    print(f"  false-answer rate  : {rate([not r['refused'] for i, r in rows if i['expected_outcome'] == 'abstain']):.2f}")
    print(f"  latency ms         : p50={latencies[len(latencies) // 2]} max={latencies[-1]}")

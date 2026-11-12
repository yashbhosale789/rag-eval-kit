"""Retrieval metrics over the adapter's final ranked list.

A chunk is a hit for a relevant span when it comes from the same document (basename of
`source`) and its normalised text contains the normalised evidence phrase.
"""

from rag_eval_kit.dataset import basename, norm

KS = (1, 3, 5)


def chunk_source(chunk: dict) -> str:
    return basename(chunk["source"])


def match_matrix(item: dict, ranked: list[dict]) -> list[list[bool]]:
    cols = [(chunk_source(c), norm(c["text"])) for c in ranked]
    return [
        [src == rel["source"] and norm(rel["evidence"]) in text for src, text in cols]
        for rel in item["relevant"]
    ]


def score(item: dict, ranked: list[dict], ks: tuple[int, ...]) -> dict[str, float]:
    matrix = match_matrix(item, ranked)
    any_of = item.get("any_of", False)
    out: dict[str, float] = {}

    for k in ks:
        covered = [any(row[:k]) for row in matrix]
        out[f"recall@{k}"] = (
            float(any(covered)) if any_of else sum(covered) / len(covered)
        )
        out[f"success@{k}"] = float(any(covered) if any_of else all(covered))

    first = [row.index(True) + 1 for row in matrix if True in row]
    out["mrr"] = 1 / min(first) if first else 0.0
    return out


def check_chunks(qid: str, ranked: object) -> None:
    if not isinstance(ranked, list):
        raise ValueError(f"{qid}: retrieve() must return a list, got {type(ranked).__name__}")
    for n, chunk in enumerate(ranked):
        missing = [k for k in ("text", "source") if not isinstance(chunk, dict) or k not in chunk]
        if missing:
            raise ValueError(f"{qid}: chunk {n} from retrieve() lacks {', '.join(missing)}")


def run(adapter, items: list[dict], ks: tuple[int, ...] = KS, dataset_id: str = "") -> list[dict]:
    rows = []
    for item in items:
        if item["expected_outcome"] != "answer":
            continue
        row = {
            "id": item["id"],
            "category": item.get("category"),
            "split": item.get("split"),
            "question": item["question"],
            "dataset": dataset_id,
            "corpus": getattr(adapter, "corpus_id", ""),
            "version": getattr(adapter, "version", ""),
        }
        try:
            ranked = adapter.retrieve(item["question"])
        except Exception as e:  # one failing question must not lose the whole run
            rows.append({**row, "error": f"{type(e).__name__}: {e}"})
            continue
        check_chunks(item["id"], ranked)  # a contract error is the user's bug: stop loudly
        rows.append({**row, "metrics": score(item, ranked, ks)})
    return rows


def means(rows: list[dict]) -> dict[str, float]:
    ok = [r["metrics"] for r in rows if "metrics" in r]
    return {m: sum(r[m] for r in ok) / len(ok) for m in (ok[0] if ok else {})}

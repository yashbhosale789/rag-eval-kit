"""Dataset schema and validator.

An item names its relevant passages by document (`source`) and a short `evidence` phrase
copied from it, never by chunk id or page, so systems that chunk differently are comparable.
"""

import json
import re
import unicodedata
from pathlib import Path

CATEGORIES = {
    "normal",
    "difficult",
    "multi_hop",
    "ambiguous",
    "unanswerable",
    "adversarial",
    "image",
    "table",
}
DIFFICULTIES = {"easy", "medium", "hard"}
OUTCOMES = {"answer", "abstain"}
ORIGINS = {"manual", "synthetic", "upstream"}
SPLITS = {"dev", "test"}
ENUMS = {
    "category": CATEGORIES,
    "difficulty": DIFFICULTIES,
    "expected_outcome": OUTCOMES,
    "origin": ORIGINS,
    "split": SPLITS,
}
REQUIRED = set(ENUMS) | {"id", "question", "ground_truth", "relevant", "ci"}

MIN_SPAN, MAX_SPAN = 25, 300
MAX_MATCHES = 2


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", text)


def basename(path: str) -> str:
    return re.split(r"[\\/]", path)[-1]


def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_chunks(path: Path) -> dict[str, list[str]]:
    """A JSONL of {"source", "text"}: the chunks the evidence spans are checked against."""
    by_source: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunk = json.loads(line)
            by_source.setdefault(basename(chunk["source"]), []).append(norm(chunk["text"]))
    return by_source


def check_span(rel: dict, chunks: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    source, evidence = rel.get("source"), rel.get("evidence")
    if source not in chunks:
        return [f"unknown source {source!r}; known: {sorted(chunks)}"], []
    span = norm(evidence) if isinstance(evidence, str) else ""
    if not span:
        return ["empty evidence"], []
    matches = sum(span in chunk for chunk in chunks[source])
    if matches == 0:
        return [f"evidence matches no chunk of {source}: {evidence[:50]!r}"], []
    warnings = []
    if matches > MAX_MATCHES:
        warnings.append(f"evidence matches {matches} chunks in {source} (too generic?)")
    length = len(" ".join(evidence.split()))
    if not MIN_SPAN <= length <= MAX_SPAN:
        warnings.append(f"evidence length {length} outside {MIN_SPAN}-{MAX_SPAN}")
    return [], warnings


def check_item(
    item: dict, chunks: dict[str, list[str]] | None, seen: set[str]
) -> tuple[list[str], list[str]]:
    qid = item.get("id", "?")
    errors: list[str] = []
    warnings: list[str] = []

    missing = REQUIRED - item.keys()
    if missing:
        return [f"{qid}: missing fields {sorted(missing)}"], warnings

    if qid in seen:
        errors.append("duplicate id")
    seen.add(qid)

    for field in ("question", "ground_truth"):
        if not str(item[field]).strip():
            errors.append(f"empty {field}")
    for field, allowed in ENUMS.items():
        if item[field] not in allowed:
            errors.append(f"{field}={item[field]!r} not in {sorted(allowed)}")

    relevant, outcome, category = (
        item["relevant"],
        item["expected_outcome"],
        item["category"],
    )
    if outcome == "abstain" and relevant:
        errors.append("abstain items must have relevant=[]")
    if outcome == "answer" and not relevant:
        errors.append("answer items need at least one relevant span")
    if category == "unanswerable" and outcome != "abstain":
        errors.append("unanswerable items must expect abstain")
    if category == "multi_hop" and len(relevant) < 2:
        errors.append("multi_hop items need at least 2 spans")

    if chunks is not None:
        for rel in relevant:
            span_errors, span_warnings = check_span(rel, chunks)
            errors += span_errors
            warnings += span_warnings
    return [f"{qid}: {m}" for m in errors], [f"{qid}: {m}" for m in warnings]


def check(
    items: list[dict], chunks: dict[str, list[str]] | None
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for item in items:
        e, w = check_item(item, chunks, seen)
        errors += e
        warnings += w
    return errors, warnings

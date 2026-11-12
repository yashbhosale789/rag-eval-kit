"""Result files: JSONL records keyed by variant|question|version, and content ids."""

import hashlib
import json
from pathlib import Path


def content_id(path: Path) -> str:
    # parsed, not raw bytes: CRLF checkouts on Windows must give the same id as CI
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


def context_hash(contexts: list[str]) -> str:
    return hashlib.sha256("\x00".join(contexts).encode()).hexdigest()[:10]


def key(variant: str, qid: str, version: str) -> str:
    return f"{variant}|{qid}|{version}"


def load(path: Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a crash; that key simply gets redone
        records[record["key"]] = record
    return records


def append(path: Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

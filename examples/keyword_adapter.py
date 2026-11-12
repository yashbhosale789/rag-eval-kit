"""A toy adapter: paragraphs ranked by shared words. No model, no key, no network.

Run from the repository root:
    rag-eval-kit retrieval --adapter examples.keyword_adapter:KeywordAdapter --dataset examples/questions.json
"""

import re
from pathlib import Path

DOCS = Path(__file__).parent / "docs"


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class KeywordAdapter:
    version = "keyword-v1"
    corpus_id = "examples-docs"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.chunks = [
            {"text": para.strip(), "source": path.name}
            for path in sorted(DOCS.glob("*.txt"))
            for para in path.read_text(encoding="utf-8").split("\n\n")
            if para.strip()
        ]

    def retrieve(self, query: str) -> list[dict]:
        q = words(query)
        ranked = sorted(self.chunks, key=lambda c: len(q & words(c["text"])), reverse=True)
        return ranked[: self.top_k]

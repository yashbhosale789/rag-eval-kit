# rag-eval-kit

Measure any RAG system with the same method: retrieval metrics, LLM-judged answers, and a paired
comparison that tells you whether a change really helped. You plug your system in through a small
adapter; the kit never needs to know how it works inside.

## Install

```bash
pip install git+https://github.com/yashbhosale789/rag-eval-kit
pip install "rag-eval-kit[judge] @ git+https://github.com/yashbhosale789/rag-eval-kit"   # adds the LLM judge
```

The core has no dependencies. Python 3.11+.

## Try it (no LLM, no key)

From a clone of this repository:

```bash
rag-eval-kit validate examples/questions.json
rag-eval-kit retrieval --adapter examples.keyword_adapter:KeywordAdapter --dataset examples/questions.json
```

## 1. Write an adapter

Any object with these members, named on the command line as `module:attr` (run the command from the
folder that contains your module; a class is instantiated with no arguments):

```python
class Adapter:
    version = "bm25-k5"      # what you changed; part of every result key
    corpus_id = "docs-2026"  # optional; recorded with every result

    def retrieve(self, query: str) -> list[dict]:
        # your ranked chunks, best first
        return [{"text": "...", "source": "manual.pdf", "page": 3}]   # page is optional

    def answer(self, query: str) -> dict:   # only needed for generate/judge
        return {"answer": "...", "contexts": ["chunk text used", "..."]}
```

`source` may be a path; only its file name is compared. Do your own query rewriting and strip any
citation markers from `answer` before returning it. An answer counts as a refusal when it contains
the refusal phrase (`--refusal-phrase`, default `don't have enough information`), when `contexts` is
empty, or when you return `"outcome": "abstain"`.

## 2. Write a dataset

A JSON list. Relevant passages are named by document and a short phrase copied from it, never by chunk
id or page, so any chunking can be scored against the same questions:

```json
{
  "id": "Q1",
  "question": "How often should I lubricate my bicycle chain?",
  "ground_truth": "Every 300 kilometres, or after any ride in the rain.",
  "expected_outcome": "answer",
  "relevant": [{"source": "bicycle-care.txt", "evidence": "Clean and lubricate the chain every 300 kilometres"}],
  "category": "normal", "difficulty": "easy", "origin": "manual", "split": "dev", "ci": true
}
```

- `expected_outcome`: `answer` or `abstain` (questions the documents cannot answer; `relevant: []`).
- `category`: `normal`, `difficult`, `multi_hop` (2+ spans), `ambiguous`, `unanswerable`, `adversarial`, `image`, `table`.
- `split`: `dev` while you tune, `test` for the final number. `ci`: items the gate checks with `--ci-only`.
- Optional `"any_of": true`: any one of the spans is enough.

`rag-eval-kit validate questions.json --chunks chunks.jsonl` also checks that every evidence phrase
appears in your chunks (`chunks.jsonl`: one `{"source", "text"}` per line).

## 3. Commands

| Command | What it does | LLM calls |
|---|---|---|
| `validate DATASET` | schema check; prints the dataset id | none |
| `retrieval --adapter A --dataset D` | recall@1/3/5, success@1/3/5, MRR per question and overall | none from the kit |
| `generate --adapter A --dataset D` | stores `answer()` per question; resumes after a crash | yours, `--yes` required |
| `judge --dataset D --version V` | faithfulness, correctness, relevancy for stored answers | judge, `--yes` required |
| `judge --dataset D --calibrate` | known-good vs known-bad answers, to check the judge | judge, `--yes` required |
| `compare A B METRIC --dataset D` | paired difference, bootstrap CI, sign test, verdict | none |
| `gate --dataset D --version V` | exit 0 pass, 1 regression, 2 incomplete | none |

Both LLM commands print how many calls they will make and stop unless you add `--yes`. Use `--limit N`
to spread a run over several days on a free tier; rerunning picks up where it stopped.

Every record carries `dataset` (a hash of the dataset file), `corpus` and `version`, so you can always
tell which numbers may be compared.

### Judge settings

Any OpenAI-compatible endpoint:

```bash
export JUDGE_BASE_URL=https://api.groq.com/openai/v1   # or OpenRouter, Gemini's OpenAI endpoint, http://localhost:11434/v1
export JUDGE_MODEL=openai/gpt-oss-20b
export JUDGE_API_KEY=...                               # leave unset for a local Ollama
```

Relevancy also embeds questions locally (`--embed-model`, default `BAAI/bge-small-en-v1.5`).
Run `judge --calibrate` once per judge model: a useful judge scores good answers well above bad
ones (AUC near 1). For reference, `gemini-3.5-flash-lite` on Gemini's OpenAI endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`) scored AUC 1.0 on all three metrics,
with 0 unparsable replies out of 60.

## Reading `compare`

```bash
rag-eval-kit compare ret:runs/before.json ret:runs/after.json mrr --dataset questions.json --split dev
rag-eval-kit compare gen:baseline@v1 gen:baseline@v2 faithfulness --dataset questions.json
```

Each question is a pair. The verdict says `A higher`, `B higher`, or `no evidence of a difference at
this sample size`: the last one means the change may still matter, but this many questions cannot show
it. When you compare several metrics at once, pass `--family N` (Bonferroni) so one lucky metric does
not count as a win. `--min-effect` sets the smallest difference worth acting on.

## Licence

MIT.

"""LLM-judged answer quality: faithfulness, correctness, relevancy.

The judge is any OpenAI-compatible chat endpoint (Groq, OpenRouter, Gemini, Ollama, ...):
    JUDGE_BASE_URL   e.g. https://api.groq.com/openai/v1
    JUDGE_MODEL      e.g. openai/gpt-oss-20b
    JUDGE_API_KEY    empty for a local Ollama
Needs the extra: pip install "rag-eval-kit[judge]".
"""

import json
import os
import re
import time
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from rag_eval_kit import store

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_state = {"last": 0.0, "gap": 4.5, "embed_model": EMBED_MODEL}

FAITHFULNESS = """You are evaluating whether an answer is faithful to the given context.

Context:
{context}

Answer: {answer}

Instructions:
1. Break the answer into individual atomic factual claims.
2. For each claim, judge whether it is supported by the context.
3. Return ONLY a JSON object: {{"total": <int>, "supported": <int>}}"""

CORRECTNESS = """You are grading an answer against a reference answer.

Question: {question}
Reference answer: {truth}
Answer to grade: {answer}

Instructions:
1. Split the reference answer into its individual key facts.
2. Count how many key facts the answer to grade states or clearly implies.
3. Say whether the answer to grade contradicts any key fact of the reference.
Return ONLY a JSON object: {{"total": <int>, "covered": <int>, "contradicts": <true or false>}}"""

RELEVANCY = """Given the following answer, write 3 different questions that this answer could be responding to.
Return ONLY a JSON object: {{"questions": ["...", "...", "..."]}}

Answer: {answer}"""


def parse_json_reply(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("no JSON object in reply", raw, 0)
    return json.loads(match.group(0))


def settings() -> tuple[str, str, str]:
    base, model = os.getenv("JUDGE_BASE_URL", ""), os.getenv("JUDGE_MODEL", "")
    if not base or not model:
        raise SystemExit(
            "set JUDGE_BASE_URL and JUDGE_MODEL (and JUDGE_API_KEY unless the endpoint is local)"
        )
    return base.rstrip("/"), model, os.getenv("JUDGE_API_KEY", "")


def _requests():
    try:
        import requests
    except ImportError:
        raise SystemExit('pip install "rag-eval-kit[judge]"') from None
    return requests


def judge_call(prompt: str) -> str:
    requests = _requests()
    base, model, key = settings()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    for attempt in range(4):
        time.sleep(max(0.0, _state["gap"] - (time.monotonic() - _state["last"])))
        try:
            response = requests.post(
                f"{base}/chat/completions", headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            try:
                return (
                    response.json()["choices"][0]["message"]["content"] or ""
                ).strip()
            except (KeyError, IndexError, TypeError):
                return ""
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            status = getattr(e.response, "status_code", None)
            if attempt == 3 or (
                status is not None and status not in {429, 500, 502, 503, 504}
            ):
                raise
            retry_after = (
                e.response.headers.get("retry-after")
                if e.response is not None
                else None
            )
            time.sleep(min(float(retry_after or 5 * 3**attempt), 120))
        finally:
            _state["last"] = time.monotonic()


@cache
def embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit('pip install "rag-eval-kit[judge]"') from None
    return SentenceTransformer(_state["embed_model"])


def faithfulness(question, truth, answer, contexts):
    block = "\n\n".join(f"[C{i}] {c}" for i, c in enumerate(contexts, 1))
    data = parse_json_reply(
        judge_call(FAITHFULNESS.format(context=block, answer=answer))
    )
    total, supported = int(data["total"]), int(data["supported"])
    if total < 1 or not 0 <= supported <= total:
        raise ValueError(f"bad counts: {data}")
    return supported / total, {"total": total, "supported": supported}


def correctness(question, truth, answer, contexts):
    data = parse_json_reply(
        judge_call(CORRECTNESS.format(question=question, truth=truth, answer=answer))
    )
    total, covered = int(data["total"]), int(data["covered"])
    if (
        total < 1
        or not 0 <= covered <= total
        or not isinstance(data["contradicts"], bool)
    ):
        raise ValueError(f"bad verdict: {data}")
    return covered / total, {
        "total": total,
        "covered": covered,
        "contradicts": data["contradicts"],
    }


def relevancy(question, truth, answer, contexts):
    data = parse_json_reply(judge_call(RELEVANCY.format(answer=answer)))
    questions = data["questions"]
    if (
        not isinstance(questions, list)
        or not questions
        or not all(isinstance(q, str) and q.strip() for q in questions)
    ):
        raise ValueError(f"bad questions: {data}")
    vectors = embedder().encode([question] + questions, normalize_embeddings=True)
    return float((vectors[1:] @ vectors[0]).mean()), {"questions": questions}


METRICS = {
    "faithfulness": faithfulness,
    "correctness": correctness,
    "relevancy": relevancy,
}


def applicable(item: dict) -> list[str]:
    return [
        m for m in METRICS if m != "correctness" or item["expected_outcome"] == "answer"
    ]


def run_metric(
    metric: str, item: dict, answer: str, contexts: list[str]
) -> tuple[float | None, dict]:
    for _ in range(2):
        try:
            score, detail = METRICS[metric](
                item["question"], item["ground_truth"], answer, contexts
            )
            return round(score, 4), detail
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None, {"error": "unparsable"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def auc(good: list[float], bad: list[float]) -> float:
    pairs = [(g, b) for g in good for b in bad]
    if not pairs:
        return float("nan")
    return sum(1.0 if g > b else 0.5 if g == b else 0.0 for g, b in pairs) / len(pairs)


def pending(
    items: list[dict], gens_path: Path, judge_path: Path, variants: list[str], version: str
) -> list[tuple[dict, str]]:
    gens = store.load(gens_path)
    done = {k for k, r in store.load(judge_path).items() if r["score"] is not None}
    todo = []
    for item in items:  # dataset order, whatever the id shape
        for variant in variants:
            gen = gens.get(store.key(variant, item["id"], version))
            if gen is None or "error" in gen or gen["refused"]:
                continue
            todo += [(gen, m) for m in applicable(item) if f"{gen['key']}|{m}" not in done]
    return todo


def judge_generations(
    items: list[dict],
    gens_path: Path,
    judge_path: Path,
    variants: list[str],
    version: str,
    limit: int | None = None,
) -> None:
    by_id = {i["id"]: i for i in items}
    todo = pending(items, gens_path, judge_path, variants, version)
    todo = todo[:limit] if limit else todo
    model = os.getenv("JUDGE_MODEL", "")
    for n, (gen, metric) in enumerate(todo, 1):
        try:
            score, detail = run_metric(metric, by_id[gen["qid"]], gen["answer"], gen["contexts"])
        except Exception as e:  # network errors from the judge endpoint
            if getattr(getattr(e, "response", None), "status_code", None) == 429:
                print("429 persisted after retries; quota likely exhausted. Rerun later to resume.")
                break
            print(f"[{n}/{len(todo)}] {gen['qid']} {metric} ERROR {str(e)[:100]}")
            continue
        store.append(judge_path, {
            "key": f"{gen['key']}|{metric}",
            "gen_key": gen["key"],
            "variant": gen["variant"],
            "qid": gen["qid"],
            "version": gen["version"],
            "dataset": gen.get("dataset"),
            "corpus": gen.get("corpus"),
            "metric": metric,
            "score": score,
            "detail": detail,
            "judge_model": model,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        print(f"[{n}/{len(todo)}] {gen['variant']} {gen['qid']} {metric} = {score}")


def summarize(judge_path: Path, variants: list[str], version: str) -> None:
    rows = [r for r in store.load(judge_path).values()
            if r["version"] == version and r["variant"] in variants]
    models = {r["judge_model"] for r in rows}
    if len(models) > 1:
        print("WARNING: mixed judge models:", sorted(models))
    for variant in variants:
        print(f"\n{variant} @ {version}:")
        for metric in METRICS:
            mine = [r for r in rows if r["variant"] == variant and r["metric"] == metric]
            scored = [r["score"] for r in mine if r["score"] is not None]
            extra = ""
            if metric == "correctness":
                extra = f"  contradictions={sum(bool(r['detail'].get('contradicts')) for r in mine if r['score'] is not None)}"
            print(f"  {metric:<13} n={len(scored)} failed={len(mine) - len(scored)} mean={mean(scored):.3f}{extra}")


def calibrate(items: list[dict], out_path: Path) -> dict:
    """Score known-good (the reference) vs known-bad (another document's reference) answers."""
    pool = [i for i in items if i["split"] == "dev" and i["expected_outcome"] == "answer" and i["relevant"]]
    picks = pool[:: max(1, len(pool) // 10)][:10]
    scores = {m: {"good": [], "bad": []} for m in METRICS}
    failed = {m: 0 for m in METRICS}
    for k, item in enumerate(picks):
        others = [p for p in pool if p["relevant"][0]["source"] != item["relevant"][0]["source"]]
        if not others:
            raise SystemExit("calibration needs dev questions from at least two documents")
        wrong = others[k % len(others)]["ground_truth"]
        contexts = [r["evidence"] for r in item["relevant"]]
        for label, answer in (("good", item["ground_truth"]), ("bad", wrong)):
            for metric in METRICS:
                score, _ = run_metric(metric, item, answer, contexts)
                if score is None:
                    failed[metric] += 1
                else:
                    scores[metric][label].append(score)
        print(f"calibrated {item['id']}")
    report = {
        metric: {
            "mean_good": round(mean(s["good"]), 3),
            "mean_bad": round(mean(s["bad"]), 3),
            "auc": round(auc(s["good"], s["bad"]), 3),
            "failed": failed[metric],
        }
        for metric, s in scores.items()
    }
    for metric, r in report.items():
        print(f"  {metric:<13} {r}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({"judge_model": os.getenv("JUDGE_MODEL", ""), "report": report, "scores": scores}, indent=2),
        encoding="utf-8",
    )
    return report

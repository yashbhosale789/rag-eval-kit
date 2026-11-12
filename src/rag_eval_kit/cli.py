import argparse
import json
import sys
from pathlib import Path

from rag_eval_kit import compare, dataset, gate, generate, retrieval, store
from rag_eval_kit.adapter import load_adapter


def load_dataset(path: str) -> list[dict]:
    """The dataset is user input: refuse to run on one that fails the schema check."""
    items = dataset.load(path)
    errors, _ = dataset.check(items, None)
    if errors:
        for e in errors:
            print("error:", e)
        raise SystemExit(f"{path}: {len(errors)} dataset errors; run `rag-eval-kit validate` for details")
    return items


def cmd_validate(args: argparse.Namespace) -> int:
    items = dataset.load(args.dataset)
    chunks = dataset.load_chunks(args.chunks) if args.chunks else None
    errors, warnings = dataset.check(items, chunks)
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    print(
        f"{len(items)} items, {len(errors)} errors, {len(warnings)} warnings, "
        f"dataset {store.content_id(args.dataset)}"
    )
    return 1 if errors else 0


def cmd_retrieval(args: argparse.Namespace) -> int:
    items = [
        i for i in load_dataset(args.dataset) if args.split in ("all", i["split"])
    ]
    adapter = load_adapter(args.adapter)
    rows = retrieval.run(adapter, items, retrieval.KS, store.content_id(args.dataset))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    errors = [r for r in rows if "error" in r]
    for r in errors:
        print(f"error: {r['id']}: {r['error']}")
    summary = "  ".join(f"{m} {v:.3f}" for m, v in retrieval.means(rows).items())
    print(f"{len(rows) - len(errors)} scored, {len(errors)} errors  {summary}")
    print(f"dataset {rows[0]['dataset'] if rows else '-'}  corpus {getattr(adapter, 'corpus_id', '') or '-'}  "
          f"version {getattr(adapter, 'version', '') or '-'}  -> {out}")
    return 1 if errors else 0


def cmd_compare(args: argparse.Namespace) -> int:
    items = {i["id"]: i for i in load_dataset(args.dataset)}
    a = compare.load(args.a, items, Path(args.runs))
    b = compare.load(args.b, items, Path(args.runs))
    r = compare.compare(
        a, b, items, args.metric, args.metric_b, args.split, args.family, args.min_effect
    )
    mb = args.metric_b or args.metric
    low, high = r["ci"]
    print(f"A={args.a}  B={args.b}  metric={args.metric}{'/' + mb if mb != args.metric else ''}  split={args.split}")
    print(f"  pairs={r['pairs']}  mean A={r['mean_a']:.3f}  B={r['mean_b']:.3f}")
    print(f"  diff A-B = {r['mean_diff']:+.3f}   {100 * (1 - r['alpha']):.1f}% CI [{low:+.3f}, {high:+.3f}]")
    print(f"  A higher on {r['wins']}, B higher on {r['losses']}, tied {r['pairs'] - r['wins'] - r['losses']}; "
          f"sign test p={r['p']:.4f} (alpha {r['alpha']:.4f})")
    print(f"  verdict: {r['verdict']}")
    for q, d in sorted(zip(r["ids"], r["diffs"]), key=lambda t: -abs(t[1]))[:20]:
        if d:
            print(f"    {q}: A={a[q][args.metric]:.2f}  B={b[q][mb]:.2f}")
    return 0


def select(items: list[dict], args: argparse.Namespace) -> list[dict]:
    if getattr(args, "ci_only", False):
        return [i for i in items if i["ci"]]
    return [i for i in items if args.split in ("all", i["split"])]


def cmd_generate(args: argparse.Namespace) -> int:
    items = select(load_dataset(args.dataset), args)
    adapter = load_adapter(args.adapter)
    out = Path(args.runs) / "generations.jsonl"
    todo = generate.pending(adapter, items, out, args.variant)
    n = min(len(todo), args.limit) if args.limit else len(todo)
    print(f"version {adapter.version}: {n} adapter.answer calls will be made ({len(items) - len(todo)} already stored)")
    if n and not args.yes:
        print("add --yes to run them (each call may spend LLM quota)")
        return 2
    incomplete = generate.run(
        adapter, items, out, args.variant, args.limit, args.refusal_phrase,
        store.content_id(args.dataset), args.gap,
    )
    generate.summarize(items, out, args.variant, adapter.version)
    return 1 if incomplete else 0


def cmd_gate(args: argparse.Namespace) -> int:
    items = select(load_dataset(args.dataset), args)
    runs = Path(args.runs)
    problems, failures, means = gate.evaluate(
        items, store.load(runs / "generations.jsonl"), store.load(runs / "judgments.jsonl"),
        args.variant, args.version, args.faith_floor, args.rel_floor,
    )
    print(f"gate: variant {args.variant!r}, version {args.version}, {len(items)} items")
    floors = {"faithfulness": f"floor {args.faith_floor}", "relevancy": f"floor {args.rel_floor}",
              "correctness": "report only"}
    for metric, value in means.items():
        print(f"  {metric:<13} mean={value:.3f} ({floors[metric]})")
    for message in failures:
        print("REGRESSION:", message)
    for message in problems:
        print("INCOMPLETE:", message)
    print("FAIL" if failures else "INCOMPLETE" if problems else "PASS")
    return 1 if failures else 2 if problems else 0


def cmd_judge(args: argparse.Namespace) -> int:
    from rag_eval_kit import judge

    judge._state.update(gap=args.gap, embed_model=args.embed_model)
    items = load_dataset(args.dataset)
    runs = Path(args.runs)
    if args.calibrate:
        pool = [i for i in items if i["split"] == "dev" and i["expected_outcome"] == "answer" and i["relevant"]]
        n = len(pool[:: max(1, len(pool) // 10)][:10]) * 2 * 3  # up to 10 items x good/bad x 3 metrics
    else:
        if not args.version:
            raise SystemExit("--version is required (the adapter version the generations were made with)")
        todo = judge.pending(items, runs / "generations.jsonl", runs / "judgments.jsonl", args.variants, args.version)
        n = min(len(todo), args.limit) if args.limit else len(todo)
    base, model, _ = judge.settings()
    print(f"{n} judge calls to {model} at {base} (relevancy also embeds locally with {args.embed_model})")
    if n and not args.yes:
        print("add --yes to run them (each call may spend LLM quota)")
        return 2
    if args.calibrate:
        judge.calibrate(items, runs / "judge_calibration.json")
        return 0
    judge.judge_generations(
        items, runs / "generations.jsonl", runs / "judgments.jsonl", args.variants, args.version, args.limit
    )
    judge.summarize(runs / "judgments.jsonl", args.variants, args.version)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-eval-kit")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check a dataset file")
    v.add_argument("dataset")
    v.add_argument("--chunks", help='JSONL of {"source", "text"} to check evidence spans against')
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("retrieval", help="score the adapter's retrieve() against a dataset")
    r.add_argument("--adapter", required=True, help="module:attr, e.g. my_rag:Adapter")
    r.add_argument("--dataset", required=True)
    r.add_argument("--split", choices=["dev", "test", "all"], default="all")
    r.add_argument("--out", default="runs/retrieval.json")
    r.set_defaults(func=cmd_retrieval)

    c = sub.add_parser("compare", help="paired comparison of two runs on the same questions")
    c.add_argument("a", help="ret:<file> or gen:<variant>@<version>")
    c.add_argument("b")
    c.add_argument("metric")
    c.add_argument("--metric-b", help="B's metric if it differs (e.g. recall@5 vs recall@3)")
    c.add_argument("--dataset", required=True)
    c.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    c.add_argument("--family", type=int, default=1, help="comparisons in this family (Bonferroni)")
    c.add_argument("--min-effect", type=float, default=0.0, help="smallest difference worth acting on")
    c.add_argument("--runs", default="runs", help="folder with generations.jsonl / judgments.jsonl")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("generate", help="run the adapter's answer() and store the results")
    g.add_argument("--adapter", required=True)
    g.add_argument("--dataset", required=True)
    g.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    g.add_argument("--ci-only", action="store_true", help="only items with ci=true")
    g.add_argument("--limit", type=int, help="at most N new calls this run")
    g.add_argument("--variant", default="baseline", help="a name for what you are testing")
    g.add_argument("--refusal-phrase", default=generate.REFUSAL_PHRASE)
    g.add_argument("--gap", type=float, default=0.0, help="minimum seconds between calls")
    g.add_argument("--runs", default="runs")
    g.add_argument("--yes", action="store_true", help="confirm the calls")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("gate", help="pass/fail on refusal behaviour and judged quality")
    t.add_argument("--dataset", required=True)
    t.add_argument("--version", required=True)
    t.add_argument("--variant", default="baseline")
    t.add_argument("--split", choices=["dev", "test", "all"], default="all")
    t.add_argument("--ci-only", action="store_true")
    t.add_argument("--faith-floor", type=float, default=gate.FAITHFULNESS_FLOOR)
    t.add_argument("--rel-floor", type=float, default=gate.RELEVANCY_FLOOR)
    t.add_argument("--runs", default="runs")
    t.set_defaults(func=cmd_gate)

    j = sub.add_parser("judge", help="LLM-judge stored generations (needs the [judge] extra)")
    j.add_argument("--dataset", required=True)
    j.add_argument("--version", help="adapter version of the generations to judge")
    j.add_argument("--variants", nargs="+", default=["baseline"])
    j.add_argument("--limit", type=int, help="at most N judge calls this run")
    j.add_argument("--gap", type=float, default=4.5, help="minimum seconds between judge calls")
    j.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="for relevancy")
    j.add_argument("--calibrate", action="store_true", help="score known-good vs known-bad answers instead")
    j.add_argument("--runs", default="runs")
    j.add_argument("--yes", action="store_true", help="confirm the calls")
    j.set_defaults(func=cmd_judge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bineval.evaluate import BinEval
from bineval.optimize import SelfPromptUpdate
from bineval.prompts import DEFAULT_EVALUATOR_PROMPT
from bineval.providers import get_provider
from bineval.questions import generate_questions, resolve_questions, save_questions


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", data.get("examples", []))
    return data


def _print_result(result, name: str = "") -> None:
    header = f"[{name}] " if name else ""
    dims = "  ".join(f"{k}={v:.2f}" for k, v in sorted(result.scores.items()))
    scaled = f"  scaled={result.scaled:.2f}" if result.scaled is not None else ""
    print(f"{header}overall={result.overall:.3f}{scaled}")
    print(f"  {dims}")
    failed = result.failing()
    if failed:
        print(f"  failing ({len(failed)}):")
        for verdict in failed:
            print(f"    - {verdict.question_id} [{verdict.dimension}]: {verdict.explanation}")
    else:
        print("  all questions passed")


def cmd_generate(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    task = Path(args.task).read_text(encoding="utf-8") if Path(args.task).exists() else args.task
    dims = tuple(d.strip() for d in args.dimensions.split(",") if d.strip())
    requirements, questions = generate_questions(task, provider, dimensions=dims)
    save_questions(args.out, questions, requirements)
    print(f"wrote {len(questions)} questions ({len(requirements)} requirements) to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    questions = resolve_questions(args.bank, args.questions)
    evaluator = BinEval(questions, provider, evaluator_prompt=DEFAULT_EVALUATOR_PROMPT)
    cases = _load_cases(Path(args.input))
    results = evaluator.evaluate_many(cases)
    payload = [r.to_dict() for r in results]
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for case, result in zip(cases, results):
        _print_result(result, name=str(case.get("id", case.get("name", ""))))
    mean = sum(r.overall for r in results) / len(results)
    print(f"mean overall={mean:.3f} over {len(results)} cases")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    questions = resolve_questions(args.bank, args.questions)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    cases = _load_cases(Path(args.input))
    updater = SelfPromptUpdate(questions, provider, max_iterations=args.iterations)
    updated = updater.run(cases, prompt)
    Path(args.out).write_text(updated + ("\n" if not updated.endswith("\n") else ""), encoding="utf-8")
    for record in updater.history:
        print(f"iteration {record.iteration}: mean={record.mean_overall:.3f} lessons={len(record.lessons)}")
        for lesson in record.lessons:
            print(f"  - {lesson}")
    print(f"wrote updated prompt to {args.out}")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    from bineval.demo_data import CASES, GENERATION_PROMPT, TASK_PROMPT

    provider = get_provider("heuristic")
    print("BinEval demo (offline heuristic evaluator)\n")
    print("1) Generate binary questions from the task prompt")
    _requirements, questions = generate_questions(TASK_PROMPT, provider)
    print(f"   {len(questions)} questions across {sorted({q.dimension for q in questions})}\n")

    print("2) Evaluate sample outputs independently on each question")
    evaluator = BinEval(questions, provider)
    results = []
    for case in CASES:
        result = evaluator.evaluate(case["input"], case["output"])
        results.append(result)
        _print_result(result, name=case["id"])
        print()

    print("3) Self-update the generation prompt from failing questions")
    updater = SelfPromptUpdate(questions, provider, max_iterations=1)
    updated = updater.run(CASES, GENERATION_PROMPT)
    print(f"   original: {GENERATION_PROMPT}")
    print(f"   updated:  {updated.strip()}")
    ranking = sorted(
        zip((c["id"] for c in CASES), results),
        key=lambda item: item[1].overall,
        reverse=True,
    )
    print("\nRanking (higher is better): " + ", ".join(f"{i}={r.overall:.2f}" for i, r in ranking))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bineval",
        description="BinEval: interpretable evaluation via atomic binary questions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-questions", help="Decompose a task prompt into binary questions")
    gen.add_argument("--task", required=True, help="Task prompt text or path")
    gen.add_argument("--out", default="questions.json")
    gen.add_argument("--provider", default="heuristic")
    gen.add_argument("--dimensions", default="coherence,consistency,fluency,relevance")
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("evaluate", help="Score outputs against a binary question bank")
    ev.add_argument("--input", required=True, help="JSON list of {id,input,output}")
    ev.add_argument("--bank", default="summeval")
    ev.add_argument("--questions", default=None, help="JSON question file (overrides --bank)")
    ev.add_argument("--provider", default="heuristic")
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=cmd_evaluate)

    opt = sub.add_parser("optimize", help="Self-update a generation prompt from failures")
    opt.add_argument("--input", required=True)
    opt.add_argument("--prompt", required=True)
    opt.add_argument("--out", default="updated_prompt.txt")
    opt.add_argument("--bank", default="summeval")
    opt.add_argument("--questions", default=None)
    opt.add_argument("--provider", default="heuristic")
    opt.add_argument("--iterations", type=int, default=2)
    opt.set_defaults(func=cmd_optimize)

    demo = sub.add_parser("demo", help="Run the bundled offline walkthrough")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

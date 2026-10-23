from __future__ import annotations

from bineval.prompts import DEFAULT_EVALUATOR_PROMPT, EVALUATION
from bineval.providers import HeuristicProvider, parse_json_object
from bineval.scoring import affine_scale, attach_dimensions, dimension_scores, overall_score
from bineval.types import BinaryQuestion, EvaluationResult, Verdict


def _complete_json(provider, prompt: str) -> dict:
    return parse_json_object(provider.complete(prompt))


def answer_question(
    provider,
    input_text: str,
    output_text: str,
    question: BinaryQuestion,
    evaluator_prompt: str = DEFAULT_EVALUATOR_PROMPT,
) -> Verdict:
    """f_E(x, y, q_i) in {0,1} with explanation e_i. Paper §3.2."""
    if isinstance(provider, HeuristicProvider):
        provider.bind_example(input_text, output_text)
    prompt = EVALUATION.format(
        evaluator_prompt=evaluator_prompt.strip(),
        input_text=input_text.strip(),
        output_text=output_text.strip(),
        question_id=question.id,
        dimension=question.dimension,
        question_text=question.text,
        violation_example=question.violation_example or "(none)",
    )
    payload = _complete_json(provider, prompt)
    answer = int(payload.get("answer", 0))
    if answer not in (0, 1):
        answer = 1 if str(payload.get("answer")).strip().lower() in {"1", "yes", "true"} else 0
    return Verdict(
        question_id=question.id,
        answer=answer,
        explanation=str(payload.get("explanation", "")).strip(),
        dimension=question.dimension,
    )


def evaluate_case(
    provider,
    input_text: str,
    output_text: str,
    questions: list[BinaryQuestion],
    evaluator_prompt: str = DEFAULT_EVALUATOR_PROMPT,
    scale: tuple[float, float] | None = (1.0, 5.0),
) -> EvaluationResult:
    verdicts = [
        answer_question(provider, input_text, output_text, question, evaluator_prompt)
        for question in questions
    ]
    verdicts = attach_dimensions(verdicts, questions)
    overall = overall_score(verdicts)
    scaled = affine_scale(overall, *scale) if scale else None
    return EvaluationResult(
        input_text=input_text,
        output_text=output_text,
        verdicts=verdicts,
        scores=dimension_scores(verdicts),
        overall=overall,
        scaled=scaled,
    )


class BinEval:
    def __init__(
        self,
        questions: list[BinaryQuestion],
        provider,
        evaluator_prompt: str = DEFAULT_EVALUATOR_PROMPT,
        scale: tuple[float, float] | None = (1.0, 5.0),
    ) -> None:
        self.questions = questions
        self.provider = provider
        self.evaluator_prompt = evaluator_prompt
        self.scale = scale

    def evaluate(self, input_text: str, output_text: str) -> EvaluationResult:
        return evaluate_case(
            self.provider,
            input_text,
            output_text,
            self.questions,
            evaluator_prompt=self.evaluator_prompt,
            scale=self.scale,
        )

    def evaluate_many(self, cases: list[dict]) -> list[EvaluationResult]:
        results = []
        for case in cases:
            result = self.evaluate(case["input"], case["output"])
            result.metadata = {k: v for k, v in case.items() if k not in {"input", "output"}}
            results.append(result)
        return results

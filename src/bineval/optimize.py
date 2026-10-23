from __future__ import annotations

from dataclasses import dataclass, field

from bineval.evaluate import BinEval, evaluate_case
from bineval.prompts import NOTE_TAKER_DISAGREEMENT, NOTE_TAKER_FAILURES, PROMPT_UPDATER
from bineval.providers import parse_json_object
from bineval.types import BinaryQuestion, EvaluationResult, Lesson, Verdict


def _lessons_from_payload(payload: dict) -> list[Lesson]:
    rows = payload.get("lessons", [])
    lessons: list[Lesson] = []
    for row in rows:
        if isinstance(row, str):
            lessons.append(Lesson(text=row))
        else:
            lessons.append(
                Lesson(
                    text=str(row.get("text", "")).strip(),
                    dimension=str(row.get("dimension", "")),
                    question_ids=tuple(row.get("question_ids", ())),
                )
            )
    return [lesson for lesson in lessons if lesson.text]


def deduplicate_lessons(lessons: list[Lesson]) -> list[Lesson]:
    """Semantic-lite dedup: normalize whitespace/case (paper §3.3 Dedup)."""
    memory: list[Lesson] = []
    index: dict[str, int] = {}
    for lesson in lessons:
        key = lesson.key()
        if key in index:
            existing = memory[index[key]]
            merged_ids = tuple(dict.fromkeys(existing.question_ids + lesson.question_ids))
            memory[index[key]] = Lesson(
                text=existing.text,
                dimension=existing.dimension or lesson.dimension,
                question_ids=merged_ids,
            )
        else:
            index[key] = len(memory)
            memory.append(lesson)
    return memory


def apply_lesson(provider, prompt: str, lesson: Lesson) -> str:
    payload = parse_json_object(
        provider.complete(PROMPT_UPDATER.format(prompt=prompt, lesson=lesson.text))
    )
    old, new = str(payload.get("old", "")), str(payload.get("new", ""))
    if old and old in prompt:
        return prompt.replace(old, new, 1)
    addition = new or lesson.text
    if addition and addition not in prompt:
        return prompt.rstrip() + " " + addition
    return prompt


def extract_failure_lessons(
    provider,
    result: EvaluationResult,
    context: str = "",
) -> list[Lesson]:
    failing = result.failing()
    if not failing:
        return []
    lines = [
        f"- [{v.dimension}] {v.question_id}: {v.explanation}" for v in failing
    ]
    payload = parse_json_object(
        provider.complete(
            NOTE_TAKER_FAILURES.format(context=context, failures="\n".join(lines))
        )
    )
    lessons = _lessons_from_payload(payload)
    for lesson in lessons:
        if not lesson.question_ids:
            object.__setattr__(
                lesson,
                "question_ids",
                tuple(v.question_id for v in failing if not lesson.dimension or v.dimension == lesson.dimension),
            )
    return lessons


def extract_disagreement_lessons(
    provider,
    input_text: str,
    output_text: str,
    source: list[Verdict],
    target: list[Verdict],
) -> list[Lesson]:
    src = {v.question_id: v for v in source}
    tgt = {v.question_id: v for v in target}
    lines = []
    for qid, s in src.items():
        t = tgt.get(qid)
        if t is None or t.answer == s.answer:
            continue
        lines.append(
            f"- {qid} ({s.dimension}): source={s.answer} ({s.explanation}) "
            f"target={t.answer} ({t.explanation})"
        )
    if not lines:
        return []
    payload = parse_json_object(
        provider.complete(
            NOTE_TAKER_DISAGREEMENT.format(
                input_text=input_text,
                output_text=output_text,
                disagreements="\n".join(lines),
            )
        )
    )
    return _lessons_from_payload(payload)


@dataclass
class UpdateRecord:
    iteration: int
    prompt: str
    mean_overall: float
    lessons: list[str]


@dataclass
class SelfPromptUpdate:
    """Improve a generation prompt from failing binary questions. Paper §3.4."""

    questions: list[BinaryQuestion]
    provider: object
    max_iterations: int = 3
    history: list[UpdateRecord] = field(default_factory=list)

    def run(self, cases: list[dict], prompt: str) -> str:
        current = prompt
        for step in range(1, self.max_iterations + 1):
            results = [
                evaluate_case(self.provider, case["input"], case["output"], self.questions)
                for case in cases
            ]
            mean = sum(r.overall for r in results) / len(results)
            raw: list[Lesson] = []
            for result in results:
                raw.extend(extract_failure_lessons(self.provider, result, context=current))
            lessons = deduplicate_lessons(raw)
            self.history.append(
                UpdateRecord(step, current, mean, [lesson.text for lesson in lessons])
            )
            if not lessons:
                return current
            for lesson in lessons:
                current = apply_lesson(self.provider, current, lesson)
        return current


@dataclass
class CrossModelUpdate:
    """Align a target evaluator prompt to a source evaluator. Paper §3.3."""

    questions: list[BinaryQuestion]
    source_provider: object
    target_provider: object
    note_provider: object
    epsilon: float = 0.05
    max_iterations: int = 3
    history: list[UpdateRecord] = field(default_factory=list)

    def run(self, cases: list[dict], target_prompt: str) -> str:
        current = target_prompt
        source = BinEval(self.questions, self.source_provider)
        for step in range(1, self.max_iterations + 1):
            target = BinEval(self.questions, self.target_provider, evaluator_prompt=current)
            raw: list[Lesson] = []
            src_overall = []
            tgt_overall = []
            for case in cases:
                s = source.evaluate(case["input"], case["output"])
                t = target.evaluate(case["input"], case["output"])
                src_overall.append(s.overall)
                tgt_overall.append(t.overall)
                raw.extend(
                    extract_disagreement_lessons(
                        self.note_provider,
                        case["input"],
                        case["output"],
                        s.verdicts,
                        t.verdicts,
                    )
                )
            mean_gap = abs(
                (sum(src_overall) / len(src_overall)) - (sum(tgt_overall) / len(tgt_overall))
            )
            lessons = deduplicate_lessons(raw)
            self.history.append(
                UpdateRecord(step, current, sum(tgt_overall) / len(tgt_overall), [l.text for l in lessons])
            )
            if mean_gap < self.epsilon or not lessons:
                return current
            for lesson in lessons:
                current = apply_lesson(self.note_provider, current, lesson)
        return current

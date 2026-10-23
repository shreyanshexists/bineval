from __future__ import annotations

import json
from pathlib import Path

from bineval.banks import SUMMEVAL_DIMENSIONS, load_bank
from bineval.prompts import QUESTION_GENERATION
from bineval.providers import parse_json_object
from bineval.types import BinaryQuestion, Requirement


def generate_questions(
    task_prompt: str,
    provider,
    dimensions: tuple[str, ...] | list[str] = SUMMEVAL_DIMENSIONS,
) -> tuple[list[Requirement], list[BinaryQuestion]]:
    """Two-step decomposition Q = F_LLM(T; M). Paper §3.1."""
    prompt = QUESTION_GENERATION.format(
        task_prompt=task_prompt.strip(),
        dimensions=", ".join(dimensions),
    )
    payload = parse_json_object(provider.complete(prompt))
    requirements = [
        Requirement(
            id=str(item.get("id", f"R{i+1}")),
            text=str(item["text"]),
            dimension=str(item.get("dimension", "overall")),
        )
        for i, item in enumerate(payload.get("requirements", []))
    ]
    questions = [BinaryQuestion.from_dict(item) for item in payload.get("questions", [])]
    if not questions:
        raise ValueError("question generation returned no questions")
    return requirements, questions


def load_questions(path: str | Path) -> list[BinaryQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["questions"] if isinstance(data, dict) and "questions" in data else data
    return [BinaryQuestion.from_dict(row) for row in rows]


def save_questions(
    path: str | Path,
    questions: list[BinaryQuestion],
    requirements: list[Requirement] | None = None,
) -> None:
    payload = {
        "requirements": [
            {"id": r.id, "text": r.text, "dimension": r.dimension} for r in (requirements or [])
        ],
        "questions": [q.to_dict() for q in questions],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_questions(bank: str | None, questions_path: str | None) -> list[BinaryQuestion]:
    if questions_path:
        return load_questions(questions_path)
    return load_bank(bank or "summeval")

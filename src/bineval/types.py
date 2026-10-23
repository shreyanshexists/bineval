from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Requirement:
    """Explicit criterion extracted from a task prompt (paper §3.1, step 1)."""

    id: str
    text: str
    dimension: str


@dataclass(frozen=True)
class BinaryQuestion:
    """Atomic yes/no probe. Yes means the output satisfies the criterion."""

    id: str
    text: str
    dimension: str
    requirement_id: str = ""
    violation_example: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinaryQuestion:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            dimension=str(data["dimension"]),
            requirement_id=str(data.get("requirement_id", "")),
            violation_example=str(data.get("violation_example", "")),
        )


@dataclass(frozen=True)
class Verdict:
    """Independent binary answer plus a short natural-language explanation."""

    question_id: str
    answer: int
    explanation: str
    dimension: str = ""

    def __post_init__(self) -> None:
        if self.answer not in (0, 1):
            raise ValueError(f"answer must be 0 or 1, got {self.answer}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Per-example BinEval scores grounded in question-level verdicts."""

    input_text: str
    output_text: str
    verdicts: list[Verdict]
    scores: dict[str, float]
    overall: float
    scaled: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def failing(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.answer == 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "scores": self.scores,
            "overall": self.overall,
            "scaled": self.scaled,
            "metadata": self.metadata,
        }


@dataclass
class Lesson:
    """Generalized diagnostic extracted from failures or disagreements."""

    text: str
    dimension: str = ""
    question_ids: tuple[str, ...] = ()

    def key(self) -> str:
        return " ".join(self.text.lower().split())

from __future__ import annotations

from collections import defaultdict

from bineval.types import BinaryQuestion, Verdict


def overall_score(verdicts: list[Verdict]) -> float:
    """Mean of binary answers. Paper §3.2: S(x,y) = (1/N) Σ f_E(x,y,q_i)."""
    if not verdicts:
        raise ValueError("cannot score an empty verdict list")
    return sum(v.answer for v in verdicts) / len(verdicts)


def dimension_scores(verdicts: list[Verdict]) -> dict[str, float]:
    """Per-dimension mean. Paper §3.2: S_d(x,y) = (1/|Q_d|) Σ_{q in Q_d} f_E."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for verdict in verdicts:
        dim = verdict.dimension or "overall"
        buckets[dim].append(verdict.answer)
    return {dim: sum(vals) / len(vals) for dim, vals in buckets.items()}


def affine_scale(score: float, low: float = 1.0, high: float = 5.0) -> float:
    """Map [0, 1] onto [low, high]. Paper §3.2: S' = S·(b-a)+a."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must lie in [0, 1], got {score}")
    if high < low:
        raise ValueError("high must be >= low")
    return score * (high - low) + low


def attach_dimensions(
    verdicts: list[Verdict], questions: list[BinaryQuestion]
) -> list[Verdict]:
    by_id = {q.id: q for q in questions}
    filled: list[Verdict] = []
    for verdict in verdicts:
        question = by_id.get(verdict.question_id)
        dim = verdict.dimension or (question.dimension if question else "")
        filled.append(
            Verdict(
                question_id=verdict.question_id,
                answer=verdict.answer,
                explanation=verdict.explanation,
                dimension=dim,
            )
        )
    return filled

"""BinEval: evaluate generations with atomic yes/no questions.

Implements the framework from Cho et al., "Ask, Don't Judge: Binary Questions
for Interpretable LLM Evaluation and Self-Improvement" (arXiv:2606.27226).
"""

from bineval.evaluate import BinEval, evaluate_case
from bineval.optimize import CrossModelUpdate, SelfPromptUpdate
from bineval.questions import generate_questions
from bineval.scoring import affine_scale, dimension_scores, overall_score
from bineval.types import (
    BinaryQuestion,
    EvaluationResult,
    Requirement,
    Verdict,
)

__all__ = [
    "BinEval",
    "BinaryQuestion",
    "CrossModelUpdate",
    "EvaluationResult",
    "Requirement",
    "SelfPromptUpdate",
    "Verdict",
    "affine_scale",
    "dimension_scores",
    "evaluate_case",
    "generate_questions",
    "overall_score",
]

__version__ = "0.1.0"

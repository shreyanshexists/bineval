from bineval.banks import load_bank
from bineval.demo_data import CASES
from bineval.evaluate import BinEval
from bineval.providers import HeuristicProvider
from bineval.questions import generate_questions


def test_summeval_bank_has_paper_counts():
    questions = load_bank("summeval")
    counts = {}
    for q in questions:
        counts[q.dimension] = counts.get(q.dimension, 0) + 1
    assert counts == {"coherence": 8, "consistency": 7, "fluency": 7, "relevance": 5}
    assert len(questions) == 27


def test_faithful_summary_outranks_failures():
    evaluator = BinEval(load_bank("summeval"), HeuristicProvider())
    by_id = {c["id"]: evaluator.evaluate(c["input"], c["output"]) for c in CASES}
    assert by_id["faithful"].overall > by_id["hallucinated"].overall
    assert by_id["faithful"].overall > by_id["garbled"].overall
    assert by_id["faithful"].scaled > 3.0
    assert by_id["hallucinated"].scores["consistency"] < by_id["faithful"].scores["consistency"]
    assert by_id["garbled"].scores["fluency"] < by_id["faithful"].scores["fluency"]


def test_each_verdict_carries_an_explanation():
    case = CASES[0]
    result = BinEval(load_bank("summeval"), HeuristicProvider()).evaluate(
        case["input"], case["output"]
    )
    assert len(result.verdicts) == 27
    assert all(v.explanation for v in result.verdicts)
    assert set(result.scores) == {"coherence", "consistency", "fluency", "relevance"}


def test_offline_question_generation_returns_summeval_bank():
    reqs, questions = generate_questions("Summarize the article.", HeuristicProvider())
    assert reqs
    assert {q.dimension for q in questions} == {
        "coherence",
        "consistency",
        "fluency",
        "relevance",
    }

import pytest

from bineval.scoring import affine_scale, dimension_scores, overall_score
from bineval.types import Verdict


def test_overall_is_mean_of_binary_answers():
    verdicts = [
        Verdict("q1", 1, "yes", "coherence"),
        Verdict("q2", 0, "no", "coherence"),
        Verdict("q3", 1, "yes", "fluency"),
    ]
    assert overall_score(verdicts) == pytest.approx(2 / 3)


def test_dimension_scores_partition_questions():
    verdicts = [
        Verdict("q1", 1, "yes", "coherence"),
        Verdict("q2", 0, "no", "coherence"),
        Verdict("q3", 1, "yes", "fluency"),
        Verdict("q4", 1, "yes", "fluency"),
    ]
    scores = dimension_scores(verdicts)
    assert scores["coherence"] == pytest.approx(0.5)
    assert scores["fluency"] == pytest.approx(1.0)


def test_affine_maps_unit_interval_to_likert():
    assert affine_scale(0.0, 1, 5) == pytest.approx(1.0)
    assert affine_scale(1.0, 1, 5) == pytest.approx(5.0)
    assert affine_scale(3 / 7, 1, 5) == pytest.approx(3 / 7 * 4 + 1)


def test_empty_verdicts_raise():
    with pytest.raises(ValueError):
        overall_score([])


def test_verdict_rejects_non_binary():
    with pytest.raises(ValueError):
        Verdict("q1", 2, "bad")

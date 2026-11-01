import pytest

from src.evaluation.retriever_evaluator import RetrieverEvaluator


def test_precision_at_k():
    retrieved = ["INC001", "INC002", "INC003"]
    relevant = {"INC001"}

    result = RetrieverEvaluator.precision_at_k(
        retrieved, relevant, 3
    )

    assert result == pytest.approx(1 / 3)


def test_recall_at_k():
    retrieved = ["INC001", "INC002", "INC003"]
    relevant = {"INC001", "INC003"}

    result = RetrieverEvaluator.recall_at_k(
        retrieved, relevant, 2
    )

    assert result == pytest.approx(0.5)


def test_reciprocal_rank():
    retrieved = ["INC002", "INC001", "INC003"]
    relevant = {"INC001"}

    result = RetrieverEvaluator.reciprocal_rank(
        retrieved, relevant
    )

    assert result == pytest.approx(0.5)


def test_reciprocal_rank_when_no_relevant_result():
    retrieved = ["INC002", "INC003"]
    relevant = {"INC001"}

    result = RetrieverEvaluator.reciprocal_rank(
        retrieved, relevant
    )

    assert result == 0.0


def test_precision_rejects_invalid_k():
    with pytest.raises(ValueError):
        RetrieverEvaluator.precision_at_k(
            ["INC001"], {"INC001"}, 0
        )


def test_recall_rejects_empty_relevant_ids():
    with pytest.raises(ValueError):
        RetrieverEvaluator.recall_at_k(
            ["INC001"], set(), 1
        )
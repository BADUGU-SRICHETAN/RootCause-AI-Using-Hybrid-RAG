import pytest

from src.evaluation.retriever_evaluator import RetrieverEvaluator


def test_hit_rate_at_k():

    retrieved_ids = ["INC001", "INC002", "INC003"]
    relevant_ids = {"INC002"}

    hit_rate = RetrieverEvaluator.hit_rate_at_k(
        retrieved_ids,
        relevant_ids,
        k=3
    )

    assert hit_rate == 1.0


def test_hit_rate_when_no_relevant_result():

    retrieved_ids = ["INC001", "INC002"]
    relevant_ids = {"INC005"}

    hit_rate = RetrieverEvaluator.hit_rate_at_k(
        retrieved_ids,
        relevant_ids,
        k=2
    )

    assert hit_rate == 0.0


def test_ndcg_at_k():

    retrieved_ids = ["INC001", "INC002", "INC003"]
    relevant_ids = {"INC002", "INC003"}

    ndcg = RetrieverEvaluator.ndcg_at_k(
        retrieved_ids,
        relevant_ids,
        k=3
    )

    assert 0.0 < ndcg <= 1.0


def test_ndcg_perfect_ranking():

    retrieved_ids = ["INC001", "INC002"]
    relevant_ids = {"INC001", "INC002"}

    ndcg = RetrieverEvaluator.ndcg_at_k(
        retrieved_ids,
        relevant_ids,
        k=2
    )

    assert ndcg == pytest.approx(1.0)


def test_ndcg_zero_when_no_relevant_result():

    retrieved_ids = ["INC003", "INC004"]
    relevant_ids = {"INC001"}

    ndcg = RetrieverEvaluator.ndcg_at_k(
        retrieved_ids,
        relevant_ids,
        k=2
    )

    assert ndcg == 0.0


def test_ndcg_penalizes_missing_relevant_documents():

    retrieved_ids = ["INC001", "INC003"]
    relevant_ids = {"INC001", "INC002"}

    ndcg = RetrieverEvaluator.ndcg_at_k(
        retrieved_ids,
        relevant_ids,
        k=2
    )

    expected = 1.0 / (
        1.0 + 1.0 / 1.0  # Placeholder replaced below
    )

    ideal_dcg = 1.0 + (1.0 / 1.584962500721156)
    expected = 1.0 / ideal_dcg

    assert ndcg == pytest.approx(expected)


def test_ndcg_zero_when_all_relevant_documents_are_missing():

    retrieved_ids = ["INC003", "INC004"]
    relevant_ids = {"INC001", "INC002"}

    ndcg = RetrieverEvaluator.ndcg_at_k(
        retrieved_ids,
        relevant_ids,
        k=2
    )

    assert ndcg == 0.0


def test_hit_rate_rejects_invalid_k():

    with pytest.raises(ValueError):

        RetrieverEvaluator.hit_rate_at_k(
            ["INC001"],
            {"INC001"},
            k=0
        )


def test_ndcg_rejects_empty_relevant_ids():

    with pytest.raises(ValueError):

        RetrieverEvaluator.ndcg_at_k(
            ["INC001"],
            set(),
            k=2
        )


def test_ndcg_rejects_invalid_k():

    with pytest.raises(ValueError):

        RetrieverEvaluator.ndcg_at_k(
            ["INC001"],
            {"INC001"},
            k=0
        )
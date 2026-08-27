from math import isclose, log2

from recsys.evaluation.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_single_target_ranking_metrics_are_hand_calculable() -> None:
    ranking = [30, 20, 10]

    assert recall_at_k(ranking, 20, 1) == 0.0
    assert recall_at_k(ranking, 20, 2) == 1.0
    assert reciprocal_rank_at_k(ranking, 20, 3) == 0.5
    assert isclose(ndcg_at_k(ranking, 20, 3), 1.0 / log2(3))


def test_coverage_and_bootstrap_are_deterministic() -> None:
    assert catalog_coverage([[1, 2], [2, 3]], {1, 2, 3, 4}) == 0.75
    first = bootstrap_mean_ci([0.0, 1.0, 1.0], samples=100, seed=42)
    second = bootstrap_mean_ci([0.0, 1.0, 1.0], samples=100, seed=42)
    assert first == second


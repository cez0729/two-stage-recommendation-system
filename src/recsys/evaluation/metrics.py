"""Ranking metrics for one held-out relevant item per recommendation query."""

from collections.abc import Iterable, Sequence
from math import log2

import numpy as np


def target_rank(ranked_items: Sequence[int], target_item: int, k: int) -> int | None:
    """Return the target's one-based rank within top-k, or None when absent."""
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item == target_item:
            return rank
    return None


def recall_at_k(ranked_items: Sequence[int], target_item: int, k: int) -> float:
    """Return 1 when the single target occurs in top-k, otherwise 0."""
    return float(target_rank(ranked_items, target_item, k) is not None)


def reciprocal_rank_at_k(ranked_items: Sequence[int], target_item: int, k: int) -> float:
    """Return reciprocal target rank within top-k, otherwise 0."""
    rank = target_rank(ranked_items, target_item, k)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked_items: Sequence[int], target_item: int, k: int) -> float:
    """Return normalized discounted gain for a single binary target."""
    rank = target_rank(ranked_items, target_item, k)
    return 0.0 if rank is None else 1.0 / log2(rank + 1)


def catalog_coverage(recommendations: Iterable[Sequence[int]], catalog: set[int]) -> float:
    """Return the fraction of eligible catalog items recommended at least once."""
    if not catalog:
        return 0.0
    recommended = {item for ranking in recommendations for item in ranking if item in catalog}
    return len(recommended) / len(catalog)


def percentile_latency(latencies_ms: Sequence[float], percentile: float) -> float:
    """Calculate a latency percentile, returning zero for an empty sample."""
    if not latencies_ms:
        return 0.0
    return float(np.percentile(np.asarray(latencies_ms), percentile))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic non-parametric bootstrap interval for a mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 128
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)

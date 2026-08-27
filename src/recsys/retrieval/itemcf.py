"""Memory-bounded item-based collaborative filtering for implicit feedback."""

import argparse
from collections import defaultdict
from itertools import combinations
from math import log1p, sqrt
from pathlib import Path

import pandas as pd

from recsys.config import load_yaml
from recsys.logging import configure_logging
from recsys.retrieval.candidate_filter import Candidate, CandidateFilter
from recsys.retrieval.popularity import PopularityRecommender


class ItemCFRecommender:
    """Retrieve items from weighted, cosine-normalized item co-occurrence."""

    def __init__(
        self,
        *,
        max_history: int = 50,
        top_neighbors: int = 200,
        recency_decay: float = 0.9,
    ) -> None:
        if max_history < 2 or top_neighbors < 1:
            raise ValueError("max_history must be >= 2 and top_neighbors must be >= 1")
        if not 0.0 < recency_decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        self.max_history = max_history
        self.top_neighbors = top_neighbors
        self.recency_decay = recency_decay
        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        self.catalog: set[int] = set()
        self.popularity = PopularityRecommender()
        self._filter: CandidateFilter | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemCFRecommender":
        """Build sparse top-neighbor lists from retrieval-training histories."""
        ordered = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
        self.popularity.fit(ordered)
        self.catalog = self.popularity.catalog
        item_counts = self.popularity.item_counts
        cooccurrence: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for _, group in ordered.groupby("user_idx", sort=False):
            history = group["item_idx"].astype(int).tolist()[-self.max_history :]
            weight = 1.0 / log1p(len(history))
            for left, right in combinations(history, 2):
                cooccurrence[left][right] += weight
                cooccurrence[right][left] += weight

        for item, related in cooccurrence.items():
            scored = [
                (neighbor, value / sqrt(item_counts[item] * item_counts[neighbor]))
                for neighbor, value in related.items()
            ]
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            self.neighbors[item] = scored[: self.top_neighbors]

        self._filter = CandidateFilter(self.catalog, fallback=self.popularity.ranking)
        return self

    def recommend(self, history: list[int], k: int) -> list[Candidate]:
        """Score neighbor items from recent history and backfill from popularity."""
        if self._filter is None:
            raise RuntimeError("ItemCFRecommender must be fitted before recommendation")
        recent = history[-self.max_history :]
        scores: dict[int, float] = defaultdict(float)
        for distance, item in enumerate(reversed(recent)):
            history_weight = self.recency_decay**distance
            for neighbor, similarity in self.neighbors.get(item, []):
                scores[neighbor] += history_weight * similarity
        candidates = [
            Candidate(item_idx=item, score=score, source="itemcf")
            for item, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        return self._filter.apply(candidates, seen_items=set(history), k=k)

    def neighbors_frame(self) -> pd.DataFrame:
        """Return sparse neighbor lists as a persistable table."""
        rows = []
        for item, neighbors in sorted(self.neighbors.items()):
            for rank, (neighbor, similarity) in enumerate(neighbors, start=1):
                rows.append(
                    {
                        "item_idx": item,
                        "neighbor_idx": neighbor,
                        "similarity": similarity,
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows, columns=["item_idx", "neighbor_idx", "similarity", "rank"])


def main() -> None:
    from recsys.evaluation.evaluator import run_baseline_evaluation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/itemcf.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_yaml(args.config)
    interactions = pd.read_parquet(config["evaluation"]["data_path"])
    model_config = config["model"]
    model = ItemCFRecommender(
        max_history=int(model_config["max_history"]),
        top_neighbors=int(model_config["top_neighbors"]),
        recency_decay=float(model_config["recency_decay"]),
    ).fit(interactions[interactions["split"] == "retrieval_train"])
    output = Path(config["artifacts"]["neighbors_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.neighbors_frame().to_parquet(output, index=False)
    run_baseline_evaluation(model, config)


if __name__ == "__main__":
    main()

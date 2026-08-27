"""Global popularity retrieval baseline."""

import argparse
from math import log1p
from pathlib import Path

import pandas as pd

from recsys.config import load_yaml
from recsys.logging import configure_logging
from recsys.retrieval.candidate_filter import Candidate, CandidateFilter


class PopularityRecommender:
    """Rank items by retrieval-training interaction count."""

    def __init__(self) -> None:
        self.ranking: list[Candidate] = []
        self.catalog: set[int] = set()
        self.item_counts: dict[int, int] = {}
        self._filter: CandidateFilter | None = None

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Fit deterministic popularity scores from training interactions only."""
        counts = interactions["item_idx"].astype(int).value_counts().to_dict()
        self.item_counts = {int(item): int(count) for item, count in counts.items()}
        self.catalog = set(self.item_counts)
        self.ranking = sorted(
            [
                Candidate(item_idx=item, score=log1p(count), source="popularity")
                for item, count in self.item_counts.items()
            ],
            key=lambda candidate: (-candidate.score, candidate.item_idx),
        )
        self._filter = CandidateFilter(self.catalog, fallback=[])
        return self

    def recommend(self, history: list[int] | set[int], k: int) -> list[Candidate]:
        """Return the most popular unique items not already seen."""
        if self._filter is None:
            raise RuntimeError("PopularityRecommender must be fitted before recommendation")
        return self._filter.apply(self.ranking, seen_items=set(history), k=k)

    def to_frame(self) -> pd.DataFrame:
        """Return the learned ranking as a persistable table."""
        return pd.DataFrame(
            [
                {
                    "item_idx": candidate.item_idx,
                    "interaction_count": self.item_counts[candidate.item_idx],
                    "popularity_score": candidate.score,
                    "rank": rank,
                }
                for rank, candidate in enumerate(self.ranking, start=1)
            ]
        )


def main() -> None:
    from recsys.evaluation.evaluator import run_baseline_evaluation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popularity.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_yaml(args.config)
    interactions = pd.read_parquet(config["evaluation"]["data_path"])
    model = PopularityRecommender().fit(interactions[interactions["split"] == "retrieval_train"])
    output = Path(config["artifacts"]["popularity_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.to_frame().to_parquet(output, index=False)
    run_baseline_evaluation(model, config)


if __name__ == "__main__":
    main()

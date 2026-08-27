import pandas as pd

from recsys.evaluation.evaluator import evaluate_split
from recsys.retrieval.candidate_filter import Candidate


class StaticRecommender:
    catalog = {1, 2, 3, 4, 5}
    item_counts = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    def __init__(self) -> None:
        self.histories: list[list[int]] = []

    def recommend(self, history: list[int], k: int) -> list[Candidate]:
        self.histories.append(history)
        return [Candidate(4, 2.0, "static"), Candidate(5, 1.0, "static")]


def test_evaluator_uses_split_specific_history() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 0],
            "item_idx": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime([1, 2, 3, 4, 5], unit="D", utc=True),
            "split": ["retrieval_train", "retrieval_train", "rank_train", "validation", "test"],
        }
    )
    model = StaticRecommender()

    validation, _ = evaluate_split(
        model,
        interactions,
        model_name="static",
        split="validation",
        ks=[10],
        bootstrap_samples=100,
        seed=42,
    )
    test, _ = evaluate_split(
        model,
        interactions,
        model_name="static",
        split="test",
        ks=[10],
        bootstrap_samples=100,
        seed=42,
    )

    assert model.histories == [[1, 2, 3], [1, 2, 3, 4]]
    assert validation["metrics"]["recall@10"] == 1.0
    assert test["metrics"]["mrr@10"] == 0.5


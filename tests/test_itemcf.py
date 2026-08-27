import pandas as pd

from recsys.retrieval.itemcf import ItemCFRecommender


def test_itemcf_builds_symmetric_normalized_neighbors() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [1, 2, 1, 3],
            "timestamp": pd.to_datetime([1, 2, 1, 2], unit="D", utc=True),
        }
    )
    model = ItemCFRecommender(max_history=10, top_neighbors=10).fit(interactions)

    assert model.neighbors[1][0][0] == 2
    assert model.neighbors[2][0][0] == 1
    assert model.neighbors[1][0][1] > 0


def test_itemcf_filters_history_and_uses_popularity_fallback() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [1, 2, 1, 3],
            "timestamp": pd.to_datetime([1, 2, 1, 2], unit="D", utc=True),
        }
    )
    model = ItemCFRecommender(max_history=10, top_neighbors=10).fit(interactions)

    result = model.recommend(history=[2], k=2)

    assert result[0].item_idx == 1
    assert all(candidate.item_idx != 2 for candidate in result)
    assert len(result) == 2


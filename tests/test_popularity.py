import pandas as pd

from recsys.retrieval.popularity import PopularityRecommender


def test_popularity_ranks_by_count_and_excludes_seen() -> None:
    interactions = pd.DataFrame({"item_idx": [2, 1, 1, 3, 3, 3]})
    model = PopularityRecommender().fit(interactions)

    result = model.recommend(history=[3], k=2)

    assert [candidate.item_idx for candidate in result] == [1, 2]
    assert all(candidate.item_idx != 3 for candidate in result)


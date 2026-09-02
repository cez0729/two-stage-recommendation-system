import pandas as pd

from recsys.data.metadata import normalize_brand, normalize_category_path, normalize_text
from recsys.evaluation.temporal import build_global_cutoff_queries
from recsys.retrieval.content import TfidfContentRecommender


def test_metadata_normalization_removes_presentation_noise() -> None:
    assert normalize_text("  A &amp; B\nGame ") == "A & B Game"
    assert normalize_brand("by\n  Nintendo") == "Nintendo"
    assert normalize_category_path(["Video Games", "PC", "<span>bad</span>"]) == "Video Games > PC"


def test_content_retriever_can_recommend_metadata_only_item() -> None:
    items = pd.DataFrame(
        {
            "item_idx": [0, 1, 2],
            "title": ["space strategy game", "space battle game", "cooking recipes"],
            "brand": ["A", "B", "C"],
            "fine_category": ["PC > Strategy", "PC > Strategy", "Books > Food"],
        }
    )
    model = TfidfContentRecommender(min_df=1, ngram_max=1).fit(items, {0, 2})
    recommendations = model.recommend([0], 2)
    assert recommendations[0].item_idx == 1
    assert recommendations[0].source == "content"


def test_global_cutoff_queries_never_use_later_history() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1],
            "item_idx": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                ["2015-01-01", "2016-02-01", "2017-01-01", "2016-02-01"], utc=True
            ),
        }
    )
    train, queries = build_global_cutoff_queries(
        interactions,
        cutoff=pd.Timestamp("2016-01-01", tz="UTC"),
        end=pd.Timestamp("2017-01-01", tz="UTC"),
    )
    assert train["item_idx"].tolist() == [1]
    assert queries[["user_idx", "target_item_idx"]].values.tolist() == [[0, 2]]
    assert queries.iloc[0]["history"] == [1]

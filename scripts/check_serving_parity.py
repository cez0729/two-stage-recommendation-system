"""Compare frozen serving recommendations with saved offline test candidates."""

import argparse

import lightgbm as lgb
import pandas as pd

from recsys.config import load_yaml
from recsys.ranking.pipeline import FEATURE_COLUMNS
from recsys.serving.pipeline import ServingPipeline


def check(config_path: str, users: int = 20) -> dict[str, int | bool]:
    config = load_yaml(config_path)
    pipeline = ServingPipeline.from_yaml(config_path)
    interactions = pd.read_parquet(config["data"]["interactions_path"])
    features = pd.read_parquet("artifacts/video_games_2018/ranking/test.parquet")
    available_users = {
        int(value.split(":", 1)[1]) for value in features["query_id"].drop_duplicates()
    }
    test = interactions[
        (interactions["split"] == "test")
        & interactions["user_idx"].isin(sorted(available_users)[:users])
    ].sort_values("user_idx")
    ranker = lgb.Booster(model_file=config["serving"]["ranker_path"])
    matched = 0
    for row in test.itertuples(index=False):
        query = features[features["query_id"] == f"test:{row.user_idx}"]
        predictions = ranker.predict(query[FEATURE_COLUMNS])
        offline_ids = (
            query.iloc[predictions.argsort()[::-1]]["item_idx"].astype(int).head(10).tolist()
        )
        online_ids = [
            item.item_idx for item in pipeline.recommend_at(row.user_id, 10, row.timestamp)
        ]
        matched += int(offline_ids == online_ids)
    result = {"users": len(test), "matched": matched, "all_equal": matched == len(test)}
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--users", type=int, default=20)
    args = parser.parse_args()
    check(args.config, args.users)


if __name__ == "__main__":
    main()

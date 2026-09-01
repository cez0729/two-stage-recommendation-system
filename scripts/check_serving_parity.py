"""Compare frozen serving recommendations with saved offline test candidates."""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.config import load_yaml
from recsys.ranking.pipeline import FEATURE_COLUMNS
from recsys.serving.pipeline import ServingPipeline


def check(
    config_path: str,
    users: int = 20,
    features_path: str | None = None,
) -> dict[str, int | bool]:
    config = load_yaml(config_path)
    pipeline = ServingPipeline.from_yaml(config_path)
    interactions = pd.read_parquet(config["data"]["interactions_path"])
    if features_path is None:
        ranker_path = str(config["serving"].get("ranker_path", ""))
        features_path = (
            "artifacts/video_games_2018/ranking_phase6_content/test.parquet"
            if "phase6_content" in ranker_path
            else "artifacts/video_games_2018/ranking/test.parquet"
        )
    features = pd.read_parquet(features_path)
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
        # Match ServingPipeline's stable descending order for tied LightGBM scores.
        order = np.argsort(-predictions, kind="stable")
        offline_ids = query.iloc[order]["item_idx"].astype(int).head(10).tolist()
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
    parser.add_argument("--features", default=None, help="Offline candidate feature parquet")
    args = parser.parse_args()
    check(args.config, args.users, args.features)


if __name__ == "__main__":
    main()

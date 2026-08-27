"""Evaluate the frozen LambdaRank model on test exactly once."""

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

from recsys.config import load_yaml
from recsys.ranking.pipeline import FEATURE_COLUMNS, FeaturePipeline, _ranking_metrics
from recsys.utils.io import write_json


def evaluate_final_test(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    output = config["outputs"]
    pipeline = FeaturePipeline(config)
    test = pipeline.build("test")
    feature_path = Path(output["feature_dir"]) / "test.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    test.frame.to_parquet(feature_path, index=False)

    model = lgb.Booster(model_file=output["model_path"])
    predictions = model.predict(test.frame[FEATURE_COLUMNS])
    baseline_metrics, baseline_users = _ranking_metrics(
        test.frame, None, total_queries=test.total_queries
    )
    ranker_metrics, ranker_users = _ranking_metrics(
        test.frame, predictions, total_queries=test.total_queries
    )
    comparison = baseline_users.merge(
        ranker_users,
        on="query_id",
        suffixes=("_retrieval", "_ranker"),
        validate="one_to_one",
    )
    comparison.to_parquet(output["test_per_user_path"], index=False)
    payload = {
        "split": "test",
        "final": True,
        "model_path": output["model_path"],
        "features": FEATURE_COLUMNS,
        "total_queries": test.total_queries,
        "retrieved_queries": test.retrieved_queries,
        "candidate_recall@200": test.retrieved_queries / test.total_queries,
        "retrieval_order_metrics": baseline_metrics,
        "lambdarank_metrics": ranker_metrics,
        "acceptance": {
            "ndcg@10_not_worse": ranker_metrics["ndcg@10"] >= baseline_metrics["ndcg@10"],
            "ndcg@10_delta": ranker_metrics["ndcg@10"] - baseline_metrics["ndcg@10"],
        },
    }
    write_json(output["final_test_metrics_path"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ranker.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_final_test(args.config), indent=2))


if __name__ == "__main__":
    main()

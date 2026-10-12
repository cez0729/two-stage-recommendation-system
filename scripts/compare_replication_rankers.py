"""Compare second-category rankers, including users missed by retrieval as zeros."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.evaluation.metrics import bootstrap_mean_ci
from recsys.utils.io import write_json


def main() -> None:
    root = Path("results/musical_instruments_2018")
    interactions = pd.read_parquet(
        "data/processed/musical_instruments_2018/interactions.parquet",
        columns=["user_idx", "split"],
    )
    users = interactions.loc[interactions["split"] == "test", "user_idx"]
    query_ids = pd.Index([f"test:{int(user)}" for user in users], name="query_id")
    if not query_ids.is_unique:
        raise ValueError("Expected one test query per user")
    files = {"three": "ranker_test_per_user.parquet",
             "four": "ranker_phase6_content_test_per_user.parquet"}
    values = {}
    metric_files = {"three": "final_test_metrics.json",
                    "four": "final_test_phase6_content_metrics.json"}
    for name, filename in files.items():
        frame = pd.read_parquet(root / filename).set_index("query_id")
        if not frame.index.is_unique or not frame.index.isin(query_ids).all():
            raise ValueError("Unexpected or repeated test queries")
        ranks = frame["target_rank_ranker"].reindex(query_ids).fillna(0).to_numpy()
        hits = (ranks > 0) & (ranks <= 10)
        values[name] = {
            "recall@10": hits.astype(float),
            "ndcg@10": np.where(hits, 1 / np.log2(np.maximum(ranks, 1) + 1), 0),
        }
        frozen = json.loads((root / metric_files[name]).read_text())
        for metric, array in values[name].items():
            if not np.isclose(array.mean(), frozen["lambdarank_metrics"][metric]):
                raise ValueError("Per-user reconstruction disagrees with frozen test metrics")
    comparisons = {}
    for metric in ("recall@10", "ndcg@10"):
        delta = values["four"][metric] - values["three"][metric]
        low, high = bootstrap_mean_ci(delta.tolist(), samples=2000, seed=42)
        comparisons[metric] = {
            "three_channel_ranker": float(values["three"][metric].mean()),
            "four_channel_ranker": float(values["four"][metric].mean()),
            "four_minus_three": float(delta.mean()),
            "paired_ci95": {"low": low, "high": high},
        }
    payload = {
        "users": len(query_ids), "bootstrap_samples": 2000, "seed": 42,
        "primary_endpoint": "ndcg@10",
        "missing_candidate_handling": "All retrieval misses contribute zero; same test users",
        "scope": "End-to-end comparison changes candidate composition and ranker features",
        "comparisons": comparisons,
        "acceptance": {
            "primary_ci_strictly_positive": comparisons["ndcg@10"]["paired_ci95"]["low"] > 0
        },
        "limitations": ["One training seed", "No online business outcome evidence"],
    }
    write_json(root / "paired_ranker_comparison.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

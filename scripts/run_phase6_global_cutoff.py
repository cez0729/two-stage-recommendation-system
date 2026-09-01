"""Test content retrieval under pre-registered global temporal cutoffs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from run_phase6_content import _comparison, _rrf, _summarize, _target_groups

from recsys.config import load_yaml
from recsys.evaluation.temporal import build_global_cutoff_queries
from recsys.retrieval.content import TfidfContentRecommender
from recsys.retrieval.itemcf import ItemCFRecommender
from recsys.retrieval.popularity import PopularityRecommender
from recsys.utils.io import sha256_file, write_json


def _rankings(model: Any, histories: list[list[int]], k: int) -> list[list[int]]:
    return [
        [candidate.item_idx for candidate in model.recommend(history, k)] for history in histories
    ]


def _run_period(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    end: pd.Timestamp | None,
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    evaluation = config["evaluation"]
    content_config = config["content"]
    train, queries = build_global_cutoff_queries(
        interactions,
        cutoff=cutoff,
        end=end,
        max_history=int(content_config["max_history"]),
    )
    histories = queries["history"].tolist()
    targets = queries["target_item_idx"].to_numpy(dtype=np.int32)
    counts = train["item_idx"].astype(int).value_counts().to_dict()
    train_catalog = set(counts)
    candidate_k = int(evaluation["retrieval_k"])
    report_k = int(evaluation["report_k"])
    constant = int(evaluation["rrf_constant"])

    content = TfidfContentRecommender(
        max_features=int(content_config["max_features"]),
        min_df=int(content_config["min_df"]),
        ngram_max=int(content_config["ngram_max"]),
        max_history=int(content_config["max_history"]),
    ).fit(items, train_catalog)
    popularity = PopularityRecommender().fit(train)
    itemcf = ItemCFRecommender(max_history=50, top_neighbors=200, recency_decay=0.9).fit(train)
    rankings = {
        "popularity": _rankings(popularity, histories, candidate_k),
        "itemcf": _rankings(itemcf, histories, candidate_k),
        "content": content.recommend_batch(
            histories,
            candidate_k,
            batch_size=int(content_config["batch_size"]),
        ),
    }
    rankings["two_collaborative"] = _rrf(
        [rankings["popularity"], rankings["itemcf"]], candidate_k, constant
    )
    rankings["hybrid_content_collaborative"] = _rrf(
        [rankings["popularity"], rankings["itemcf"], rankings["content"]],
        candidate_k,
        constant,
    )

    groups, thresholds = _target_groups(targets, counts)
    metrics = {}
    grouped_frames = []
    per_user = queries[["user_idx", "target_item_idx", "target_timestamp"]].copy()
    per_user["group"] = groups
    for name, model_rankings in rankings.items():
        summary, grouped = _summarize(
            model_rankings,
            targets,
            groups,
            counts,
            len(content.catalog),
            report_k=report_k,
            candidate_k=candidate_k,
        )
        metrics[name] = summary
        grouped.insert(0, "model", name)
        grouped_frames.append(grouped)
        per_user[f"{name}_hit@100"] = [
            int(int(target) in ranking[:report_k])
            for ranking, target in zip(model_rankings, targets, strict=True)
        ]
        per_user[f"{name}_hit@200"] = [
            int(int(target) in ranking[:candidate_k])
            for ranking, target in zip(model_rankings, targets, strict=True)
        ]

    samples = int(evaluation["bootstrap_samples"])
    seed = int(config["seed"])
    comparisons = {
        "content_minus_itemcf_strict_cold_r100": _comparison(
            rankings["content"],
            rankings["itemcf"],
            targets,
            groups == "strict_cold",
            k=report_k,
            samples=samples,
            seed=seed,
        ),
        "hybrid_minus_collaborative_tail_r100": _comparison(
            rankings["hybrid_content_collaborative"],
            rankings["two_collaborative"],
            targets,
            groups == "tail",
            k=report_k,
            samples=samples,
            seed=seed,
        ),
        "hybrid_minus_collaborative_overall_cr200": _comparison(
            rankings["hybrid_content_collaborative"],
            rankings["two_collaborative"],
            targets,
            np.ones(len(targets), dtype=bool),
            k=candidate_k,
            samples=samples,
            seed=seed,
        ),
    }
    result = {
        "label": label,
        "cutoff": cutoff.isoformat(),
        "end_exclusive": end.isoformat() if end is not None else None,
        "training_interactions": int(len(train)),
        "training_users": int(train["user_idx"].nunique()),
        "training_items": int(train["item_idx"].nunique()),
        "evaluation_users": int(len(queries)),
        "group_thresholds": thresholds,
        "group_counts": {
            group: int((groups == group).sum()) for group in ["head", "mid", "tail", "strict_cold"]
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "elapsed_seconds": perf_counter() - started,
    }
    return result, pd.concat(grouped_frames, ignore_index=True)


def run(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    interactions = pd.read_parquet(config["data"]["interactions_path"])
    items = pd.read_parquet(config["data"]["content_items_path"])
    temporal = config["global_temporal"]
    validation_cutoff = pd.Timestamp(temporal["validation_cutoff"])
    final_cutoff = pd.Timestamp(temporal["final_cutoff"])
    validation, validation_groups = _run_period(
        interactions,
        items,
        cutoff=validation_cutoff,
        end=final_cutoff,
        config=config,
        label="global_validation",
    )
    final, final_groups = _run_period(
        interactions,
        items,
        cutoff=final_cutoff,
        end=None,
        config=config,
        label="global_final_holdout",
    )
    output = Path(config["outputs"]["global_temporal_results"])
    pd.concat(
        [
            validation_groups.assign(period="global_validation"),
            final_groups.assign(period="global_final_holdout"),
        ],
        ignore_index=True,
    ).to_csv(output.with_name("global_temporal_group_metrics.csv"), index=False)
    payload = {
        "experiment_id": "phase6_global_temporal_content_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": temporal,
        "scope": (
            "Robustness check for content, Popularity, and ItemCF. The old Two-Tower checkpoint "
            "is intentionally excluded because it was trained under a different split."
        ),
        "metadata_snapshot_limit": (
            "Static content metadata availability time is not independently verified."
        ),
        "data_sha256": sha256_file(config["data"]["interactions_path"]),
        "periods": {"validation": validation, "final_holdout": final},
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6_content.yaml")
    args = parser.parse_args()
    payload = run(args.config)
    for period, result in payload["periods"].items():
        content = result["metrics"]["content"]["recall@100"]
        hybrid = result["metrics"]["hybrid_content_collaborative"]["recall@100"]
        print(
            f"{period}: users={result['evaluation_users']} content R@100={content:.4f} "
            f"hybrid R@100={hybrid:.4f} elapsed={result['elapsed_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()

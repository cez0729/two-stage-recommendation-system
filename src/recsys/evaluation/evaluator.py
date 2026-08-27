"""Model-agnostic full-catalog evaluation for chronological recommendation splits."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import pandas as pd

from recsys.evaluation.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    ndcg_at_k,
    percentile_latency,
    recall_at_k,
    reciprocal_rank_at_k,
    target_rank,
)
from recsys.retrieval.candidate_filter import Candidate
from recsys.utils.io import sha256_file, write_json

LOGGER = logging.getLogger(__name__)


class Recommender(Protocol):
    """Minimum interface required by the shared evaluator."""

    catalog: set[int]
    popularity: Any

    def recommend(self, history: list[int], k: int) -> list[Candidate]: ...


def _query_rows(interactions: pd.DataFrame, split: str) -> list[tuple[int, list[int], int]]:
    history_splits = {
        "validation": {"retrieval_train", "rank_train"},
        "test": {"retrieval_train", "rank_train", "validation"},
    }
    if split not in history_splits:
        raise ValueError(f"Unsupported evaluation split: {split}")

    ordered = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
    queries: list[tuple[int, list[int], int]] = []
    for user_idx, group in ordered.groupby("user_idx", sort=True):
        targets = group[group["split"] == split]
        if len(targets) != 1:
            raise ValueError(f"User {user_idx} has {len(targets)} targets in {split}")
        history = group[group["split"].isin(history_splits[split])]["item_idx"].astype(int).tolist()
        queries.append((int(user_idx), history, int(targets.iloc[0]["item_idx"])))
    return queries


def _popularity_counts(model: Recommender) -> dict[int, int]:
    if hasattr(model, "item_counts"):
        return model.item_counts
    return model.popularity.item_counts


def evaluate_split(
    model: Recommender,
    interactions: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    ks: list[int],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one model on every query against its complete training catalog."""
    queries = _query_rows(interactions, split)
    maximum_k = max(ks)
    popularity_counts = _popularity_counts(model)
    rows: list[dict[str, Any]] = []
    recommendation_lists: list[list[int]] = []
    source_counts: dict[str, int] = {}

    for user_idx, history, target in queries:
        started = perf_counter()
        candidates = model.recommend(history, maximum_k)
        latency_ms = (perf_counter() - started) * 1000.0
        ranking = [candidate.item_idx for candidate in candidates]
        recommendation_lists.append(ranking[:10])
        for candidate in candidates:
            source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        row: dict[str, Any] = {
            "model": model_name,
            "split": split,
            "user_idx": user_idx,
            "target_item_idx": target,
            "target_in_training_catalog": target in model.catalog,
            "history_length": len(history),
            "recommendation_count": len(ranking),
            "target_rank": target_rank(ranking, target, maximum_k),
            "latency_ms": latency_ms,
            "recommendations": ranking,
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranking, target, k)
            row[f"mrr@{k}"] = reciprocal_rank_at_k(ranking, target, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranking, target, k)
        rows.append(row)

    per_user = pd.DataFrame(rows)
    metrics: dict[str, float] = {}
    confidence_intervals: dict[str, dict[str, float]] = {}
    for k in ks:
        for metric_name in ["recall", "mrr", "ndcg"]:
            column = f"{metric_name}@{k}"
            values = per_user[column].astype(float).tolist()
            metrics[column] = float(np.mean(values))
            if metric_name == "recall" or k == 10:
                low, high = bootstrap_mean_ci(
                    values, samples=bootstrap_samples, confidence=0.95, seed=seed
                )
                confidence_intervals[column] = {"low": low, "high": high}

        available = per_user[per_user["target_in_training_catalog"]]
        metrics[f"recall@{k}_given_available"] = (
            float(available[f"recall@{k}"].mean()) if len(available) else 0.0
        )

    metrics["coverage@10"] = catalog_coverage(recommendation_lists, model.catalog)
    latencies = per_user["latency_ms"].astype(float).tolist()
    metrics["p50_latency_ms"] = percentile_latency(latencies, 50)
    metrics["p95_latency_ms"] = percentile_latency(latencies, 95)

    recommended_items = [item for ranking in recommendation_lists for item in ranking]
    recommended_popularity = [popularity_counts.get(item, 0) for item in recommended_items]
    target_popularity = [popularity_counts.get(target, 0) for _, _, target in queries]
    metrics["recommended_mean_popularity"] = float(np.mean(recommended_popularity))
    metrics["target_mean_popularity"] = float(np.mean(target_popularity))
    metrics["popularity_bias_ratio"] = (
        metrics["recommended_mean_popularity"] / metrics["target_mean_popularity"]
        if metrics["target_mean_popularity"]
        else 0.0
    )

    result = {
        "model": model_name,
        "split": split,
        "evaluation_mode": "full_catalog",
        "num_users": len(per_user),
        "catalog_size": len(model.catalog),
        "target_catalog_availability": float(per_user["target_in_training_catalog"].mean()),
        "cold_target_rate": float(1.0 - per_user["target_in_training_catalog"].mean()),
        "users_with_fewer_than_max_k": int((per_user["recommendation_count"] < maximum_k).sum()),
        "source_item_counts": source_counts,
        "metrics": metrics,
        "confidence_intervals_95": confidence_intervals,
    }
    return result, per_user


def run_baseline_evaluation(model: Recommender, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured splits and persist JSON plus per-user evidence."""
    evaluation = config["evaluation"]
    interactions = pd.read_parquet(evaluation["data_path"])
    model_name = str(config["model"]["name"])
    results_dir = Path(evaluation["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    split_results: dict[str, Any] = {}

    for split in evaluation["splits"]:
        result, per_user = evaluate_split(
            model,
            interactions,
            model_name=model_name,
            split=str(split),
            ks=[int(k) for k in evaluation["ks"]],
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["seed"]),
        )
        per_user.to_parquet(results_dir / f"{model_name}_{split}_per_user.parquet", index=False)
        split_results[str(split)] = result

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "run_id": f"{model_name}_baseline",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": "unavailable_not_a_git_repository",
        "config_sha256": config_hash,
        "data_manifest_sha256": sha256_file(evaluation["data_manifest_path"]),
        "splits": split_results,
    }
    output = results_dir / f"{model_name}_metrics.json"
    write_json(output, payload)
    LOGGER.info("Wrote %s evaluation: %s", model_name, output)
    return payload

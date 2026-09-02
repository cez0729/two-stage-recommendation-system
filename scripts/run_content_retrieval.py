"""Evaluate content retrieval and fixed-budget multi-channel complementarity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import binomtest

from recsys.config import load_yaml
from recsys.evaluation.compare_baselines import paired_bootstrap_difference
from recsys.retrieval.content import TfidfContentRecommender
from recsys.retrieval.dataset import build_evaluation_queries
from recsys.retrieval.itemcf import ItemCFRecommender
from recsys.retrieval.popularity import PopularityRecommender
from recsys.retrieval.train import retrieve_topk
from recsys.retrieval.two_tower import TwoTowerModel
from recsys.utils.io import sha256_file, write_json


def _histories(queries: Any) -> list[list[int]]:
    histories = queries.histories.numpy()
    lengths = queries.history_lengths.numpy()
    return [
        (histories[row, : int(length)] - 1).astype(int).tolist()
        for row, length in enumerate(lengths)
    ]


def _rrf(rankings: list[list[list[int]]], k: int, constant: int) -> list[list[int]]:
    output = []
    for user_rows in zip(*rankings, strict=True):
        scores: dict[int, float] = {}
        for ranking in user_rows:
            for rank, item in enumerate(ranking, start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
        ordered = sorted(scores, key=lambda item: (-scores[item], item))
        output.append(ordered[:k])
    return output


def _target_groups(
    targets: np.ndarray, counts: dict[int, int]
) -> tuple[np.ndarray, dict[str, int]]:
    nonzero = np.asarray(list(counts.values()), dtype=np.int32)
    tail_max = int(np.quantile(nonzero, 0.33, method="higher"))
    mid_max = int(np.quantile(nonzero, 0.67, method="higher"))
    groups = []
    for target in targets:
        count = counts.get(int(target), 0)
        if count == 0:
            groups.append("strict_cold")
        elif count <= tail_max:
            groups.append("tail")
        elif count <= mid_max:
            groups.append("mid")
        else:
            groups.append("head")
    return np.asarray(groups), {"tail_max_count": tail_max, "mid_max_count": mid_max}


def _hits(rankings: list[list[int]], targets: np.ndarray, k: int) -> np.ndarray:
    values = [
        int(int(target) in ranking[:k])
        for ranking, target in zip(rankings, targets, strict=True)
    ]
    return np.asarray(
        values,
        dtype=np.float64,
    )


def _ranks(rankings: list[list[int]], targets: np.ndarray) -> np.ndarray:
    values = []
    for ranking, target in zip(rankings, targets, strict=True):
        try:
            values.append(ranking.index(int(target)) + 1)
        except ValueError:
            values.append(0)
    return np.asarray(values, dtype=np.int32)


def _mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(((left == 1) & (right == 0)).sum())
    right_only = int(((left == 0) & (right == 1)).sum())
    discordant = left_only + right_only
    p_value = (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "left_only_hits": left_only,
        "right_only_hits": right_only,
        "discordant_users": discordant,
        "exact_p_value": p_value,
    }


def _summarize(
    rankings: list[list[int]],
    targets: np.ndarray,
    groups: np.ndarray,
    counts: dict[int, int],
    catalog_size: int,
    *,
    report_k: int,
    candidate_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ranks = _ranks(rankings, targets)
    hit_report = ((ranks > 0) & (ranks <= report_k)).astype(float)
    hit_candidate = ((ranks > 0) & (ranks <= candidate_k)).astype(float)
    rows = []
    for group in ["head", "mid", "tail", "strict_cold"]:
        mask = groups == group
        rows.append(
            {
                "group": group,
                "users": int(mask.sum()),
                f"recall@{report_k}": float(hit_report[mask].mean()) if mask.any() else 0.0,
                f"candidate_recall@{candidate_k}": (
                    float(hit_candidate[mask].mean()) if mask.any() else 0.0
                ),
            }
        )
    recommended = [item for ranking in rankings for item in ranking[:report_k]]
    target_popularity = np.asarray([counts.get(int(target), 0) for target in targets])
    recommended_popularity = np.asarray([counts.get(int(item), 0) for item in recommended])
    metrics = {
        f"recall@{report_k}": float(hit_report.mean()),
        f"candidate_recall@{candidate_k}": float(hit_candidate.mean()),
        f"coverage@{report_k}": float(len(set(recommended)) / catalog_size),
        "mean_recommendation_count": float(np.mean([len(ranking) for ranking in rankings])),
        "min_recommendation_count": int(min(map(len, rankings))),
        "max_recommendation_count": int(max(map(len, rankings))),
        "recommended_mean_training_popularity": float(recommended_popularity.mean()),
        "target_mean_training_popularity": float(target_popularity.mean()),
        "popularity_bias_ratio": (
            float(recommended_popularity.mean() / target_popularity.mean())
            if target_popularity.mean()
            else 0.0
        ),
    }
    return metrics, pd.DataFrame(rows)


def _comparison(
    left_rankings: list[list[int]],
    right_rankings: list[list[int]],
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    k: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = _hits(left_rankings, targets, k)[mask]
    right = _hits(right_rankings, targets, k)[mask]
    return {
        "users": int(mask.sum()),
        "left_recall": float(left.mean()) if len(left) else 0.0,
        "right_recall": float(right.mean()) if len(right) else 0.0,
        "left_minus_right": paired_bootstrap_difference(
            left - right, samples=samples, seed=seed
        ),
        "mcnemar": _mcnemar(left, right),
    }


def _collaborative_rankings(
    interactions: pd.DataFrame,
    histories: list[list[int]],
    split: str,
    k: int,
    user_limit: int | None,
) -> dict[str, list[list[int]]]:
    train = interactions[interactions["split"] == "retrieval_train"]
    popularity = PopularityRecommender().fit(train)
    itemcf = ItemCFRecommender(max_history=50, top_neighbors=200, recency_decay=0.9).fit(train)
    popularity_rows = [
        [candidate.item_idx for candidate in popularity.recommend(history, k)]
        for history in histories
    ]
    itemcf_rows = [
        [candidate.item_idx for candidate in itemcf.recommend(history, k)] for history in histories
    ]

    checkpoint = torch.load(
        "models/video_games_2018/two_tower_v3_logq.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = TwoTowerModel(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    catalog = np.sort(train["item_idx"].astype(int).unique())
    queries = build_evaluation_queries(
        interactions,
        split=split,
        training_catalog=set(map(int, catalog)),
        max_history=int(checkpoint["model_args"]["max_history"]),
    )
    if user_limit is not None:
        queries.user_ids = queries.user_ids[:user_limit]
        queries.histories = queries.histories[:user_limit]
        queries.history_lengths = queries.history_lengths[:user_limit]
        queries.target_items = queries.target_items[:user_limit]
        queries.seen_histories = queries.seen_histories[:user_limit]
    two_tower, _, _ = retrieve_topk(
        model,
        queries,
        catalog,
        k=k,
        batch_size=256,
        device=torch.device("cpu"),
    )
    return {
        "popularity": popularity_rows,
        "itemcf": itemcf_rows,
        "two_tower": two_tower.astype(int).tolist(),
    }


def run(config_path: str, splits: list[str], user_limit: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    data = config["data"]
    output = config["outputs"]
    evaluation = config["evaluation"]
    content_config = config["content"]
    interactions = pd.read_parquet(data["interactions_path"])
    items = pd.read_parquet(data["content_items_path"])
    train = interactions[interactions["split"] == "retrieval_train"]
    train_catalog = set(train["item_idx"].astype(int))
    counts = train["item_idx"].astype(int).value_counts().to_dict()
    content = TfidfContentRecommender(
        max_features=int(content_config["max_features"]),
        min_df=int(content_config["min_df"]),
        ngram_max=int(content_config["ngram_max"]),
        max_history=int(content_config["max_history"]),
    ).fit(items, train_catalog)
    joblib.dump(content.vectorizer, output["vectorizer_path"])
    sparse.save_npz(output["item_matrix_path"], content.item_matrix)
    np.save(output["item_ids_path"], content.item_ids)

    candidate_k = int(evaluation["retrieval_k"])
    report_k = int(evaluation["report_k"])
    rrf_constant = int(evaluation["rrf_constant"])
    samples = int(evaluation["bootstrap_samples"])
    seed = int(config["seed"])
    results_dir = Path(output["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "content_experiment.json"
    split_results = {}
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        split_results.update(
            {
                split: result
                for split, result in existing.get("splits", {}).items()
                if split not in splits
            }
        )

    for split in splits:
        started = perf_counter()
        queries = build_evaluation_queries(
            interactions,
            split=split,
            training_catalog=content.catalog,
            max_history=int(content_config["max_history"]),
        )
        histories = _histories(queries)
        targets = queries.target_items.numpy()
        if user_limit is not None:
            histories = histories[:user_limit]
            targets = targets[:user_limit]
        content_rows = content.recommend_batch(
            histories,
            candidate_k,
            batch_size=int(content_config["batch_size"]),
        )
        rankings = _collaborative_rankings(
            interactions, histories, split, candidate_k, user_limit
        )
        rankings["content"] = content_rows
        combinations_by_name = {
            "popularity_itemcf": ["popularity", "itemcf"],
            "popularity_two_tower": ["popularity", "two_tower"],
            "itemcf_two_tower": ["itemcf", "two_tower"],
            "three_collaborative": ["popularity", "itemcf", "two_tower"],
            "hybrid_rrf_two_tower_content": ["two_tower", "content"],
            "all_four_channels": ["popularity", "itemcf", "two_tower", "content"],
        }
        for name, members in combinations_by_name.items():
            rankings[name] = _rrf(
                [rankings[member] for member in members], candidate_k, rrf_constant
            )

        groups, thresholds = _target_groups(targets, counts)
        model_metrics = {}
        grouped_frames = []
        per_user_frames = []
        for name, model_rankings in rankings.items():
            metrics, grouped = _summarize(
                model_rankings,
                targets,
                groups,
                counts,
                len(content.catalog),
                report_k=report_k,
                candidate_k=candidate_k,
            )
            model_metrics[name] = metrics
            grouped.insert(0, "model", name)
            grouped_frames.append(grouped)
            ranks = _ranks(model_rankings, targets)
            per_user_frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "user_idx": queries.user_ids.numpy()[: len(targets)],
                        "target_item_idx": targets,
                        "group": groups,
                        "target_rank": np.where(ranks > 0, ranks, np.nan),
                        f"recall@{report_k}": ((ranks > 0) & (ranks <= report_k)).astype(int),
                        f"candidate_recall@{candidate_k}": (
                            (ranks > 0) & (ranks <= candidate_k)
                        ).astype(int),
                        "recommendations": model_rankings,
                    }
                )
            )

        masks = {
            "overall": np.ones(len(targets), dtype=bool),
            "tail": groups == "tail",
            "strict_cold": groups == "strict_cold",
        }
        primary_comparisons = {
            "content_minus_two_tower_strict_cold_r100": _comparison(
                rankings["content"],
                rankings["two_tower"],
                targets,
                masks["strict_cold"],
                k=report_k,
                samples=samples,
                seed=seed,
            ),
            "hybrid_minus_two_tower_tail_r100": _comparison(
                rankings["hybrid_rrf_two_tower_content"],
                rankings["two_tower"],
                targets,
                masks["tail"],
                k=report_k,
                samples=samples,
                seed=seed,
            ),
            "all_four_minus_three_collaborative_overall_cr200": _comparison(
                rankings["all_four_channels"],
                rankings["three_collaborative"],
                targets,
                masks["overall"],
                k=candidate_k,
                samples=samples,
                seed=seed,
            ),
        }
        pd.concat(grouped_frames, ignore_index=True).to_csv(
            results_dir / f"{split}_group_metrics.csv", index=False
        )
        pd.concat(per_user_frames, ignore_index=True).to_parquet(
            results_dir / f"{split}_per_user.parquet", index=False
        )
        split_results[split] = {
            "users": int(len(targets)),
            "group_thresholds": thresholds,
            "group_counts": {
                group: int((groups == group).sum())
                for group in ["head", "mid", "tail", "strict_cold"]
            },
            "metrics": model_metrics,
            "primary_comparisons": primary_comparisons,
            "elapsed_seconds": perf_counter() - started,
        }

    payload = {
        "experiment_id": "phase6_content_tfidf_fixed_budget_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_disclosure": (
            "The content direction was motivated by a previously observed test failure. "
            "Hyperparameters were fixed before this run; validation is reported first, and "
            "test is treated as confirmatory rather than an untouched discovery set."
        ),
        "model_scope": (
            "TF-IDF content-only baseline plus score-level RRF hybrid; this is not a trained "
            "Hybrid Item Tower."
        ),
        "candidate_budget": candidate_k,
        "content_vocabulary_size": len(content.vectorizer.vocabulary_),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "data_sha256": sha256_file(data["interactions_path"]),
        "metadata_sha256": sha256_file(data["raw_metadata_path"]),
        "splits": split_results,
    }
    write_json(result_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/content_retrieval.yaml")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()
    payload = run(args.config, args.splits, args.max_users)
    for split, result in payload["splits"].items():
        content = result["metrics"]["content"]
        hybrid = result["metrics"]["hybrid_rrf_two_tower_content"]
        print(
            f"{split}: content R@100={content['recall@100']:.4f}; "
            f"hybrid R@100={hybrid['recall@100']:.4f}; "
            f"elapsed={result['elapsed_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()

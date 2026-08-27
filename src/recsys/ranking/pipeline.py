"""Generate leakage-safe multi-channel candidates and train LambdaRank."""

import argparse
import json
from dataclasses import dataclass
from math import log1p
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRanker

from recsys.config import load_yaml
from recsys.evaluation.metrics import bootstrap_mean_ci
from recsys.retrieval.dataset import EvaluationQueries, build_evaluation_queries
from recsys.retrieval.faiss_index import FaissIndexFlatIP
from recsys.retrieval.itemcf import ItemCFRecommender
from recsys.retrieval.popularity import PopularityRecommender
from recsys.retrieval.two_tower import TwoTowerModel
from recsys.utils.io import write_json
from recsys.utils.seed import seed_everything

FEATURE_COLUMNS = [
    "two_tower_score",
    "two_tower_rank",
    "itemcf_score",
    "itemcf_rank",
    "popularity_score",
    "popularity_rank",
    "source_count",
    "rrf_score",
    "best_source_rank",
    "user_history_length",
    "user_mean_rating",
    "user_rating_std",
    "days_since_last_action",
    "item_popularity_log",
    "item_mean_rating",
    "item_rating_std",
]


@dataclass
class CandidateDataset:
    frame: pd.DataFrame
    group_sizes: list[int]
    total_queries: int
    retrieved_queries: int


def _reciprocal_rank(rank: int, constant: int) -> float:
    return 1.0 / (constant + rank)


def fuse_candidates(
    two_tower: list[tuple[int, float]],
    itemcf: list[tuple[int, float]],
    popularity: list[tuple[int, float]],
    *,
    pool_size: int,
    rrf_constant: int,
) -> list[dict[str, float | int]]:
    """Fuse channel rankings with deterministic reciprocal-rank fusion."""
    records: dict[int, dict[str, float | int]] = {}
    for source, candidates in (
        ("two_tower", two_tower),
        ("itemcf", itemcf),
        ("popularity", popularity),
    ):
        for rank, (item_idx, score) in enumerate(candidates, start=1):
            record = records.setdefault(int(item_idx), {"item_idx": int(item_idx)})
            record[f"{source}_score"] = float(score)
            record[f"{source}_rank"] = rank

    for record in records.values():
        ranks = [
            int(record[f"{source}_rank"])
            for source in ("two_tower", "itemcf", "popularity")
            if f"{source}_rank" in record
        ]
        record["source_count"] = len(ranks)
        record["rrf_score"] = sum(_reciprocal_rank(rank, rrf_constant) for rank in ranks)
        record["best_source_rank"] = min(ranks)
    return sorted(
        records.values(),
        key=lambda row: (-float(row["rrf_score"]), int(row["item_idx"])),
    )[:pool_size]


class FeaturePipeline:
    """Shared offline feature builder for rank-train and validation cutoffs."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.interactions = pd.read_parquet(config["data"]["interactions_path"])
        self.retrieval_train = self.interactions[
            self.interactions["split"] == "retrieval_train"
        ]
        self.catalog = set(self.retrieval_train["item_idx"].astype(int).unique())
        self.popularity = PopularityRecommender().fit(self.retrieval_train)
        itemcf_config = config["retrieval"]["itemcf"]
        self.itemcf = ItemCFRecommender(
            max_history=int(itemcf_config["max_history"]),
            top_neighbors=int(itemcf_config["top_neighbors"]),
            recency_decay=float(itemcf_config["recency_decay"]),
        ).fit(self.retrieval_train)
        self.faiss = FaissIndexFlatIP.load(
            config["retrieval"]["faiss_index_path"],
            config["retrieval"]["faiss_item_ids_path"],
        )
        checkpoint = torch.load(
            config["retrieval"]["two_tower_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        self.two_tower = TwoTowerModel(**checkpoint["model_args"])
        self.two_tower.load_state_dict(checkpoint["model_state_dict"])
        self.two_tower.eval()
        self.max_history = int(checkpoint["model_args"]["max_history"])
        counts = self.popularity.item_counts
        popularity_order = [candidate.item_idx for candidate in self.popularity.ranking]
        self.popularity_rank = {item: rank for rank, item in enumerate(popularity_order, 1)}
        self.popularity_score = {item: log1p(count) for item, count in counts.items()}
        self.item_stats = self.retrieval_train.groupby("item_idx").agg(
            item_mean_rating=("rating", "mean"),
            item_rating_std=("rating", lambda values: values.std(ddof=0)),
            item_first_interaction=("timestamp", "min"),
            item_last_interaction=("timestamp", "max"),
        )

    @staticmethod
    def _history_splits(split: str) -> set[str]:
        return {
            "rank_train": {"retrieval_train"},
            "validation": {"retrieval_train", "rank_train"},
            "test": {"retrieval_train", "rank_train", "validation"},
        }[split]

    def _queries_and_vectors(self, split: str) -> tuple[EvaluationQueries, np.ndarray]:
        queries = build_evaluation_queries(
            self.interactions,
            split=split,
            training_catalog=self.catalog,
            max_history=self.max_history,
        )
        with torch.inference_mode():
            vectors = self.two_tower.encode_users(
                queries.user_ids,
                queries.histories,
                queries.history_lengths,
            ).numpy()
        return queries, vectors

    def _ordered_histories(self, split: str) -> dict[int, pd.DataFrame]:
        allowed = self._history_splits(split)
        history = self.interactions[self.interactions["split"].isin(allowed)].sort_values(
            ["user_idx", "timestamp", "item_idx"], kind="stable"
        )
        return {int(user): group for user, group in history.groupby("user_idx", sort=True)}

    def build(self, split: str) -> CandidateDataset:
        queries, user_vectors = self._queries_and_vectors(split)
        retrieval_config = self.config["retrieval"]
        two_tower_k = int(retrieval_config["two_tower_candidates"])
        itemcf_k = int(retrieval_config["itemcf_candidates"])
        popularity_k = int(retrieval_config["popularity_candidates"])
        pool_size = int(retrieval_config["pool_size"])
        two_tower_ids, two_tower_scores = self.faiss.search(
            user_vectors,
            k=two_tower_k,
            seen_item_ids=queries.seen_histories,
        )
        histories = self._ordered_histories(split)
        target_rows = self.interactions[self.interactions["split"] == split].set_index("user_idx")
        collected: list[pd.DataFrame] = []
        group_sizes: list[int] = []
        retrieved_queries = 0

        for row, user_idx_tensor in enumerate(queries.user_ids):
            user_idx = int(user_idx_tensor)
            history_frame = histories[user_idx]
            history = history_frame["item_idx"].astype(int).tolist()
            itemcf_candidates = [
                (candidate.item_idx, candidate.score)
                for candidate in self.itemcf.recommend(history, k=itemcf_k)
                if candidate.source == "itemcf"
            ]
            popularity_candidates = [
                (candidate.item_idx, candidate.score)
                for candidate in self.popularity.recommend(history, k=popularity_k)
            ]
            fused = fuse_candidates(
                list(zip(two_tower_ids[row].tolist(), two_tower_scores[row].tolist(), strict=True)),
                itemcf_candidates,
                popularity_candidates,
                pool_size=pool_size,
                rrf_constant=int(retrieval_config["rrf_constant"]),
            )
            if len(fused) < pool_size:
                backfill = self.popularity.recommend(history, k=pool_size)
                emitted = {int(candidate["item_idx"]) for candidate in fused}
                for candidate in backfill:
                    if candidate.item_idx in emitted:
                        continue
                    fused.append(
                        {
                            "item_idx": candidate.item_idx,
                            "popularity_score": candidate.score,
                            "popularity_rank": self.popularity_rank[candidate.item_idx],
                            "source_count": 1,
                            "rrf_score": _reciprocal_rank(
                                self.popularity_rank[candidate.item_idx],
                                int(retrieval_config["rrf_constant"]),
                            ),
                            "best_source_rank": self.popularity_rank[candidate.item_idx],
                        }
                    )
                    emitted.add(candidate.item_idx)
                    if len(fused) == pool_size:
                        break

            target_item = int(target_rows.loc[user_idx, "item_idx"])
            candidate_ids = np.asarray([int(candidate["item_idx"]) for candidate in fused])
            target_positions = np.flatnonzero(candidate_ids == target_item)
            if not len(target_positions):
                continue
            retrieved_queries += 1
            candidate_frame = pd.DataFrame(fused)
            candidate_frame["query_id"] = f"{split}:{user_idx}"
            candidate_frame["user_idx"] = user_idx
            candidate_frame["label"] = (candidate_ids == target_item).astype(np.int8)
            candidate_frame["retrieval_rank"] = np.arange(1, len(fused) + 1)
            defaults = {
                "two_tower_score": -2.0,
                "two_tower_rank": two_tower_k + 1,
                "itemcf_score": 0.0,
                "itemcf_rank": itemcf_k + 1,
            }
            for column, default in defaults.items():
                if column not in candidate_frame:
                    candidate_frame[column] = default
                else:
                    candidate_frame[column] = candidate_frame[column].fillna(default)
            candidate_frame["popularity_score"] = candidate_frame["item_idx"].map(
                self.popularity_score
            )
            candidate_frame["popularity_rank"] = candidate_frame["item_idx"].map(
                self.popularity_rank
            )
            candidate_frame["user_history_length"] = len(history_frame)
            candidate_frame["user_mean_rating"] = float(history_frame["rating"].mean())
            candidate_frame["user_rating_std"] = float(history_frame["rating"].std(ddof=0))
            target_time = pd.Timestamp(target_rows.loc[user_idx, "timestamp"])
            candidate_frame["days_since_last_action"] = max(
                0.0,
                (target_time - history_frame["timestamp"].max()).total_seconds() / 86400,
            )
            candidate_frame["item_popularity_log"] = candidate_frame["popularity_score"]
            item_stats = self.item_stats.reindex(candidate_ids)
            candidate_frame["item_mean_rating"] = item_stats["item_mean_rating"].to_numpy()
            candidate_frame["item_rating_std"] = (
                item_stats["item_rating_std"].fillna(0.0).to_numpy()
            )
            candidate_frame["item_age_days"] = np.maximum(
                0.0,
                (target_time - item_stats["item_first_interaction"])
                .dt.total_seconds()
                .to_numpy()
                / 86400,
            )
            candidate_frame["item_days_since_last_interaction"] = np.maximum(
                0.0,
                (target_time - item_stats["item_last_interaction"])
                .dt.total_seconds()
                .to_numpy()
                / 86400,
            )
            collected.append(candidate_frame)
            group_sizes.append(len(candidate_frame))
            if (row + 1) % 2000 == 0:
                print(
                    f"{split}: processed={row + 1}/{len(queries.user_ids)} "
                    f"retrieved={retrieved_queries}",
                    flush=True,
                )

        frame = pd.concat(collected, ignore_index=True)
        if frame[FEATURE_COLUMNS].isna().any().any():
            raise ValueError(f"{split} feature frame contains missing values")
        return CandidateDataset(
            frame=frame,
            group_sizes=group_sizes,
            total_queries=len(queries.user_ids),
            retrieved_queries=retrieved_queries,
        )


def _ranking_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray | None,
    *,
    total_queries: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    ranks: list[int] = []
    rows: list[dict[str, Any]] = []
    prediction_series = None if predictions is None else pd.Series(predictions, index=frame.index)
    for query_id, group in frame.groupby("query_id", sort=False):
        if prediction_series is None:
            ordered = group.sort_values(["retrieval_rank", "item_idx"], kind="stable")
        else:
            ordered = group.assign(_prediction=prediction_series.loc[group.index]).sort_values(
                ["_prediction", "item_idx"], ascending=[False, True], kind="stable"
            )
        rank = int(np.flatnonzero(ordered["label"].to_numpy() == 1)[0]) + 1
        ranks.append(rank)
        rows.append({"query_id": query_id, "target_rank": rank})
    padded = np.asarray(ranks + [0] * (total_queries - len(ranks)), dtype=np.int32)
    metrics: dict[str, float] = {"candidate_recall@200": len(ranks) / total_queries}
    for k in (5, 10, 20):
        hit = (padded > 0) & (padded <= k)
        metrics[f"recall@{k}"] = float(hit.mean())
        metrics[f"mrr@{k}"] = float(np.where(hit, 1.0 / np.maximum(padded, 1), 0).mean())
        metrics[f"ndcg@{k}"] = float(
            np.where(hit, 1.0 / np.log2(np.maximum(padded, 1) + 1), 0).mean()
        )
    return metrics, pd.DataFrame(rows)


def train_ranker(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    output = config["outputs"]
    Path(output["feature_dir"]).mkdir(parents=True, exist_ok=True)
    training_path = Path(output["feature_dir"]) / "rank_train.parquet"
    validation_path = Path(output["feature_dir"]) / "validation.parquet"
    reuse_features = bool(config["data"].get("reuse_existing_features", False))
    if reuse_features and training_path.exists() and validation_path.exists():
        training_frame = pd.read_parquet(training_path)
        validation_frame = pd.read_parquet(validation_path)
        split_counts = (
            pd.read_parquet(config["data"]["interactions_path"], columns=["split"])["split"]
            .value_counts()
            .to_dict()
        )
        training = CandidateDataset(
            frame=training_frame,
            group_sizes=training_frame.groupby("query_id", sort=False).size().tolist(),
            total_queries=int(split_counts["rank_train"]),
            retrieved_queries=int(training_frame["query_id"].nunique()),
        )
        validation = CandidateDataset(
            frame=validation_frame,
            group_sizes=validation_frame.groupby("query_id", sort=False).size().tolist(),
            total_queries=int(split_counts["validation"]),
            retrieved_queries=int(validation_frame["query_id"].nunique()),
        )
    else:
        pipeline = FeaturePipeline(config)
        training = pipeline.build("rank_train")
        validation = pipeline.build("validation")
        training.frame.to_parquet(training_path, index=False)
        validation.frame.to_parquet(validation_path, index=False)

    model_config = config["model"]
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        num_leaves=int(model_config["num_leaves"]),
        min_child_samples=int(model_config["min_child_samples"]),
        subsample=float(model_config["subsample"]),
        colsample_bytree=float(model_config["colsample_bytree"]),
        reg_lambda=float(model_config["reg_lambda"]),
        random_state=seed,
        n_jobs=int(model_config["n_jobs"]),
        verbosity=-1,
    )
    ranker.fit(
        training.frame[FEATURE_COLUMNS],
        training.frame["label"],
        group=training.group_sizes,
        eval_set=[(validation.frame[FEATURE_COLUMNS], validation.frame["label"])],
        eval_group=[validation.group_sizes],
        eval_at=[10],
        callbacks=[lgb.early_stopping(int(model_config["early_stopping_rounds"]), verbose=False)],
    )
    Path(output["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(output["model_path"])
    validation_predictions = ranker.predict(
        validation.frame[FEATURE_COLUMNS], num_iteration=ranker.best_iteration_
    )
    baseline_metrics, baseline_users = _ranking_metrics(
        validation.frame, None, total_queries=validation.total_queries
    )
    ranker_metrics, ranker_users = _ranking_metrics(
        validation.frame, validation_predictions, total_queries=validation.total_queries
    )
    low, high = bootstrap_mean_ci(
        np.r_[
            (ranker_users["target_rank"].to_numpy() <= 10).astype(float),
            np.zeros(validation.total_queries - len(ranker_users)),
        ],
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": ranker.booster_.feature_importance(importance_type="gain"),
            "split": ranker.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    Path(output["importance_path"]).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output["importance_path"], index=False)
    comparison = baseline_users.merge(
        ranker_users, on="query_id", suffixes=("_retrieval", "_ranker"), validate="one_to_one"
    )
    comparison.to_parquet(output["per_user_path"], index=False)
    reloaded = lgb.Booster(model_file=output["model_path"])
    reload_predictions = reloaded.predict(
        validation.frame[FEATURE_COLUMNS].iloc[:200], num_iteration=ranker.best_iteration_
    )
    payload = {
        "selection_split": "validation",
        "test_evaluated": False,
        "features": FEATURE_COLUMNS,
        "best_iteration": int(ranker.best_iteration_),
        "datasets": {
            "rank_train": {
                "total_queries": training.total_queries,
                "trainable_queries": training.retrieved_queries,
                "candidate_recall@200": training.retrieved_queries / training.total_queries,
                "rows": len(training.frame),
            },
            "validation": {
                "total_queries": validation.total_queries,
                "retrieved_queries": validation.retrieved_queries,
                "candidate_recall@200": validation.retrieved_queries / validation.total_queries,
                "rows_with_retrieved_target": len(validation.frame),
            },
        },
        "retrieval_order_metrics": baseline_metrics,
        "lambdarank_metrics": ranker_metrics,
        "recall@10_confidence_interval_95": {"low": low, "high": high},
        "acceptance": {
            "ndcg@10_not_worse": ranker_metrics["ndcg@10"] >= baseline_metrics["ndcg@10"],
            "ndcg@10_delta": ranker_metrics["ndcg@10"] - baseline_metrics["ndcg@10"],
            "reload_predictions_identical": bool(
                np.allclose(reload_predictions, validation_predictions[:200])
            ),
        },
        "feature_importance_top10": importance.head(10).to_dict("records"),
    }
    write_json(output["metrics_path"], payload)
    write_json(output["feature_schema_path"], {"features": FEATURE_COLUMNS})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ranker.yaml")
    args = parser.parse_args()
    print(json.dumps(train_ranker(args.config), indent=2))


if __name__ == "__main__":
    main()

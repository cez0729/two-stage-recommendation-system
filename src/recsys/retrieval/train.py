"""Train, select, export, and evaluate the two-tower retrieval model."""

import argparse
import hashlib
import json
import logging
import platform
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from recsys.config import load_yaml
from recsys.evaluation.metrics import bootstrap_mean_ci
from recsys.logging import configure_logging
from recsys.retrieval.dataset import (
    EvaluationQueries,
    PrefixInteractionDataset,
    build_evaluation_queries,
)
from recsys.retrieval.two_tower import TwoTowerModel
from recsys.utils.io import sha256_file, write_json
from recsys.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


def _model_arguments(config: dict[str, Any], interactions: pd.DataFrame) -> dict[str, Any]:
    model = config["model"]
    return {
        "num_users": int(interactions["user_idx"].max()) + 1,
        "num_items": int(interactions["item_idx"].max()) + 1,
        "embedding_dim": int(model["embedding_dim"]),
        "hidden_dim": int(model["hidden_dim"]),
        "dropout": float(model["dropout"]),
        "max_history": int(model["history_length"]),
        "temperature": float(model["temperature"]),
    }


def _slice_queries(queries: EvaluationQueries, stop: int) -> EvaluationQueries:
    return replace(
        queries,
        user_ids=queries.user_ids[:stop],
        histories=queries.histories[:stop],
        history_lengths=queries.history_lengths[:stop],
        target_items=queries.target_items[:stop],
        seen_histories=queries.seen_histories[:stop],
    )


@torch.inference_mode()
def retrieve_topk(
    model: TwoTowerModel,
    queries: EvaluationQueries,
    catalog: np.ndarray,
    *,
    k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run exact full-catalog dot-product retrieval with seen-item filtering."""
    model.eval()
    catalog_tensor = torch.tensor(catalog, dtype=torch.long, device=device)
    item_vectors = model.encode_items(catalog_tensor)
    catalog_positions = {int(item): position for position, item in enumerate(catalog)}
    ranking_batches: list[np.ndarray] = []
    latencies_ms: list[float] = []

    for start in range(0, len(queries.user_ids), batch_size):
        stop = min(start + batch_size, len(queries.user_ids))
        started = perf_counter()
        user_vectors = model.encode_users(
            queries.user_ids[start:stop].to(device),
            queries.histories[start:stop].to(device),
            queries.history_lengths[start:stop].to(device),
        )
        scores = user_vectors @ item_vectors.T
        for row, seen in enumerate(queries.seen_histories[start:stop]):
            positions = [catalog_positions[item] for item in seen if item in catalog_positions]
            if positions:
                scores[row, torch.tensor(positions, dtype=torch.long, device=device)] = -torch.inf
        top_positions = torch.topk(scores, k=k, dim=1, sorted=True).indices
        rankings = catalog_tensor[top_positions].cpu().numpy()
        elapsed_per_user = (perf_counter() - started) * 1000.0 / (stop - start)
        ranking_batches.append(rankings)
        latencies_ms.extend([elapsed_per_user] * (stop - start))
    return (
        np.concatenate(ranking_batches, axis=0),
        np.asarray(latencies_ms),
        item_vectors.cpu().numpy(),
    )


def _target_ranks(rankings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matches = rankings == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.zeros(len(targets), dtype=np.int32)
    ranks[found] = matches[found].argmax(axis=1) + 1
    return ranks


def summarize_rankings(
    rankings: np.ndarray,
    queries: EvaluationQueries,
    catalog: np.ndarray,
    latencies_ms: np.ndarray,
    *,
    split: str,
    ks: list[int],
    bootstrap_samples: int,
    seed: int,
    item_counts: dict[int, int],
    single_query_p95_ms: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate metrics, uncertainty, and user-level evidence from exact rankings."""
    targets = queries.target_items.numpy()
    ranks = _target_ranks(rankings, targets)
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    per_user: dict[str, Any] = {
        "user_idx": queries.user_ids.numpy(),
        "target_item_idx": targets,
        "target_rank": np.where(ranks > 0, ranks, np.nan),
        "target_in_training_catalog": np.isin(targets, catalog),
        "history_length": queries.history_lengths.numpy(),
        "latency_ms_batch_amortized": latencies_ms,
        "recommendations": rankings.tolist(),
    }
    for k in ks:
        recall = ((ranks > 0) & (ranks <= k)).astype(float)
        reciprocal = np.where((ranks > 0) & (ranks <= k), 1.0 / np.maximum(ranks, 1), 0.0)
        ndcg = np.where(
            (ranks > 0) & (ranks <= k), 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0
        )
        metrics[f"recall@{k}"] = float(recall.mean())
        metrics[f"mrr@{k}"] = float(reciprocal.mean())
        metrics[f"ndcg@{k}"] = float(ndcg.mean())
        low, high = bootstrap_mean_ci(recall, samples=bootstrap_samples, seed=seed)
        intervals[f"recall@{k}"] = {"low": low, "high": high}
        per_user[f"recall@{k}"] = recall
        per_user[f"mrr@{k}"] = reciprocal
        per_user[f"ndcg@{k}"] = ndcg

    available = np.isin(targets, catalog)
    for k in ks:
        hit = (ranks > 0) & (ranks <= k)
        metrics[f"recall@{k}_given_available"] = float(hit[available].mean())
    metrics["coverage@10"] = float(len(np.unique(rankings[:, :10])) / len(catalog))
    metrics["p50_latency_ms_batch_amortized"] = float(np.percentile(latencies_ms, 50))
    metrics["p95_latency_ms_batch_amortized"] = float(np.percentile(latencies_ms, 95))
    metrics["p95_latency_ms_single_query"] = single_query_p95_ms
    recommended_popularity = [item_counts.get(int(item), 0) for item in rankings[:, :10].flat]
    target_popularity = [item_counts.get(int(item), 0) for item in targets]
    metrics["recommended_mean_popularity"] = float(np.mean(recommended_popularity))
    metrics["target_mean_popularity"] = float(np.mean(target_popularity))
    metrics["popularity_bias_ratio"] = (
        metrics["recommended_mean_popularity"] / metrics["target_mean_popularity"]
    )
    result = {
        "split": split,
        "evaluation_mode": "full_catalog_exact_dot_product",
        "num_users": len(targets),
        "catalog_size": len(catalog),
        "target_catalog_availability": float(available.mean()),
        "cold_target_rate": float(1.0 - available.mean()),
        "metrics": metrics,
        "confidence_intervals_95": intervals,
    }
    return result, pd.DataFrame(per_user)


def _load_baseline_recall(model: str, split: str = "test") -> float:
    path = Path(f"results/video_games_2018/baselines/{model}_metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["splits"][split]["metrics"]["recall@100"])


def run_training(config_path: str | Path) -> dict[str, Any]:
    """Train on retrieval data and optionally evaluate the frozen model on test."""
    config = load_yaml(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(str(config["training"]["device"]))
    interactions = pd.read_parquet(config["data"]["interactions_path"])
    retrieval_train = interactions[interactions["split"] == "retrieval_train"]
    catalog = np.sort(retrieval_train["item_idx"].astype(int).unique())
    catalog_set = set(map(int, catalog))
    max_history = int(config["model"]["history_length"])
    train_dataset = PrefixInteractionDataset(retrieval_train, max_history=max_history)
    validation_queries = build_evaluation_queries(
        interactions,
        split="validation",
        training_catalog=catalog_set,
        max_history=max_history,
    )
    run_test = bool(config["evaluation"].get("run_test", True))
    test_queries = None
    if run_test:
        test_queries = build_evaluation_queries(
            interactions,
            split="test",
            training_catalog=catalog_set,
            max_history=max_history,
        )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        generator=generator,
    )
    model_args = _model_arguments(config, interactions)
    model = TwoTowerModel(**model_args).to(device)
    sampling_correction_weight = float(
        config["training"].get("sampling_correction_weight", 0.0)
    )
    item_counts_array = np.bincount(
        retrieval_train["item_idx"].to_numpy(dtype=np.int64),
        minlength=model_args["num_items"],
    )
    item_sampling_probabilities = torch.tensor(
        item_counts_array / item_counts_array.sum(),
        dtype=torch.float32,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_config = config["training"].get("scheduler", {})
    scheduler = None
    if bool(scheduler_config.get("enabled", False)):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            min_lr=float(scheduler_config.get("min_lr", 1e-5)),
        )
    output = config["outputs"]
    model_path = Path(output["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_two_tower")
    run_dir = Path(output["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, float | int]] = []
    best_recall = -1.0
    best_epoch = 0
    stale_epochs = 0
    maximum_k = max(int(k) for k in config["evaluation"]["ks"])
    evaluation_only = int(config["training"]["max_epochs"]) == 0

    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        started = perf_counter()
        for users, histories, lengths, targets in train_loader:
            users = users.to(device)
            histories = histories.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.in_batch_loss(
                users,
                histories,
                lengths,
                targets,
                item_sampling_probabilities=item_sampling_probabilities,
                sampling_correction_weight=sampling_correction_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(users)
            total_examples += len(users)
        rankings, _, _ = retrieve_topk(
            model,
            validation_queries,
            catalog,
            k=maximum_k,
            batch_size=int(config["evaluation"]["batch_size"]),
            device=device,
        )
        ranks = _target_ranks(rankings, validation_queries.target_items.numpy())
        validation_recall = float(((ranks > 0) & (ranks <= 100)).mean())
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "validation_recall@100": validation_recall,
            "learning_rate": learning_rate,
            "epoch_seconds": perf_counter() - started,
        }
        history.append(epoch_record)
        write_json(run_dir / "epoch_metrics.json", {"epochs": history})
        LOGGER.info(
            "epoch=%d loss=%.5f validation_recall@100=%.5f lr=%.6g seconds=%.1f",
            epoch,
            epoch_record["train_loss"],
            validation_recall,
            learning_rate,
            epoch_record["epoch_seconds"],
        )
        if validation_recall > best_recall + 1e-6:
            best_recall = validation_recall
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": model_args,
                    "catalog": catalog,
                    "best_epoch": best_epoch,
                    "validation_recall@100": best_recall,
                },
                model_path,
            )
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step(validation_recall)
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            break

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if evaluation_only:
        best_epoch = int(checkpoint["best_epoch"])
        best_recall = float(checkpoint["validation_recall@100"])
    model.eval()
    ks = [int(k) for k in config["evaluation"]["ks"]]
    item_counts = {
        int(item): int(count)
        for item, count in retrieval_train["item_idx"].value_counts().items()
    }
    split_results: dict[str, Any] = {}
    per_user_dir = Path(output["per_user_dir"])
    per_user_dir.mkdir(parents=True, exist_ok=True)
    exported_item_vectors: np.ndarray | None = None
    evaluation_splits = [("validation", validation_queries)]
    if test_queries is not None:
        evaluation_splits.append(("test", test_queries))
    for split, queries in evaluation_splits:
        rankings, latencies, item_vectors = retrieve_topk(
            model,
            queries,
            catalog,
            k=maximum_k,
            batch_size=int(config["evaluation"]["batch_size"]),
            device=device,
        )
        _, single_latencies, _ = retrieve_topk(
            model,
            _slice_queries(queries, min(200, len(queries.user_ids))),
            catalog,
            k=maximum_k,
            batch_size=1,
            device=device,
        )
        result, per_user = summarize_rankings(
            rankings,
            queries,
            catalog,
            latencies,
            split=split,
            ks=ks,
            bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=seed,
            item_counts=item_counts,
            single_query_p95_ms=float(np.percentile(single_latencies, 95)),
        )
        per_user.to_parquet(per_user_dir / f"{split}.parquet", index=False)
        split_results[split] = result
        exported_item_vectors = item_vectors

    item_vectors_path = Path(output["item_vectors_path"])
    item_ids_path = Path(output["item_ids_path"])
    item_vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(item_vectors_path, exported_item_vectors)
    np.save(item_ids_path, catalog)
    comparison_split = "test" if run_test else "validation"
    popularity_recall = _load_baseline_recall("popularity", comparison_split)
    itemcf_recall = _load_baseline_recall("itemcf", comparison_split)
    two_tower_recall = float(split_results[comparison_split]["metrics"]["recall@100"])
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_manifest_sha256": sha256_file(config["data"]["data_manifest_path"]),
        "config_sha256": config_hash,
        "device": str(device),
        "evaluation_only": evaluation_only,
        "best_epoch": best_epoch,
        "selected_validation_recall@100": best_recall,
        "training_history": history,
        "splits": split_results,
        "acceptance": {
            "comparison_split": comparison_split,
            "popularity_recall@100": popularity_recall,
            "itemcf_recall@100": itemcf_recall,
            "two_tower_recall@100": two_tower_recall,
            "meets_popularity_floor": two_tower_recall >= popularity_recall,
            "gap_to_itemcf": two_tower_recall - itemcf_recall,
        },
        "artifacts": {
            "checkpoint": model_path.as_posix(),
            "item_vectors": item_vectors_path.as_posix(),
            "item_ids": item_ids_path.as_posix(),
        },
    }
    write_json(output["metrics_path"], payload)
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
        },
    )
    write_json(run_dir / "final_metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/two_tower.yaml")
    args = parser.parse_args()
    configure_logging()
    result = run_training(args.config)
    LOGGER.info("Two-tower acceptance: %s", result["acceptance"])


if __name__ == "__main__":
    main()

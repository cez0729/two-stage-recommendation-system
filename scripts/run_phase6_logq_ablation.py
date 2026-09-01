"""Run and aggregate the pre-registered three-seed raw-vs-logQ ablation."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from recsys.config import load_yaml
from recsys.retrieval.train import run_training
from recsys.utils.io import write_json

SEED_42_RESULTS = {
    "raw": "results/video_games_2018/tuning/two_tower_v2_convergence_clean_metrics.json",
    "logq": "results/video_games_2018/tuning/two_tower_v3_logq_metrics.json",
}
METRICS = ["recall@10", "recall@100", "ndcg@10", "coverage@10", "popularity_bias_ratio"]


def _paths(variant: str, seed: int) -> dict[str, str]:
    stem = f"two_tower_{variant}_seed{seed}"
    return {
        "model_path": f"models/video_games_2018/phase6/{stem}.pt",
        "item_vectors_path": f"artifacts/video_games_2018/phase6/{stem}_item_vectors.npy",
        "item_ids_path": f"artifacts/video_games_2018/phase6/{stem}_item_ids.npy",
        "metrics_path": f"results/video_games_2018/phase6/logq_ablation/{stem}_metrics.json",
        "per_user_dir": f"results/video_games_2018/phase6/logq_ablation/{stem}_per_user",
        "runs_dir": f"runs/video_games_2018/phase6/logq_ablation/{stem}",
    }


def _run_one(variant: str, seed: int, *, force: bool) -> dict[str, Any]:
    if seed == 42 and not force:
        return json.loads(Path(SEED_42_RESULTS[variant]).read_text(encoding="utf-8"))
    base_path = (
        "configs/two_tower_v2_convergence.yaml"
        if variant == "raw"
        else "configs/two_tower_v3_logq.yaml"
    )
    config = copy.deepcopy(load_yaml(base_path))
    config["seed"] = seed
    config["outputs"] = _paths(variant, seed)
    config_path = Path(f"artifacts/video_games_2018/phase6/configs/{variant}_seed{seed}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metrics_path = Path(config["outputs"]["metrics_path"])
    if metrics_path.is_file() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return run_training(config_path)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def run(seeds: list[int], *, force: bool = False) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("At least three pre-registered seeds are required")
    results: dict[str, dict[int, dict[str, Any]]] = {"raw": {}, "logq": {}}
    for seed in seeds:
        for variant in ["raw", "logq"]:
            print(f"Running {variant} seed={seed}", flush=True)
            results[variant][seed] = _run_one(variant, seed, force=force)

    rows_by_variant: dict[str, list[dict[str, Any]]] = {"raw": [], "logq": []}
    table_rows = []
    for variant, seed_results in results.items():
        for seed, payload in seed_results.items():
            metrics = payload["splits"]["validation"]["metrics"]
            row = {metric: float(metrics[metric]) for metric in METRICS}
            rows_by_variant[variant].append(row)
            table_rows.append({"variant": variant, "seed": seed, **row})

    paired_differences = {}
    for metric in METRICS:
        raw = np.asarray(
            [results["raw"][seed]["splits"]["validation"]["metrics"][metric] for seed in seeds]
        )
        logq = np.asarray(
            [results["logq"][seed]["splits"]["validation"]["metrics"][metric] for seed in seeds]
        )
        differences = logq - raw
        paired_differences[metric] = {
            "mean_logq_minus_raw": float(differences.mean()),
            "std_sample": float(differences.std(ddof=1)),
            "all_seeds_positive": bool((differences > 0).all()),
        }

    payload = {
        "experiment_id": "phase6_logq_three_seed_ablation_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "controlled_factors": (
            "Raw and logQ use the same data, architecture, 30 epochs, optimizer, scheduler, "
            "batch size, and evaluation; sampling_correction_weight is the only mechanism change."
        ),
        "seed_42_provenance": (
            "Reuses previously completed controlled runs with matching settings; other seeds "
            "are executed by this Phase 6 orchestrator."
        ),
        "per_seed": table_rows,
        "summary": {variant: _aggregate(rows) for variant, rows in rows_by_variant.items()},
        "paired_seed_differences": paired_differences,
    }
    write_json("results/video_games_2018/phase6/logq_ablation_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = run(args.seeds, force=args.force)
    for variant, summary in payload["summary"].items():
        recall = summary["recall@100"]
        print(f"{variant}: validation R@100={recall['mean']:.4f} +/- {recall['std_sample']:.4f}")


if __name__ == "__main__":
    main()

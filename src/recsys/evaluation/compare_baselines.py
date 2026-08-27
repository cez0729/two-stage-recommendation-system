"""Create paired user-level statistical comparisons between retrieval baselines."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.utils.io import write_json


def paired_bootstrap_difference(
    differences: np.ndarray, *, samples: int = 5000, seed: int = 42
) -> dict[str, float | bool]:
    """Estimate a paired mean-difference interval by resampling users."""
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    batch_size = 128
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def compare_split(
    popularity_path: Path,
    itemcf_path: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare the same users under Popularity and ItemCF."""
    popularity = pd.read_parquet(popularity_path)
    itemcf = pd.read_parquet(itemcf_path)
    paired = popularity.merge(
        itemcf,
        on=["split", "user_idx", "target_item_idx"],
        suffixes=("_popularity", "_itemcf"),
        validate="one_to_one",
    )
    metrics = {}
    for column in ["recall@10", "recall@20", "recall@50", "recall@100", "ndcg@10"]:
        differences = (
            paired[f"{column}_itemcf"].to_numpy(dtype=float)
            - paired[f"{column}_popularity"].to_numpy(dtype=float)
        )
        metrics[column] = paired_bootstrap_difference(
            differences, samples=samples, seed=seed
        )
    return {"num_paired_users": len(paired), "itemcf_minus_popularity": metrics}


def run_comparison(results_dir: Path, samples: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Compare validation and test files and persist a JSON report."""
    comparisons = {}
    for split in ["validation", "test"]:
        comparisons[split] = compare_split(
            results_dir / f"popularity_{split}_per_user.parquet",
            results_dir / f"itemcf_{split}_per_user.parquet",
            samples=samples,
            seed=seed,
        )
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "paired non-parametric bootstrap over users",
        "bootstrap_samples": samples,
        "seed": seed,
        "comparisons": comparisons,
    }
    write_json(results_dir / "baseline_comparison.json", payload)
    summary_rows = []
    for model in ["popularity", "itemcf"]:
        metrics_path = results_dir / f"{model}_metrics.json"
        metrics_payload = pd.read_json(metrics_path, typ="series")
        for split in ["validation", "test"]:
            split_result = metrics_payload["splits"][split]
            metric = split_result["metrics"]
            summary_rows.append(
                {
                    "model": model,
                    "split": split,
                    "num_users": split_result["num_users"],
                    "catalog_size": split_result["catalog_size"],
                    "target_catalog_availability": split_result[
                        "target_catalog_availability"
                    ],
                    "recall@10": metric["recall@10"],
                    "recall@20": metric["recall@20"],
                    "recall@50": metric["recall@50"],
                    "recall@100": metric["recall@100"],
                    "recall@100_given_available": metric["recall@100_given_available"],
                    "ndcg@10": metric["ndcg@10"],
                    "mrr@10": metric["mrr@10"],
                    "coverage@10": metric["coverage@10"],
                    "popularity_bias_ratio": metric["popularity_bias_ratio"],
                    "p95_latency_ms": metric["p95_latency_ms"],
                }
            )
    pd.DataFrame(summary_rows).to_csv(results_dir / "retrieval_comparison.csv", index=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/baselines")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_comparison(Path(args.results_dir), samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()

"""Measure serving latency as the final candidate budget changes."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np

from recsys.serving.pipeline import ServingPipeline


def benchmark(
    config_path: str,
    budgets: list[int],
    users: int,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    pipeline = ServingPipeline.from_yaml(config_path)
    user_ids = list(pipeline.histories)[:users]
    if not user_ids:
        raise ValueError("No known users are available for benchmarking")

    rows: list[dict[str, float | int]] = []
    for budget in budgets:
        pipeline.pool_size = budget
        for index in range(warmup):
            pipeline.recommend(user_ids[index % len(user_ids)], 10)

        durations: list[float] = []
        for _ in range(repeats):
            for user_id in user_ids:
                started = perf_counter()
                pipeline.recommend(user_id, 10)
                durations.append((perf_counter() - started) * 1000.0)
        tracemalloc.start()
        pipeline.recommend(user_ids[0], 10)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        values = np.asarray(durations, dtype=np.float64)
        rows.append(
            {
                "candidate_budget": budget,
                "measured_requests": len(durations),
                "mean_ms": float(values.mean()),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "peak_python_allocated_mib": peak_bytes / 1024**2,
            }
        )
    return {
        "config": config_path,
        "known_users": len(user_ids),
        "repeats_per_budget": repeats,
        "warmup_per_budget": warmup,
        "budgets": rows,
        "interpretation": {
            "memory_scope": (
                "tracemalloc Python allocations only; native Torch/FAISS allocations "
                "are not included"
            ),
            "latency_scope": "single-process sequential serving on the current host",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving_content.yaml")
    parser.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        default="results/video_games_2018/phase6/candidate_budget_benchmark.json",
    )
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets):
        parser.error("budgets must be positive")
    result = benchmark(args.config, args.budgets, args.users, args.repeats, args.warmup)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Profile every stage of the frozen serving pipeline on known users."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from recsys.serving.pipeline import ServingPipeline


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def profile(
    config_path: str,
    users: int,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    pipeline = ServingPipeline.from_yaml(config_path)
    user_ids = list(pipeline.histories)[:users]
    if not user_ids:
        raise ValueError("No known users are available for profiling")

    for index in range(warmup):
        pipeline.recommend(user_ids[index % len(user_ids)], 10)

    stage_values: dict[str, list[float]] = defaultdict(list)
    candidate_counts: list[int] = []
    total_values: list[float] = []
    for _ in range(repeats):
        for user_id in user_ids:
            _, trace = pipeline.recommend_with_trace(user_id, 10)
            total_values.append(float(trace["total_pipeline_ms"]))
            candidate_counts.append(int(trace["candidate_count"]))
            for stage, value in trace.items():
                if stage not in {"candidate_count", "fallback", "total_pipeline_ms"}:
                    stage_values[stage].append(float(value))

    total_mean = mean(total_values)
    stage_summary = {}
    for stage, values in stage_values.items():
        stage_summary[stage] = {
            **_summary(values),
            "share_of_total_mean_pct": mean(values) / total_mean * 100.0,
        }
    stages_by_mean = sorted(
        stage_summary,
        key=lambda stage: stage_summary[stage]["mean_ms"],
        reverse=True,
    )
    return {
        "config": config_path,
        "known_users": len(user_ids),
        "repeats_per_user": repeats,
        "warmup_requests": warmup,
        "measured_requests": len(total_values),
        "candidate_budget": pipeline.pool_size,
        "candidate_count": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": mean(candidate_counts),
        },
        "total_pipeline": _summary(total_values),
        "stages": stage_summary,
        "stages_by_mean_desc": stages_by_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving.yaml")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output", default="results/video_games_2018/serving_stage_profile.json"
    )
    args = parser.parse_args()
    if args.users < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("users and repeats must be positive; warmup cannot be negative")

    result = profile(args.config, args.users, args.repeats, args.warmup)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Run a closed-loop concurrent load test against the recommendation API."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _LOCAL.session = session
    return session


def _request(endpoint: str, timeout: float) -> tuple[float, bool]:
    started = perf_counter()
    try:
        response = _session().get(endpoint, timeout=timeout)
        success = response.status_code == 200
    except requests.RequestException:
        success = False
    return (perf_counter() - started) * 1000.0, success


def run_level(
    base_url: str,
    user_ids: list[str],
    *,
    concurrency: int,
    requests_count: int,
    timeout: float,
) -> dict[str, float | int]:
    endpoints = [
        f"{base_url.rstrip('/')}/recommend/{user_ids[index % len(user_ids)]}?k=10"
        for index in range(requests_count)
    ]
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        observations = list(executor.map(lambda url: _request(url, timeout), endpoints))
    elapsed = perf_counter() - started
    latencies = np.asarray([latency for latency, _ in observations], dtype=np.float64)
    successes = sum(success for _, success in observations)
    return {
        "concurrency": concurrency,
        "requests": requests_count,
        "successes": successes,
        "errors": requests_count - successes,
        "error_rate": float(1.0 - successes / requests_count),
        "qps": float(requests_count / elapsed),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(latencies.mean()),
        "wall_seconds": elapsed,
    }


def _plot(frame: pd.DataFrame, output: Path) -> None:
    figure, left = plt.subplots(figsize=(8, 4.8))
    right = left.twinx()
    left.plot(frame["concurrency"], frame["qps"], marker="o", color="#167D9A", label="QPS")
    right.plot(
        frame["concurrency"],
        frame["p95_ms"],
        marker="s",
        color="#C44E52",
        label="P95 latency",
    )
    left.set_xlabel("Concurrent clients")
    left.set_ylabel("Throughput (requests/s)", color="#167D9A")
    right.set_ylabel("P95 latency (ms)", color="#C44E52")
    left.set_xticks(frame["concurrency"])
    left.grid(axis="y", alpha=0.25)
    figure.suptitle("Local single-process latency-throughput curve")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--interactions", default="data/processed/video_games_2018/interactions.parquet"
    )
    parser.add_argument("--output", default="results/video_games_2018/phase6/load_test.json")
    args = parser.parse_args()
    if any(level < 1 for level in args.concurrency) or args.requests_per_level < 1:
        parser.error("concurrency and requests-per-level must be positive")

    interactions = pd.read_parquet(args.interactions, columns=["user_id"])
    user_ids = interactions["user_id"].drop_duplicates().astype(str).head(100).tolist()
    if not user_ids:
        raise ValueError("No user IDs available for load testing")
    health = requests.get(f"{args.url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    for index in range(args.warmup):
        endpoint = f"{args.url.rstrip('/')}/recommend/{user_ids[index % len(user_ids)]}?k=10"
        response = requests.get(endpoint, timeout=args.timeout)
        response.raise_for_status()

    levels = [
        run_level(
            args.url,
            user_ids,
            concurrency=level,
            requests_count=args.requests_per_level,
            timeout=args.timeout,
        )
        for level in args.concurrency
    ]
    payload: dict[str, Any] = {
        "environment": "local single Uvicorn process; no Redis configured",
        "method": "closed-loop clients using 100 known users",
        "warmup_requests": args.warmup,
        "capacity_claim": (
            "These measurements describe this machine and workload only; they are not a "
            "production capacity or SLA claim."
        ),
        "levels": levels,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(levels)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    _plot(frame, output.with_suffix(".png"))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

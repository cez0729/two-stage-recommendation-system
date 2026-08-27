"""Warm up and benchmark the local serving API."""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="A0266076X6KPZ6CCHGVS")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--output", default="results/video_games_2018/serving_benchmark.json"
    )
    args = parser.parse_args()
    endpoint = f"{args.url.rstrip('/')}/recommend/{args.user_id}?k=10"
    for _ in range(args.warmup):
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
    latencies = []
    started = time.perf_counter()
    for _ in range(args.requests):
        request_started = time.perf_counter()
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        latencies.append((time.perf_counter() - request_started) * 1000)
    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)
    result = {
        "warmup_requests": args.warmup,
        "measured_requests": args.requests,
        "p50_ms": ordered[int(0.50 * (len(ordered) - 1))],
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "qps": args.requests / elapsed,
        "mean_ms": statistics.mean(latencies),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

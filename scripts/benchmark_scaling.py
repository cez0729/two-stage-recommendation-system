"""Separate HTTP worker scaling from synthetic exact-index catalog scaling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import psutil
import requests
import torch
from load_test_api import run_level

from recsys.utils.io import write_json


def index_scaling(sizes: list[int], repeats: int = 50) -> list[dict]:
    """Measure exact search on seeded normalized synthetic vectors, not real items."""
    faiss.omp_set_num_threads(1)
    rows = []
    for size in sizes:
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((size, 64), dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(64)
        started = time.perf_counter()
        index.add(vectors)
        build_seconds = time.perf_counter() - started
        queries = rng.standard_normal((repeats, 64), dtype=np.float32)
        faiss.normalize_L2(queries)
        index.search(queries[:1], 100)
        durations = []
        for query in queries:
            started = time.perf_counter()
            index.search(query[None, :], 100)
            durations.append((time.perf_counter() - started) * 1000)
        rows.append({
            "catalog_size": size, "dimension": 64, "k": 100,
            "repetitions": repeats, "build_seconds": build_seconds,
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)),
            "serialized_index_mib": faiss.serialize_index(index).nbytes / 1024**2,
            "input_vectors_mib": vectors.nbytes / 1024**2,
        })
    return rows


def worker_scaling(workers: list[int], output_dir: Path, repeats: int = 3) -> list[dict]:
    users = pd.read_parquet(
        "data/processed/video_games_2018/interactions.parquet", columns=["user_id"]
    )["user_id"].drop_duplicates().astype(str).head(100).tolist()
    rows = []
    for count in workers:
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "RECSYS_CONFIG": "configs/serving_content.yaml",
                    "FEEDBACK_DB_PATH": str(output_dir / f"feedback_workers_{count}.db")})
        env.pop("REDIS_URL", None)
        command = [sys.executable, "-m", "uvicorn", "recsys.serving.api:app",
                   "--host", "127.0.0.1", "--port", str(port), "--workers", str(count),
                   "--no-access-log"]
        with (output_dir / f"workers_{count}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 240
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Uvicorn exited during initialization")
                    try:
                        response = requests.get(base + "/health", timeout=2)
                        startup_log = (output_dir / f"workers_{count}.log").read_text(
                            encoding="utf-8", errors="replace"
                        )
                        ready_workers = startup_log.count("Application startup complete.")
                        if response.ok and ready_workers >= count:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Uvicorn startup exceeded 240 seconds")
                    time.sleep(0.5)
                # Exercise sufficient fresh connections to warm all workers.
                for i in range(20):
                    response = requests.get(base + f"/recommend/{users[i]}?k=10", timeout=60)
                    response.raise_for_status()
                for repetition in range(1, repeats + 1):
                    for concurrency in [1, 5, 10, 20]:
                        result = run_level(base, users, concurrency=concurrency,
                                           requests_count=100, timeout=60)
                        active = psutil.Process(process.pid)
                        family = [active, *active.children(recursive=True)]
                        result.update(workers=count, repetition=repetition, rss_sum_mib=sum(
                            p.memory_info().rss for p in family if p.is_running()
                        ) / 1024**2)
                        rows.append(result)
                        write_json(output_dir / "worker_results_partial.json", {"rows": rows})
                        print(json.dumps(result), flush=True)
            finally:
                try:
                    family = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    family = []
                for child in family:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                psutil.wait_procs(family, timeout=10)
                for child in family:
                    if child.is_running():
                        child.kill()
                process.wait(timeout=10)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--catalog-sizes", nargs="+", type=int, default=[13923, 100000, 1000000])
    parser.add_argument("--output", default="results/video_games_2018/scaling/scaling.json")
    args = parser.parse_args()
    if args.repeats < 1 or any(worker < 1 for worker in args.workers):
        parser.error("Worker counts and repetitions must be positive")
    if any(size < 100 for size in args.catalog_sizes):
        parser.error("Catalog sizes must support Top-100 queries")
    log_dir = Path("runs/scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_rows = worker_scaling(args.workers, log_dir, args.repeats)
    frame = pd.DataFrame(worker_rows)
    medians = frame.groupby(["workers", "concurrency"])[["qps", "p95_ms"]].median()
    scaling_check = None
    if 1 in args.workers and 4 in args.workers:
        single = medians.loc[(1, 20)]
        four = medians.loc[(4, 20)]
        scaling_check = {
            "criterion": "At concurrency 20, four workers raise median QPS and lower median P95",
            "throughput_ratio_4_vs_1": float(four["qps"] / single["qps"]),
            "p95_ratio_4_vs_1": float(four["p95_ms"] / single["p95_ms"]),
            "met": bool(four["qps"] > single["qps"] and four["p95_ms"] < single["p95_ms"]),
        }
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "logical_cpus": os.cpu_count(), "faiss_threads": 1,
                        "processor": platform.processor(), "torch": torch.__version__,
                        "faiss": faiss.__version__,
                        "physical_memory_gib": psutil.virtual_memory().total / 1024**3},
        "scope": "Real content-serving HTTP worker sweep; separate synthetic IndexFlatIP sweep",
        "caveats": ["No Redis; 200 final candidates; one native thread per worker",
                    "Sum of RSS double-counts shared pages; not unique physical RAM",
                    "Synthetic vectors measure index cost only, not recommendation accuracy"],
        "acceptance": {"all_http_requests_succeeded": all(row["errors"] == 0
                                                        for row in worker_rows),
                       "four_worker_scaling": scaling_check},
        "worker_results": worker_rows,
        "index_results": index_scaling(args.catalog_sizes),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

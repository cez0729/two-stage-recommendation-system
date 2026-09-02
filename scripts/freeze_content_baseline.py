"""Freeze Phase 1-5 evidence with reproducible SHA-256 fingerprints."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from recsys.config import load_yaml
from recsys.utils.io import sha256_file, write_json

FROZEN_PATHS = [
    "configs/data.yaml",
    "configs/two_tower_v3_final.yaml",
    "configs/ranker.yaml",
    "configs/serving.yaml",
    "data/processed/video_games_2018/interactions.parquet",
    "data/processed/video_games_2018/items.parquet",
    "artifacts/video_games_2018/data_manifest.json",
    "artifacts/video_games_2018/faiss/item_flat.index",
    "artifacts/video_games_2018/faiss/faiss_item_ids.npy",
    "artifacts/video_games_2018/ranking/feature_schema.json",
    "models/video_games_2018/two_tower_v3_logq.pt",
    "models/video_games_2018/ranker.txt",
    "results/video_games_2018/two_tower_v3_final_metrics.json",
    "results/video_games_2018/final_test_metrics.json",
    "results/video_games_2018/serving_stage_profile.json",
    "results/video_games_2018/serving_benchmark_profiled.json",
]


def freeze(config_path: str) -> dict[str, object]:
    config = load_yaml(config_path)
    missing = [path for path in FROZEN_PATHS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze baseline; missing files: {missing}")
    files = {
        path: {"sha256": sha256_file(path), "bytes": Path(path).stat().st_size}
        for path in FROZEN_PATHS
    }
    payload: dict[str, object] = {
        "baseline": "Phase 1-5 Baseline V1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "policy": (
            "Phase 6 does not overwrite these assets. Model selection uses validation; "
            "the previously observed test set is not a tuning target."
        ),
        "files": files,
    }
    write_json(config["outputs"]["baseline_manifest"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/content_retrieval.yaml")
    args = parser.parse_args()
    payload = freeze(args.config)
    print(f"Frozen {len(payload['files'])} baseline assets")


if __name__ == "__main__":
    main()

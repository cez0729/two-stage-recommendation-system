"""Serving artifact manifest and integrity validation."""

from pathlib import Path
from typing import Any

import numpy as np

from recsys.utils.io import sha256_file, write_json

ARTIFACTS = {
    "two_tower_metrics": "results/video_games_2018/two_tower_v3_final_metrics.json",
    "faiss_benchmark": "results/video_games_2018/faiss_benchmark.json",
    "ranking_validation_metrics": "results/video_games_2018/ranking_validation_metrics.json",
    "final_test_metrics": "results/video_games_2018/final_test_metrics.json",
    "final_group_analysis": "results/video_games_2018/final_group_analysis.csv",
    "two_tower_checkpoint": "models/video_games_2018/two_tower_v3_logq.pt",
    "ranker": "models/video_games_2018/ranker.txt",
}


def build_manifest(root: str | Path = ".", version: str = "v1") -> dict[str, Any]:
    root = Path(root)
    entries = []
    for name, relative_path in ARTIFACTS.items():
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing serving artifact: {path}")
        entries.append(
            {
                "name": name,
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    item_ids_path = root / "artifacts/video_games_2018/faiss/faiss_item_ids.npy"
    item_count = int(len(np.load(item_ids_path))) if item_ids_path.is_file() else None
    return {"version": version, "item_count": item_count, "artifacts": entries}


def validate_manifest(manifest: dict[str, Any], root: str | Path = ".") -> None:
    root = Path(root)
    for entry in manifest["artifacts"]:
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")


def write_manifest(path: str | Path, root: str | Path = ".", version: str = "v1") -> dict[str, Any]:
    manifest = build_manifest(root, version)
    write_json(path, manifest)
    return manifest

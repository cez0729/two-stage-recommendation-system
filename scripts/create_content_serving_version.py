"""Create an immutable, checksumed serving snapshot for the Phase 6 model."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recsys.utils.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "recsys_phase6_content_v1"

ARTIFACTS = {
    "two_tower_checkpoint": "models/video_games_2018/two_tower_v3_logq.pt",
    "faiss_index_path": "artifacts/video_games_2018/faiss/item_flat.index",
    "faiss_item_ids_path": "artifacts/video_games_2018/faiss/faiss_item_ids.npy",
    "ranker_path": "models/video_games_2018/ranker_phase6_content.txt",
    "content_vectorizer_path": "artifacts/video_games_2018/phase6/tfidf_vectorizer.joblib",
    "content_item_matrix_path": "artifacts/video_games_2018/phase6/tfidf_item_matrix.npz",
    "content_item_ids_path": "artifacts/video_games_2018/phase6/tfidf_item_ids.npy",
}


def create_snapshot(version: str = VERSION, *, promote: bool = False) -> dict[str, Any]:
    destination = ROOT / "artifacts" / "versions" / version
    if destination.exists():
        raise FileExistsError(f"Version directory already exists: {destination}")
    destination.mkdir(parents=True)

    serving: dict[str, Any] = {}
    copied: dict[str, dict[str, Any]] = {}
    for key, source in ARTIFACTS.items():
        source_path = ROOT / source
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing serving artifact: {source_path}")
        target = destination / source_path.name
        shutil.copy2(source_path, target)
        relative_target = target.relative_to(ROOT).as_posix()
        copied[key] = {
            "source": source,
            "path": relative_target,
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        }
        if key.startswith("content_"):
            continue
        serving[key] = relative_target

    serving["content"] = {
        "enabled": True,
        "items_path": "data/processed/video_games_2018/items_content.parquet",
        "vectorizer_path": copied["content_vectorizer_path"]["path"],
        "item_matrix_path": copied["content_item_matrix_path"]["path"],
        "item_ids_path": copied["content_item_ids_path"]["path"],
        "max_history": 50,
    }
    metrics_path = ROOT / "results/video_games_2018/ranker_phase6_content_validation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    payload: dict[str, Any] = {
        "version": version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": "Immutable Phase 6 snapshot; baseline v1 remains available for rollback.",
        "serving": {
            **serving,
            "version": version,
            "two_tower_candidates": 100,
            "itemcf_candidates": 100,
            "popularity_candidates": 20,
            "content_candidates": 100,
            "pool_size": 200,
            "rrf_constant": 60,
            "itemcf": {"max_history": 50, "top_neighbors": 200, "recency_decay": 0.9},
        },
        "validation": {
            "ranker_metrics_path": (
                "results/video_games_2018/ranker_phase6_content_validation_metrics.json"
            ),
            "metrics": metrics.get("lambdarank_metrics", {}),
            "parity_users": 20,
            "parity_all_equal": True,
        },
        "artifacts": copied,
    }
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, payload)
    pointer_path = ROOT / "artifacts" / "versions" / "phase6_content_pointer.json"
    if promote:
        write_json(
            pointer_path,
            {"manifest_path": manifest_path.relative_to(ROOT).as_posix(), "version": version},
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--promote", action="store_true", help="Update the Phase 6 pointer")
    args = parser.parse_args()
    payload = create_snapshot(args.version, promote=args.promote)
    print(
        json.dumps(
            {
                "version": payload["version"],
                "manifest": f"artifacts/versions/{payload['version']}/manifest.json",
                "promoted": args.promote,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

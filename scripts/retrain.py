"""Reproducible orchestration and versioned snapshot for existing model stages."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from recsys.config import load_yaml
from recsys.serving.artifacts import ARTIFACTS
from recsys.utils.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def _run_stage(name: str, command: list[str], dry_run: bool) -> None:
    resolved = [sys.executable if token == "python" else token for token in command]
    print(f"[{name}] {' '.join(resolved)}")
    if not dry_run:
        subprocess.run(resolved, cwd=ROOT, check=True)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    names = ["two-stage-recsys", "torch", "pandas", "faiss-cpu", "lightgbm", "fastapi"]
    return {
        name: metadata.version(name)
        for name in names
        if _has_distribution(name)
    }


def _has_distribution(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _load_metrics() -> dict[str, float]:
    two_tower = json.loads(
        (ROOT / "results/video_games_2018/two_tower_v3_final_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    final_test = json.loads(
        (ROOT / "results/video_games_2018/final_test_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "two_tower_recall@100": float(two_tower["splits"]["test"]["metrics"]["recall@100"]),
        "candidate_recall@200": float(final_test["candidate_recall@200"]),
        "lambdarank_ndcg@10": float(final_test["lambdarank_metrics"]["ndcg@10"]),
    }


def _copy_snapshot(version: str, serving_config: dict[str, Any]) -> dict[str, str]:
    destination = ROOT / "artifacts" / "versions" / version
    destination.mkdir(parents=True, exist_ok=False)
    source_paths = {
        "two_tower_checkpoint": serving_config["serving"]["two_tower_checkpoint"],
        "faiss_index_path": serving_config["serving"]["faiss_index_path"],
        "faiss_item_ids_path": serving_config["serving"]["faiss_item_ids_path"],
        "ranker_path": serving_config["serving"]["ranker_path"],
    }
    resolved: dict[str, str] = {}
    for key, source in source_paths.items():
        source_path = ROOT / source
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing snapshot artifact: {source_path}")
        target = destination / source_path.name
        shutil.copy2(source_path, target)
        resolved[key] = target.relative_to(ROOT).as_posix()
    return resolved


def _parity_check() -> bool:
    result = subprocess.run(
        [sys.executable, "scripts/check_serving_parity.py", "--users", "20"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return "all_equal': True" in result.stdout or '"all_equal": True' in result.stdout


def create_version_manifest(
    version: str, config: dict[str, Any], *, parity_passed: bool
) -> dict[str, Any]:
    serving_config = load_yaml(ROOT / config["serving_config"])
    serving_paths = _copy_snapshot(version, serving_config)
    metrics = _load_metrics()
    thresholds = config["promotion"]["thresholds"]
    acceptance = {
        key: value >= float(thresholds[key]["minimum"]) - float(thresholds[key]["tolerance"])
        for key, value in metrics.items()
    }
    acceptance["parity"] = parity_passed
    payload: dict[str, Any] = {
        "version": version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit() or "unavailable",
        "python": platform.python_version(),
        "package_versions": _package_versions(),
        "data_fingerprint": sha256_file(ROOT / config["data_manifest_path"]),
        "metrics": metrics,
        "promotion": {
            "thresholds": thresholds,
            "acceptance": acceptance,
            "eligible": all(acceptance.values()),
        },
        "serving": serving_paths,
        "source_artifacts": {
            name: {"path": path, "sha256": sha256_file(ROOT / path)}
            for name, path in ARTIFACTS.items()
        },
    }
    manifest_path = ROOT / "artifacts" / "versions" / version / "manifest.json"
    write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain.yaml")
    parser.add_argument("--version")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config)
    version = args.version or datetime.now(UTC).strftime(
        f"{config.get('version_prefix', 'recsys')}_%Y%m%dT%H%M%SZ"
    )
    if not args.snapshot_only:
        for stage in config["stages"]:
            _run_stage(str(stage["name"]), list(stage["command"]), args.dry_run)
    if args.dry_run:
        print(f"dry_run_version={version}")
        return
    parity_passed = True if args.skip_parity else _parity_check()
    manifest = create_version_manifest(version, config, parity_passed=parity_passed)
    print(json.dumps({"version": version, "eligible": manifest["promotion"]["eligible"]}))


if __name__ == "__main__":
    main()

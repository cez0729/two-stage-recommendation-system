"""Download and audit Amazon Reviews 2023 category files."""

import argparse
import gzip
import json
import logging
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recsys.config import load_yaml
from recsys.data.schemas import missing_fields
from recsys.logging import configure_logging
from recsys.utils.io import sha256_file, write_json

LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> None:
    """Download atomically and keep an existing non-empty file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        LOGGER.info("Using existing file: %s", destination)
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    LOGGER.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "two-stage-recsys/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    LOGGER.info("Saved %s (%.2f MB)", destination, destination.stat().st_size / 1024**2)


def audit_jsonl_gz(
    path: Path,
    required_fields: set[str],
    limit: int | None = None,
) -> dict[str, Any]:
    """Stream a gzipped JSONL file and validate its records."""
    fields: set[str] = set()
    rows = 0
    invalid_json_rows = 0
    missing_required_rows = 0
    examples: list[dict[str, Any]] = []

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if limit is not None and rows >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_rows += 1
                continue
            if not isinstance(record, dict):
                invalid_json_rows += 1
                continue
            rows += 1
            fields.update(record)
            missing = missing_fields(record, required_fields)
            if missing:
                missing_required_rows += 1
                if len(examples) < 5:
                    examples.append({"line": line_number, "missing": missing})

    if rows == 0:
        raise ValueError(f"No valid JSON object found in {path}")
    if required_fields.difference(fields):
        raise ValueError(f"Required fields absent from {path}: {sorted(required_fields - fields)}")
    return {
        "rows_scanned": rows,
        "scan_limit": limit,
        "fields": sorted(fields),
        "invalid_json_rows": invalid_json_rows,
        "missing_required_rows": missing_required_rows,
        "missing_examples": examples,
    }


def run_download(config_path: str | Path, limit: int | None = None) -> dict[str, Any]:
    """Download review/metadata archives, audit schemas, and write a manifest."""
    config = load_yaml(config_path)
    dataset = config["dataset"]
    raw_dir = Path(config["paths"]["raw_dir"])
    category = str(dataset["category"])
    review_path = raw_dir / str(dataset.get("review_filename", f"{category}.jsonl.gz"))
    metadata_path = raw_dir / str(
        dataset.get("metadata_filename", f"meta_{category}.jsonl.gz")
    )
    review_schema = dataset.get(
        "review_schema",
        {
            "user_id": "user_id",
            "item_id": "parent_asin",
            "timestamp": "timestamp",
            "rating": "rating",
        },
    )
    metadata_schema = dataset.get(
        "metadata_schema",
        {"item_id": "parent_asin", "title": "title", "category": "main_category"},
    )
    review_required = {
        str(review_schema[key]) for key in ["user_id", "item_id", "timestamp", "rating"]
    }
    metadata_required = {
        str(metadata_schema[key]) for key in ["item_id", "title", "category"]
    }

    download_file(str(dataset["review_url"]), review_path)
    download_file(str(dataset["metadata_url"]), metadata_path)

    files = {
        "reviews": {
            "path": review_path.as_posix(),
            "url": dataset["review_url"],
            "bytes": review_path.stat().st_size,
            "sha256": sha256_file(review_path),
            "audit": audit_jsonl_gz(review_path, review_required, limit),
        },
        "metadata": {
            "path": metadata_path.as_posix(),
            "url": dataset["metadata_url"],
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
            "audit": audit_jsonl_gz(metadata_path, metadata_required, limit),
        },
    }
    manifest = {
        "dataset": dataset["name"],
        "category": category,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
    }
    output = Path(config["paths"]["artifacts_dir"]) / "raw_manifest.json"
    write_json(output, manifest)
    LOGGER.info("Wrote raw manifest: %s", output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Audit at most N rows per file")
    args = parser.parse_args()
    configure_logging()
    run_download(args.config, args.limit)


if __name__ == "__main__":
    main()

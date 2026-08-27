"""Clean Amazon review interactions and create deterministic ID mappings."""

import argparse
import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.config import load_yaml
from recsys.logging import configure_logging
from recsys.utils.io import sha256_file, write_json

LOGGER = logging.getLogger(__name__)
DEFAULT_REVIEW_SCHEMA = {
    "user_id": "user_id",
    "item_id": "parent_asin",
    "timestamp": "timestamp",
    "rating": "rating",
    "timestamp_unit": "ms",
}


def read_jsonl_fields(path: Path, fields: list[str]) -> pd.DataFrame:
    """Read selected fields from gzipped JSONL while ignoring unused large text fields."""
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            rows.append({field: record.get(field) for field in fields})
    return pd.DataFrame.from_records(rows, columns=fields)


def iterative_k_core(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Repeatedly remove sparse users/items until both constraints hold."""
    current = interactions
    iterations: list[dict[str, int]] = []
    while True:
        before = len(current)
        valid_users = current["user_id"].value_counts()
        kept_users = valid_users[valid_users >= min_user_interactions].index
        current = current[current["user_id"].isin(kept_users)]
        valid_items = current["item_id"].value_counts()
        kept_items = valid_items[valid_items >= min_item_interactions].index
        current = current[current["item_id"].isin(kept_items)]
        iterations.append(
            {
                "iteration": len(iterations) + 1,
                "rows": len(current),
                "users": current["user_id"].nunique(),
                "items": current["item_id"].nunique(),
            }
        )
        if len(current) == before:
            return current.copy(), iterations
        if current.empty:
            raise ValueError("K-core filtering removed every interaction")


def _cap_interactions(interactions: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Cap by deterministically selecting users, keeping each selected user's chronology."""
    if len(interactions) <= maximum:
        return interactions
    users = pd.Series(interactions["user_id"].unique()).sample(frac=1, random_state=seed)
    counts = interactions["user_id"].value_counts()
    cumulative = counts.reindex(users).cumsum()
    selected = cumulative[cumulative <= maximum].index
    capped = interactions[interactions["user_id"].isin(selected)]
    if capped.empty:
        raise ValueError("Interaction cap is smaller than the first complete user history")
    return capped


def clean_interactions(
    raw: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, filter, k-core, cap, and map review interactions."""
    dataset = config["dataset"]
    review_schema = dataset.get("review_schema", DEFAULT_REVIEW_SCHEMA)
    stages: dict[str, int] = {"raw_rows": len(raw)}
    rename = {
        str(review_schema["user_id"]): "user_id",
        str(review_schema["item_id"]): "item_id",
        str(review_schema["timestamp"]): "timestamp",
        str(review_schema["rating"]): "rating",
    }
    frame = raw.rename(columns=rename).copy()
    frame = frame.dropna(subset=["user_id", "item_id", "timestamp", "rating"])
    stages["after_required_null_filter"] = len(frame)

    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "rating"])
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], unit=str(review_schema.get("timestamp_unit", "ms")), utc=True
    )
    frame = frame.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="first")
    stages["after_exact_deduplication"] = len(frame)

    if bool(dataset.get("deduplicate_user_item", True)):
        frame = frame.sort_values("timestamp", kind="stable")
        frame = frame.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    stages["after_user_item_deduplication"] = len(frame)

    frame = frame[frame["rating"] >= float(dataset["min_rating"])]
    stages["after_rating_filter"] = len(frame)
    frame, k_core_iterations = iterative_k_core(
        frame,
        int(dataset["min_user_interactions"]),
        int(dataset["min_item_interactions"]),
    )
    stages["after_k_core"] = len(frame)
    frame = _cap_interactions(frame, int(dataset["max_interactions"]), int(config["project_seed"]))
    stages["after_interaction_cap"] = len(frame)

    user_ids = sorted(frame["user_id"].unique())
    item_ids = sorted(frame["item_id"].unique())
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    frame["user_idx"] = frame["user_id"].map(user_map).astype("int32")
    frame["item_idx"] = frame["item_id"].map(item_map).astype("int32")
    frame = frame.sort_values(
        ["user_idx", "timestamp", "item_idx"], kind="stable"
    ).reset_index(drop=True)
    stats = {
        "stages": stages,
        "k_core_iterations": k_core_iterations,
        "users": len(user_ids),
        "items": len(item_ids),
        "timestamp_min": frame["timestamp"].min().isoformat(),
        "timestamp_max": frame["timestamp"].max().isoformat(),
        "rating_mean": round(float(frame["rating"].mean()), 6),
    }
    return frame, stats


def _clean_items(raw: pd.DataFrame, item_mapping: pd.DataFrame) -> pd.DataFrame:
    items = raw.copy()
    items = items.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"], keep="first")
    items["item_id"] = items["item_id"].astype(str)
    items["title"] = items["title"].fillna("").astype(str)
    items["category"] = items["category"].fillna("Unknown").astype(str)
    for column in ["price", "avg_rating", "rating_number"]:
        items[column] = pd.to_numeric(items[column], errors="coerce")
    items = item_mapping.merge(items, on="item_id", how="left")
    items["title"] = items["title"].fillna("")
    items["category"] = items["category"].fillna("Unknown")
    columns = [
        "item_idx", "item_id", "title", "category", "price", "avg_rating", "rating_number"
    ]
    return items[columns]


def run_clean(config_path: str | Path) -> dict[str, Any]:
    """Run the full cleaning job and persist Parquet files plus a manifest."""
    config = load_yaml(config_path)
    paths = config["paths"]
    category = config["dataset"]["category"]
    raw_dir = Path(paths["raw_dir"])
    processed_dir = Path(paths["processed_dir"])
    artifacts_dir = Path(paths["artifacts_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "mappings").mkdir(parents=True, exist_ok=True)

    dataset = config["dataset"]
    review_schema = dataset.get("review_schema", DEFAULT_REVIEW_SCHEMA)
    review_fields = [
        str(review_schema[key]) for key in ["user_id", "item_id", "timestamp", "rating"]
    ]
    review_filename = str(dataset.get("review_filename", f"{category}.jsonl.gz"))
    metadata_filename = str(dataset.get("metadata_filename", f"meta_{category}.jsonl.gz"))
    LOGGER.info("Reading review fields from compressed JSONL")
    raw_reviews = read_jsonl_fields(raw_dir / review_filename, review_fields)
    interactions, stats = clean_interactions(raw_reviews, config)
    user_mapping = interactions[["user_idx", "user_id"]].drop_duplicates().sort_values("user_idx")
    item_mapping = interactions[["item_idx", "item_id"]].drop_duplicates().sort_values("item_idx")

    LOGGER.info("Reading metadata fields from compressed JSONL")
    metadata_schema = dataset.get(
        "metadata_schema",
        {
            "item_id": "parent_asin",
            "title": "title",
            "category": "main_category",
            "price": "price",
            "avg_rating": "average_rating",
            "rating_number": "rating_number",
        },
    )
    metadata_fields = [str(value) for value in metadata_schema.values() if value is not None]
    raw_items = read_jsonl_fields(raw_dir / metadata_filename, metadata_fields)
    raw_items = raw_items.rename(
        columns={str(value): key for key, value in metadata_schema.items() if value is not None}
    )
    for optional in ["price", "avg_rating", "rating_number"]:
        if optional not in raw_items:
            raw_items[optional] = None
    items = _clean_items(raw_items, item_mapping)
    users = (
        interactions.groupby(["user_idx", "user_id"], as_index=False)
        .agg(
            total_interactions=("item_idx", "size"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
        )
    )

    outputs = {
        "interactions": processed_dir / "interactions_clean.parquet",
        "items": processed_dir / "items.parquet",
        "users": processed_dir / "users.parquet",
        "user_mapping": artifacts_dir / "mappings" / "user_mapping.parquet",
        "item_mapping": artifacts_dir / "mappings" / "item_mapping.parquet",
    }
    interactions.to_parquet(outputs["interactions"], index=False)
    items.to_parquet(outputs["items"], index=False)
    users.to_parquet(outputs["users"], index=False)
    user_mapping.to_parquet(outputs["user_mapping"], index=False)
    item_mapping.to_parquet(outputs["item_mapping"], index=False)

    interaction_schema = {
        "table": "interactions_clean",
        "grain": "one retained positive user-item interaction",
        "columns": {
            "user_idx": "contiguous internal user number, starting at 0",
            "item_idx": "contiguous internal item number, starting at 0",
            "user_id": "original anonymized Amazon user identifier",
            "item_id": "original Amazon item identifier used by the configured dataset",
            "timestamp": "interaction time converted from the configured unit to UTC",
            "rating": "original star rating; retained rows are at least min_rating",
        },
    }
    write_json(artifacts_dir / "schemas" / "interaction_schema.json", interaction_schema)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": config["dataset"]["name"],
        "category": category,
        "filters": {
            key: config["dataset"][key]
            for key in [
                "min_rating",
                "deduplicate_user_item",
                "min_user_interactions",
                "min_item_interactions",
                "max_interactions",
            ]
        },
        **stats,
        "metadata_match": {
            "matched_items": int(items["title"].ne("").sum()),
            "missing_titles": int(items["title"].eq("").sum()),
        },
        "outputs": {
            name: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    write_json(artifacts_dir / "data_manifest.json", manifest)
    LOGGER.info(
        "Clean data: %d interactions, %d users, %d items",
        len(interactions),
        stats["users"],
        stats["items"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    configure_logging()
    run_clean(args.config)


if __name__ == "__main__":
    main()

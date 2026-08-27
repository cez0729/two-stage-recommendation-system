"""Create leakage-safe per-user chronological recommendation splits."""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.config import load_yaml
from recsys.logging import configure_logging
from recsys.utils.io import sha256_file, write_json

LOGGER = logging.getLogger(__name__)
TARGET_SPLITS = ["rank_train", "validation", "test"]


def chronological_split(
    interactions: pd.DataFrame,
    *,
    same_timestamp_policy: str = "error",
) -> pd.DataFrame:
    """Assign all but each user's final three events to retrieval training."""
    required = {"user_idx", "item_idx", "timestamp"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    frame = interactions.sort_values(
        ["user_idx", "timestamp", "item_idx"], kind="stable"
    ).reset_index(drop=True)
    counts = frame.groupby("user_idx")["item_idx"].transform("size")
    if (counts < 5).any():
        invalid = int(frame.loc[counts < 5, "user_idx"].nunique())
        raise ValueError(f"Found {invalid} users with fewer than five interactions")

    if same_timestamp_policy == "error":
        position_from_end = frame.groupby("user_idx").cumcount(ascending=False)
        frame["split"] = "retrieval_train"
        frame.loc[position_from_end == 2, "split"] = "rank_train"
        frame.loc[position_from_end == 1, "split"] = "validation"
        frame.loc[position_from_end == 0, "split"] = "test"
        return frame
    if same_timestamp_policy != "select_one_per_target_timestamp":
        raise ValueError(f"Unknown same_timestamp_policy: {same_timestamp_policy}")

    frame["_timestamp_rank_from_end"] = (
        frame.groupby("user_idx")["timestamp"].rank(method="dense", ascending=False).astype(int)
    )
    distinct_counts = frame.groupby("user_idx")["_timestamp_rank_from_end"].max()
    history_counts = (
        frame["_timestamp_rank_from_end"].gt(3).groupby(frame["user_idx"]).sum()
    )
    eligible_users = distinct_counts[(distinct_counts >= 4) & (history_counts >= 2)].index
    eligible = frame[frame["user_idx"].isin(eligible_users)].copy()
    history = eligible[eligible["_timestamp_rank_from_end"] > 3].copy()
    history["split"] = "retrieval_train"
    target_candidates = eligible[eligible["_timestamp_rank_from_end"] <= 3]
    targets = target_candidates.groupby(
        ["user_idx", "_timestamp_rank_from_end"], sort=False, as_index=False
    ).tail(1).copy()
    targets["split"] = targets["_timestamp_rank_from_end"].map(
        {3: "rank_train", 2: "validation", 1: "test"}
    )
    if targets.empty:
        raise ValueError("No users remain after resolving timestamp ties")
    result = pd.concat([history, targets], ignore_index=True)
    result = result.drop(columns="_timestamp_rank_from_end")
    result = result.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
    result.attrs["split_diagnostics"] = {
        "input_users": int(frame["user_idx"].nunique()),
        "eligible_users": int(result["user_idx"].nunique()),
        "excluded_users_insufficient_distinct_timestamps": int(
            frame["user_idx"].nunique() - len(eligible_users)
        ),
        "ambiguous_target_rows_dropped": int(len(target_candidates) - len(targets)),
        "same_timestamp_policy": same_timestamp_policy,
    }
    return result.reset_index(drop=True)


def validate_split(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate counts, strict timestamps, and unseen target items."""
    target_counts = (
        frame[frame["split"].isin(TARGET_SPLITS)]
        .groupby(["user_idx", "split"])
        .size()
        .unstack(fill_value=0)
    )
    bad_counts = target_counts.ne(1).any(axis=1)
    if bad_counts.any():
        raise ValueError(f"Target split counts are invalid for {int(bad_counts.sum())} users")

    timestamps = frame.pivot_table(
        index="user_idx", columns="split", values="timestamp", aggfunc="max"
    )
    strict_order = (
        (timestamps["retrieval_train"] < timestamps["rank_train"])
        & (timestamps["rank_train"] < timestamps["validation"])
        & (timestamps["validation"] < timestamps["test"])
    )
    if not strict_order.all():
        raise ValueError(
            f"Strict chronological ordering failed for {int((~strict_order).sum())} users; "
            "inspect duplicate timestamps before continuing"
        )

    seen: dict[int, set[int]] = {}
    target_repeats = 0
    for row in frame.itertuples(index=False):
        history = seen.setdefault(int(row.user_idx), set())
        if row.split in TARGET_SPLITS and int(row.item_idx) in history:
            target_repeats += 1
        history.add(int(row.item_idx))
    if target_repeats:
        raise ValueError(f"Found {target_repeats} target items already present in user history")

    return {
        "users": int(frame["user_idx"].nunique()),
        "items": int(frame["item_idx"].nunique()),
        "interactions": len(frame),
        "strict_time_order": True,
        "target_items_absent_from_history": True,
    }


def run_split(config_path: str | Path) -> dict[str, Any]:
    """Load cleaned interactions, split them, validate, and persist statistics."""
    config = load_yaml(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    input_path = processed_dir / "interactions_clean.parquet"
    output_path = processed_dir / "interactions.parquet"

    interactions = pd.read_parquet(input_path)
    same_timestamp_policy = str(config.get("split", {}).get("same_timestamp_policy", "error"))
    split_frame = chronological_split(
        interactions, same_timestamp_policy=same_timestamp_policy
    )
    split_diagnostics = split_frame.attrs.get("split_diagnostics", {})
    validation = validate_split(split_frame)
    split_frame.to_parquet(output_path, index=False)

    split_counts = {
        str(name): int(count) for name, count in split_frame["split"].value_counts().items()
    }
    split_users = {
        str(name): int(count)
        for name, count in split_frame.groupby("split")["user_idx"].nunique().items()
    }
    stats = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **validation,
        "input_interactions": len(interactions),
        "split_diagnostics": split_diagnostics,
        "split_interactions": split_counts,
        "split_users": split_users,
        "output": {
            "path": output_path.as_posix(),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
    }
    write_json(artifacts_dir / "split_stats.json", stats)
    LOGGER.info("Wrote chronological splits: %s", split_counts)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    configure_logging()
    run_split(args.config)


if __name__ == "__main__":
    main()

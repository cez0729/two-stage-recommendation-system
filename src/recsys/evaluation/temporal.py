"""Global-cutoff query construction for temporal robustness experiments."""

from __future__ import annotations

import pandas as pd


def build_global_cutoff_queries(
    interactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    end: pd.Timestamp | None = None,
    max_history: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit before cutoff and select each eligible user's first later interaction."""
    ordered = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
    train = ordered[ordered["timestamp"] < cutoff].copy()
    evaluation = ordered[ordered["timestamp"] >= cutoff]
    if end is not None:
        evaluation = evaluation[evaluation["timestamp"] < end]
    first_targets = evaluation.groupby("user_idx", sort=True).head(1)
    histories = (
        train.groupby("user_idx", sort=True)["item_idx"]
        .apply(lambda values: values.astype(int).tolist()[-max_history:])
        .rename("history")
    )
    queries = first_targets[["user_idx", "item_idx", "timestamp"]].merge(
        histories,
        left_on="user_idx",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    queries = queries.rename(
        columns={"item_idx": "target_item_idx", "timestamp": "target_timestamp"}
    ).reset_index(drop=True)
    if queries.empty:
        raise ValueError("Global cutoff produced no users with both history and a later target")
    return train, queries

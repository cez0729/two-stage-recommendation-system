"""Temporal prefix samples and evaluation queries for two-tower retrieval."""

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset


class PrefixInteractionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Each target interaction is paired only with the user's preceding history."""

    def __init__(self, interactions: pd.DataFrame, max_history: int) -> None:
        ordered = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
        users: list[int] = []
        targets: list[int] = []
        histories: list[list[int]] = []
        lengths: list[int] = []
        for user_idx, group in ordered.groupby("user_idx", sort=True):
            items = group["item_idx"].astype(int).tolist()
            for position, target in enumerate(items):
                prefix = items[max(0, position - max_history) : position]
                padded = [item + 1 for item in prefix]
                padded.extend([0] * (max_history - len(padded)))
                users.append(int(user_idx))
                targets.append(target)
                histories.append(padded)
                lengths.append(len(prefix))
        self.user_ids = torch.tensor(users, dtype=torch.long)
        self.target_items = torch.tensor(targets, dtype=torch.long)
        self.histories = torch.tensor(histories, dtype=torch.long)
        self.history_lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.user_ids[index],
            self.histories[index],
            self.history_lengths[index],
            self.target_items[index],
        )


@dataclass
class EvaluationQueries:
    """Tensor inputs plus full histories required for seen-item filtering."""

    user_ids: torch.Tensor
    histories: torch.Tensor
    history_lengths: torch.Tensor
    target_items: torch.Tensor
    seen_histories: list[set[int]]


def build_evaluation_queries(
    interactions: pd.DataFrame,
    *,
    split: str,
    training_catalog: set[int],
    max_history: int,
) -> EvaluationQueries:
    """Build cutoff-safe validation or test queries."""
    history_splits = {
        "rank_train": {"retrieval_train"},
        "validation": {"retrieval_train", "rank_train"},
        "test": {"retrieval_train", "rank_train", "validation"},
    }
    if split not in history_splits:
        raise ValueError(f"Unsupported evaluation split: {split}")
    ordered = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable")
    users: list[int] = []
    histories: list[list[int]] = []
    lengths: list[int] = []
    targets: list[int] = []
    seen_histories: list[set[int]] = []
    for user_idx, group in ordered.groupby("user_idx", sort=True):
        target_rows = group[group["split"] == split]
        if len(target_rows) != 1:
            raise ValueError(f"User {user_idx} has {len(target_rows)} targets in {split}")
        full_history = (
            group[group["split"].isin(history_splits[split])]["item_idx"].astype(int).tolist()
        )
        model_history = [item for item in full_history if item in training_catalog][-max_history:]
        padded = [item + 1 for item in model_history]
        padded.extend([0] * (max_history - len(padded)))
        users.append(int(user_idx))
        histories.append(padded)
        lengths.append(len(model_history))
        targets.append(int(target_rows.iloc[0]["item_idx"]))
        seen_histories.append(set(full_history))
    return EvaluationQueries(
        user_ids=torch.tensor(users, dtype=torch.long),
        histories=torch.tensor(histories, dtype=torch.long),
        history_lengths=torch.tensor(lengths, dtype=torch.long),
        target_items=torch.tensor(targets, dtype=torch.long),
        seen_histories=seen_histories,
    )

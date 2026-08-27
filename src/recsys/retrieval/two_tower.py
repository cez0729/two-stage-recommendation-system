"""PyTorch two-tower retrieval model with prefix-history user encoding."""

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerModel(nn.Module):
    """Encode users and items into a shared normalized retrieval space."""

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_history: int = 20,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.max_history = max_history
        self.temperature = temperature
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_users(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user IDs, mean prefix histories, and normalized history length."""
        user_vectors = self.user_embedding(user_ids)
        history_embeddings = self.item_embedding(history_items)
        mask = history_items.ne(0).unsqueeze(-1)
        history_sum = (history_embeddings * mask).sum(dim=1)
        denominator = history_lengths.clamp_min(1).unsqueeze(1).to(history_sum.dtype)
        history_mean = history_sum / denominator
        normalized_length = (
            torch.log1p(history_lengths.to(history_sum.dtype))
            / torch.log1p(torch.tensor(float(self.max_history), device=history_sum.device))
        ).unsqueeze(1)
        features = torch.cat([user_vectors, history_mean, normalized_length], dim=1)
        return F.normalize(self.user_mlp(features), dim=1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode zero-based item IDs into normalized vectors."""
        embeddings = self.item_embedding(item_ids + 1)
        return F.normalize(self.item_mlp(embeddings), dim=1)

    def in_batch_loss(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_lengths: torch.Tensor,
        target_items: torch.Tensor,
        item_sampling_probabilities: torch.Tensor | None = None,
        sampling_correction_weight: float = 0.0,
    ) -> torch.Tensor:
        """Use multi-positive in-batch softmax with optional logQ correction."""
        users = self.encode_users(user_ids, history_items, history_lengths)
        items = self.encode_items(target_items)
        logits = users @ items.T / self.temperature
        if item_sampling_probabilities is not None and sampling_correction_weight > 0:
            target_probabilities = item_sampling_probabilities[target_items].clamp_min(1e-12)
            logits = logits - sampling_correction_weight * target_probabilities.log().unsqueeze(0)
        positive_mask = user_ids[:, None].eq(user_ids[None, :]) | target_items[:, None].eq(
            target_items[None, :]
        )
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()

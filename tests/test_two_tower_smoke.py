import pandas as pd
import torch

from recsys.retrieval.dataset import PrefixInteractionDataset
from recsys.retrieval.two_tower import TwoTowerModel


def _toy_interactions() -> pd.DataFrame:
    rows = []
    for user in range(4):
        for position, item in enumerate([user, user + 4, user + 8]):
            rows.append(
                {
                    "user_idx": user,
                    "item_idx": item,
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                    + pd.Timedelta(days=position),
                }
            )
    return pd.DataFrame(rows)


def test_prefix_dataset_never_contains_its_target() -> None:
    dataset = PrefixInteractionDataset(_toy_interactions(), max_history=5)

    for _, history, _, target in dataset:
        shifted_target = int(target) + 1
        assert shifted_target not in history.tolist()


def test_gru_encoding_uses_order_and_ignores_padding() -> None:
    torch.manual_seed(42)
    model = TwoTowerModel(
        num_users=2, num_items=12, embedding_dim=8, hidden_dim=16,
        dropout=0.0, max_history=4, history_encoder="gru",
    ).eval()
    users = torch.tensor([0])
    lengths = torch.tensor([2])
    original = model.encode_users(users, torch.tensor([[1, 2, 0, 0]]), lengths)
    reversed_ = model.encode_users(users, torch.tensor([[2, 1, 0, 0]]), lengths)
    changed_padding = model.encode_users(users, torch.tensor([[1, 2, 3, 4]]), lengths)
    assert not torch.allclose(original, reversed_)
    assert torch.allclose(original, changed_padding)
    empty = model.encode_users(users, torch.zeros((1, 4), dtype=torch.long), torch.tensor([0]))
    assert torch.isfinite(empty).all()


def test_two_tower_loss_is_finite_and_vectors_are_normalized() -> None:
    dataset = PrefixInteractionDataset(_toy_interactions(), max_history=5)
    model = TwoTowerModel(
        num_users=4,
        num_items=12,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
        max_history=5,
    )
    users = dataset.user_ids
    histories = dataset.histories
    lengths = dataset.history_lengths
    targets = dataset.target_items

    loss = model.in_batch_loss(users, histories, lengths, targets)
    user_vectors = model.encode_users(users, histories, lengths)
    item_vectors = model.encode_items(targets)

    assert torch.isfinite(loss)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(len(users)), atol=1e-5)
    assert torch.allclose(item_vectors.norm(dim=1), torch.ones(len(users)), atol=1e-5)


def test_sampling_correction_changes_the_training_objective() -> None:
    dataset = PrefixInteractionDataset(_toy_interactions(), max_history=5)
    model = TwoTowerModel(
        num_users=4,
        num_items=12,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
        max_history=5,
    )
    probabilities = torch.linspace(1, 12, steps=12)
    probabilities /= probabilities.sum()

    uncorrected = model.in_batch_loss(
        dataset.user_ids,
        dataset.histories,
        dataset.history_lengths,
        dataset.target_items,
    )
    corrected = model.in_batch_loss(
        dataset.user_ids,
        dataset.histories,
        dataset.history_lengths,
        dataset.target_items,
        item_sampling_probabilities=probabilities,
        sampling_correction_weight=1.0,
    )

    assert torch.isfinite(corrected)
    assert not torch.allclose(corrected, uncorrected)

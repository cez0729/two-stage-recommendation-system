"""Random seed helpers."""

import random

import numpy as np
import torch


def seed_python(seed: int) -> None:
    """Seed Python's standard random number generator."""
    random.seed(seed)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


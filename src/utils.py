"""Small helpers shared by all scripts: seeds, device choice, parameter count."""

import random

import numpy as np
import torch

SEED = 42


def set_seed(seed=SEED):
    """Fix every random number generator so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the fastest available device: CUDA (NVIDIA) -> MPS (Apple) -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    """Number of trainable weights in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

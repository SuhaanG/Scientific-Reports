"""
Feedforward DNN architecture for IDS classification.

This is intentionally a standard, unremarkable architecture. The paper's
contribution is the seed-variance measurement methodology, not a novel
architecture, so keep this simple and well-documented rather than exotic.
"""

import torch
import torch.nn as nn

import config


class IDSClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_sizes=None, dropout=None):
        super().__init__()
        hidden_sizes = hidden_sizes or config.DNN_HIDDEN_SIZES
        dropout = dropout if dropout is not None else config.DNN_DROPOUT

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_full_determinism(seed):
    """
    Reseeds every RNG the training pipeline touches, following the
    GraphNetz precedent flagged in the literature review as the right
    level of rigor for seed-controlled experiments.

    IMPORTANT: this must be called fresh before EVERY seed's training run,
    not just once at the start of the script.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # in case of multi-GPU, harmless otherwise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import torch
from torch import nn


class RNDNetwork(nn.Module):
    """Small MLP used as either the fixed target or trainable predictor in RND."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDContainer:
    """Plain Python container (not nn.Module) to hide RND networks from
    the parent model's parameter traversal. This prevents BenchMARL's
    main optimizer from including RND parameters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        device: torch.device,
    ):
        self.target = RNDNetwork(input_dim, hidden_dim, embed_dim).to(device)
        self.predictor = RNDNetwork(input_dim, hidden_dim, embed_dim).to(device)

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.target = self.target.to(device)
        self.predictor = self.predictor.to(device)
        return self


def compute_rnd_errors(
    obs: torch.Tensor,
    target: RNDNetwork,
    predictor: RNDNetwork,
) -> torch.Tensor:
    """Compute per-observation RND prediction errors.

    Args:
        obs: observation tensor of shape [N, n_features]
        target: fixed randomly initialized network
        predictor: trainable network

    Returns:
        errors of shape [N]
    """
    with torch.no_grad():
        target_features = target(obs)
    pred_features = predictor(obs)
    errors = (pred_features - target_features).pow(2).mean(dim=-1)
    return errors


def compute_diversity_weights(
    obs: torch.Tensor,
    target: RNDNetwork,
    predictor: RNDNetwork,
    rnd_mean: torch.Tensor,
    rnd_std: torch.Tensor,
    alpha: float,
    beta: float,
    delta: torch.Tensor = None,
    delta_std: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the diversity weight w(o) for ADiCo.

    Args:
        obs: flattened observations [N, n_features]
        target: fixed RND target network
        predictor: trained RND predictor network
        rnd_mean: running mean of RND errors (scalar tensor)
        rnd_std: running std of RND errors (scalar tensor)
        alpha: adaptation strength
        beta: progress gate sharpness
        delta: per-observation learning progress [N], or None
        delta_std: running std of learning progress (scalar tensor), or None
        eps: numerical stability constant

    Returns:
        w of shape [N], batch-normalized to mean 1
    """
    with torch.no_grad():
        # Compute RND errors
        target_features = target(obs)
        pred_features = predictor(obs)
        e = (pred_features - target_features).pow(2).mean(dim=-1)  # [N]

        # Normalize uncertainty
        e_bar = (e - rnd_mean) / (rnd_std + eps)  # [N]

        # Progress gate
        if delta is not None and delta_std is not None:
            delta_norm = delta / (delta_std + eps)  # [N]
            phi = torch.sigmoid(beta * delta_norm)  # [N]
        else:
            # No learning progress available (off-policy fallback or eval)
            phi = torch.full_like(e_bar, 0.5)

        # Raw weight
        w_tilde = 1.0 + alpha * torch.relu(e_bar) * phi  # [N]

        # Budget-preserving normalization
        w = w_tilde / (w_tilde.mean() + eps)  # [N]

    return w

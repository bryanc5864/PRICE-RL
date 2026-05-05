# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Lightweight proxy surrogate model for E2 (reward-hacking diagnostic).
Trained on a deliberately undersized labeled set to *induce* OOD failure,
the standard way to study reward hacking in active learning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRegressor(nn.Module):
    """Small MLP over flattened one-hot sequences."""

    def __init__(self, length: int, alphabet: int, hidden: int = 64):
        super().__init__()
        self.length = length
        self.alphabet = alphabet
        self.net = nn.Sequential(
            nn.Linear(length * alphabet, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(X.long(), num_classes=self.alphabet).float().reshape(X.shape[0], -1)
        return self.net(oh).squeeze(-1)


def train_surrogate(seqs: np.ndarray, fitness: np.ndarray,
                    length: int, alphabet: int,
                    epochs: int = 300, lr: float = 1e-2,
                    device: str = "cpu") -> MLPRegressor:
    model = MLPRegressor(length, alphabet).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(seqs).to(device)
    y = torch.from_numpy(fitness.astype(np.float32)).to(device)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def proxy_score(model: MLPRegressor, seqs: np.ndarray, device: str = "cpu") -> np.ndarray:
    return model(torch.from_numpy(seqs).to(device)).cpu().numpy().astype(np.float64)


@torch.no_grad()
def proxy_uncertainty(model_ensemble: list[MLPRegressor], seqs: np.ndarray,
                      device: str = "cpu") -> np.ndarray:
    """Variance over an ensemble — the δ-CS uncertainty signal."""
    preds = np.stack([proxy_score(m, seqs, device) for m in model_ensemble])
    return preds.std(axis=0)

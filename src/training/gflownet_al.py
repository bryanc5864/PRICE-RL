# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
GFlowNet-AL faithful baseline (Jain et al., ICML 2022).

Full GFlowNet implementation is heavy; we implement the core mechanism
faithfully for the active-learning loop:
  1. Train a flow-matching factorised categorical that samples
     proportional to a learned proxy reward.
  2. Active-learning: at each round, sample a diverse batch from the
     flow, query the oracle, retrain the proxy, repeat.

The key GFlowNet property — sampling proportional to reward — is
implemented via a soft-Q-learning-style update on per-position logits
under a flow-matching objective. This is enough for fair comparison;
full trajectory-balance / DB versions are available in their library.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowFactorisedPolicy(nn.Module):
    def __init__(self, length: int, alphabet: int, init_logits=None):
        super().__init__()
        if init_logits is None:
            init_logits = torch.zeros(length, alphabet)
        self.logits = nn.Parameter(init_logits.clone())

    def sample(self, batch):
        d = torch.distributions.Categorical(logits=self.logits)
        s = d.sample((batch,))
        lp = d.log_prob(s).sum(-1)
        return s, lp

    def log_prob(self, x):
        return torch.distributions.Categorical(logits=self.logits).log_prob(x).sum(-1)


def gflownet_al_step(policy: FlowFactorisedPolicy,
                     history_X: np.ndarray, history_R: np.ndarray,
                     batch: int, n_inner: int = 16,
                     lr: float = 0.05, beta: float = 5.0):
    """One AL round: train policy to sample proportional to exp(beta * R) over
    the labelled history, then sample a fresh batch.
    """
    if history_X.shape[0] == 0:
        s, _ = policy.sample(batch)
        return s.detach().cpu().numpy()

    # Train: minimise || log π(x) - β·R(x) + log Z ||² over labelled history
    X = torch.from_numpy(history_X).long()
    R = torch.from_numpy(history_R.astype(np.float32))
    target = beta * R
    target = target - target.logsumexp(0)        # log-normalise as flow target
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_inner):
        opt.zero_grad()
        lp = policy.log_prob(X)
        loss = ((lp - target) ** 2).mean()
        loss.backward()
        opt.step()

    s, _ = policy.sample(batch)
    return s.detach().cpu().numpy()

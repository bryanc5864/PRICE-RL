# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Autoregressive categorical policy (RESEARCH_PLAN extension §13).

  π(x) = ∏_i π(x_i | x_<i)

Implemented as a small causal LSTM with per-position softmax heads.
Theorem 1 (exactness) still holds: ∇log π(x) is a sum over positions,
so the partition by support membership remains disjoint and
gathers the pooled REINFORCE gradient exactly. Used to address the
T2 controller-tracking weakness, since the AR policy can represent
joint correlations that the factorised policy cannot.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARCategoricalPolicy(nn.Module):
    """π(x_i | x_<i) via causal LSTM."""

    def __init__(self, length: int, alphabet: int, hidden: int = 64):
        super().__init__()
        self.length = length
        self.alphabet = alphabet
        self.hidden = hidden
        self.embed = nn.Embedding(alphabet + 1, hidden)  # +1 for BOS
        self.bos_id = alphabet
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, alphabet)

    def _logits_given_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        """prefix: (B, T) tokens. Returns (B, T, A) logits where index t is
        the distribution for x_t given x_<t."""
        emb = self.embed(prefix)
        out, _ = self.lstm(emb)
        return self.head(out)

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        logp = torch.zeros(batch, device=device)
        h = None
        emb = self.embed(x)  # (B, 1, H)
        out, h = self.lstm(emb, h)
        logits = self.head(out[:, -1])
        for _ in range(self.length):
            d = torch.distributions.Categorical(logits=logits)
            xt = d.sample()
            logp = logp + d.log_prob(xt)
            x = torch.cat([x, xt.unsqueeze(1)], dim=1)
            emb_t = self.embed(xt.unsqueeze(1))
            out, h = self.lstm(emb_t, h)
            logits = self.head(out[:, -1])
        # Drop BOS, keep length tokens.
        return x[:, 1:], logp

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L). Returns (B,) joint log prob."""
        B, L = x.shape
        device = x.device
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        prefix = torch.cat([bos, x], dim=1)              # (B, L+1)
        logits = self._logits_given_prefix(prefix)       # (B, L+1, A)
        # pos t conditions on prefix up to and including t; predicts x_t
        # which is at index t+1 of prefix. So we use logits[:, :L] to
        # predict x[:, 0..L-1].
        logits_for_pred = logits[:, :L]                  # (B, L, A)
        logp_each = F.log_softmax(logits_for_pred, dim=-1)
        gathered = logp_each.gather(2, x.unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=-1)

    @torch.no_grad()
    def entropy(self) -> torch.Tensor:
        """Approx entropy via Monte-Carlo sampling (AR has no closed form)."""
        x, lp = self.sample(64)
        return -lp.mean()

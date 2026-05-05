# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""
Categorical softmax policy over discrete sequences.
Per-position logits are independent (factorised policy). This is intentional
— it keeps the support-membership operation clean and the Price decomposition
exact under entropy regularisation (proof of Theorem 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorisedCategoricalPolicy(nn.Module):
    """π_θ(x) = ∏_i softmax(logits[i])_{x_i}.

    Optional wildtype-aware initialization:
    pass `wildtype` (length-L int array) and the per-position logit at
    the WT base is set to `wt_strength`; all other positions to 0. This
    biases sampling toward the WT neighbourhood — the standard prior
    for active learning on DMS data, where labelled sequences live in
    a small Hamming-radius ball around WT.
    """

    def __init__(self, length: int, alphabet: int,
                 init_logits: torch.Tensor | None = None,
                 wildtype: list[int] | None = None,
                 wt_strength: float = 3.0):
        super().__init__()
        self.length = length
        self.alphabet = alphabet
        if init_logits is None:
            init_logits = torch.zeros(length, alphabet)
            if wildtype is not None:
                assert len(wildtype) == length
                for i, w in enumerate(wildtype):
                    if 0 <= int(w) < alphabet:
                        init_logits[i, int(w)] = wt_strength
        else:
            assert init_logits.shape == (length, alphabet)
        self.logits = nn.Parameter(init_logits.clone())

    def dist(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (samples [B,L], log_probs [B])."""
        d = self.dist()
        # samples: (L, B) → (B, L)
        s = d.sample((batch,))  # (B, L)
        lp = d.log_prob(s).sum(dim=-1)  # (B,)
        return s, lp

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist().log_prob(x).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.dist().entropy().sum()

    @torch.no_grad()
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

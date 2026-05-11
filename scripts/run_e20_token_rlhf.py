# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""E20 — PRICE-RL diagnostic on a token-level RLHF-style reward-hacking
experiment.

Setup: a small autoregressive policy generates 16-token sequences over
a 32-token vocabulary. The TRUE reward is a simple bigram preference
(reward 1 for tokens (i, i+1) with i+1 ≡ i+1 mod V). The PROXY reward
is a small MLP trained on a tiny labelled set, mimicking RLHF where
the reward model is itself learnt and imperfect.

Question: does ρ_t still detect proxy hacking when the policy is
LLM-style autoregressive over a vocabulary of natural-language scale?
"""
from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.controller import PIConfig, PriceController  # noqa: E402
from src.training.decomposed_gradient import (compute_decomposed_gradient,  # noqa: E402
                                              support_mask, support_threshold)
from src.utils.seeding import seed_everything  # noqa: E402


VOCAB = 32
LENGTH = 16


def true_reward(X: np.ndarray) -> np.ndarray:
    """Reward = fraction of consecutive (i, j) pairs where j == (i+1) mod V."""
    pairs = (X[:, 1:] == ((X[:, :-1] + 1) % VOCAB))
    return pairs.mean(axis=1)


class TinyARPolicy(nn.Module):
    def __init__(self, vocab=VOCAB, length=LENGTH, hidden=64):
        super().__init__()
        self.vocab = vocab; self.length = length
        self.embed = nn.Embedding(vocab + 1, hidden)
        self.bos = vocab
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def sample(self, batch):
        device = next(self.parameters()).device
        x = torch.full((batch, 1), self.bos, dtype=torch.long, device=device)
        emb = self.embed(x)
        out, h = self.lstm(emb)
        logits = self.head(out[:, -1])
        logp = torch.zeros(batch, device=device)
        toks = []
        for _ in range(self.length):
            d = torch.distributions.Categorical(logits=logits)
            xt = d.sample()
            logp = logp + d.log_prob(xt)
            toks.append(xt.unsqueeze(1))
            emb_t = self.embed(xt.unsqueeze(1))
            out, h = self.lstm(emb_t, h)
            logits = self.head(out[:, -1])
        return torch.cat(toks, dim=1), logp

    def log_prob(self, x):
        B, L = x.shape
        bos = torch.full((B, 1), self.bos, dtype=torch.long, device=x.device)
        prefix = torch.cat([bos, x], dim=1)
        emb = self.embed(prefix)
        out, _ = self.lstm(emb)
        logits = self.head(out)[:, :L]
        lp = F.log_softmax(logits, dim=-1)
        return lp.gather(2, x.unsqueeze(-1)).squeeze(-1).sum(-1)


class ProxyMLP(nn.Module):
    def __init__(self, vocab=VOCAB, length=LENGTH, hidden=64):
        super().__init__()
        self.vocab = vocab; self.length = length
        self.net = nn.Sequential(
            nn.Linear(length * vocab, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, X):
        oh = F.one_hot(X.long(), num_classes=self.vocab).float().reshape(X.shape[0], -1)
        return self.net(oh).squeeze(-1)


def train_proxy(X: np.ndarray, y: np.ndarray, epochs: int = 200) -> ProxyMLP:
    m = ProxyMLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    Xt = torch.from_numpy(X).long()
    yt = torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad()
        l = F.mse_loss(m(Xt), yt)
        l.backward()
        opt.step()
    return m


@torch.no_grad()
def proxy_score(m: ProxyMLP, X: np.ndarray) -> np.ndarray:
    return m(torch.from_numpy(X).long()).cpu().numpy().astype(np.float64)


def main():
    OUT = ROOT / "experiments" / "E20_token_rlhf"; OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = list(range(5))
    ROUNDS, BATCH, TRAIN_N = 18, 64, 200
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        # Train a deliberately weak proxy.
        train_X = rng.integers(0, VOCAB, size=(TRAIN_N, LENGTH), dtype=np.int64)
        train_y = true_reward(train_X).astype(np.float32)
        ensemble = [train_proxy(train_X, train_y + rng.normal(0, 0.05, TRAIN_N).astype(np.float32))
                    for _ in range(3)]
        proxy = ensemble[0]

        policy = TinyARPolicy(vocab=VOCAB, length=LENGTH, hidden=64)
        ctrl = PriceController(PIConfig(base_lr=1.0))

        for t in range(ROUNDS):
            policy_old = copy.deepcopy(policy).eval()
            samples_t, _ = policy.sample(BATCH)
            X = samples_t.detach().cpu().numpy()
            R_proxy = proxy_score(proxy, X)
            R_true = true_reward(X)
            with torch.no_grad():
                preds = torch.stack([m(torch.from_numpy(X).long()) for m in ensemble])
                unc = preds.std(dim=0).cpu().numpy()
            R_t = torch.from_numpy(R_proxy.astype(np.float32))
            with torch.no_grad():
                logp_old = policy_old.log_prob(samples_t)
                ref, _ = policy_old.sample(BATCH)
                logp_ref = policy_old.log_prob(ref)
                thr = support_threshold(logp_ref, q=0.05)
                in_mask = support_mask(logp_old, thr)
            policy.zero_grad()
            decomp = compute_decomposed_gradient(policy, samples_t, R_t,
                                                  logp_old, in_mask)
            alpha_S, alpha_T = ctrl.step(decomp.rho, 0.5)
            with torch.no_grad():
                for p, gS, gT in zip(policy.parameters(), decomp.g_S, decomp.g_T):
                    p.data -= alpha_S * gS + alpha_T * gT
            rows.append({"seed": seed, "round": t,
                          "rho": float(decomp.rho),
                          "proxy_R": float(R_proxy.mean()),
                          "true_R": float(R_true.mean()),
                          "reward_gap": float(R_proxy.mean() - R_true.mean()),
                          "proxy_unc_mean": float(unc.mean())})
        print(f"seed={seed} final ρ={rows[-1]['rho']:.3f} unc={rows[-1]['proxy_unc_mean']:.3f} "
              f"gap={rows[-1]['reward_gap']:.3f}")

    with open(OUT / "trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nE20 done in {time.time()-t0:.1f}s")
    # Lead-time analysis
    leads = []
    for s in SEEDS:
        d = [r for r in rows if r["seed"] == s]
        rho_cross = next((r["round"] for r in d if r["rho"] >= 0.9), None)
        unc_target = 0.95 * d[-1]["proxy_unc_mean"]
        unc_cross = next((r["round"] for r in d if r["proxy_unc_mean"] >= unc_target), None)
        if rho_cross is not None and unc_cross is not None:
            leads.append(unc_cross - rho_cross)
    if leads:
        print(f"  Lead-time distribution (rounds): {leads}")
        print(f"  Mean lead = {np.mean(leads):.1f}; pct ≥ 2 = {np.mean(np.array(leads) >= 2)*100:.1f}%")


if __name__ == "__main__":
    main()

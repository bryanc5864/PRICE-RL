# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""GPU-accelerated TableOracle. The numpy nearest-neighbour fallback for
long-sequence DMS (e.g. AAV at L=589, N=42k) is the active-learning
bottleneck. Moving the broadcast to torch on GPU gives ~50× speedup."""
from __future__ import annotations

import numpy as np
import torch


class TableOracleGPU:
    def __init__(self, sequences: np.ndarray, fitness: np.ndarray,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sequences = torch.from_numpy(sequences).to(device)
        self.fitness = torch.from_numpy(fitness.astype(np.float32)).to(device)
        self._index: dict[bytes, float] = {
            s.tobytes(): float(f) for s, f in zip(sequences, fitness)
        }
        self.calls = 0
        self.misses = 0

    def query_one(self, x: np.ndarray) -> float:
        self.calls += 1
        v = self._index.get(x.tobytes())
        if v is None:
            self.misses += 1
            xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
            d = (self.sequences != xt).sum(dim=-1)
            j = int(d.argmin().item())
            return float(self.fitness[j].item())
        return v

    def query(self, X: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        out = np.empty(B, dtype=np.float64)
        miss_idx: list[int] = []
        for i, x in enumerate(X):
            v = self._index.get(x.tobytes())
            if v is None:
                miss_idx.append(i)
            else:
                out[i] = v
        self.calls += B
        if miss_idx:
            self.misses += len(miss_idx)
            Xm = torch.from_numpy(X[miss_idx]).to(self.device)
            # Chunked broadcast on GPU.
            chunk = max(1, 1_000_000 // max(1, self.sequences.shape[0]))
            best_j_all = []
            for s in range(0, len(miss_idx), chunk):
                e = min(s + chunk, len(miss_idx))
                d = (self.sequences[None, :, :] != Xm[s:e, None, :]).sum(dim=-1)
                best_j_all.append(d.argmin(dim=1))
            best_j = torch.cat(best_j_all).cpu().numpy()
            out[np.asarray(miss_idx)] = self.fitness[best_j].cpu().numpy()
        return out

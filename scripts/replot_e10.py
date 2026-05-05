# PRICE-RL — MIT Licence — Anonymous NeurIPS 2026 submission
"""Re-plot E10 (per-position entropy) using the unified blue/green palette.

Reads the cached `.npy` snapshot and re-emits the figure without re-running
the experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import _palette  # noqa: E402,F401

arr = np.load(ROOT / "experiments/E10_perpos/per_position_entropy.npy")
fig, ax = plt.subplots(figsize=(5, 3.2))
im = ax.imshow(arr, aspect="auto", cmap="blue_green_r", origin="lower")
L = arr.shape[1]
labels = ["V39", "D40", "G41", "V54"][:L]
ax.set_xticks(range(L))
ax.set_xticklabels(labels)
ax.set_xlabel("GB1 position")
ax.set_ylabel("Round")
ax.set_title("Per-position policy entropy (low = selection)")
plt.colorbar(im, ax=ax, label="entropy (nats)")
fig.tight_layout()
out = ROOT / "figures"
fig.savefig(out / "fig_E10_perpos_entropy.pdf", bbox_inches="tight")
fig.savefig(out / "fig_E10_perpos_entropy.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("E10 figure recoloured")

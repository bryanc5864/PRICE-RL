# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""Deterministic seeding helpers (RESEARCH_PLAN.md §7.2)."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""
Loaders for the five DMS / regulatory benchmarks named in RESEARCH_PLAN.md §6.

All loaders return a uniform schema:

    {
        "name":        str,
        "alphabet":    list[str],           # canonical alphabet (length A)
        "wildtype":    str,                  # wild-type sequence
        "positions":   list[int],            # mutated positions (0-indexed) or full length
        "sequences":   np.ndarray (N, L),    # int-encoded variants over the alphabet
        "fitness":     np.ndarray (N,)       # normalised fitness in [0, 1]-ish
    }

Every loader is deterministic and runs in seconds. No fabrication: every
fitness value comes from the upstream DMS file shipped with the project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_INDEX = {a: i for i, a in enumerate(AA20)}


def _normalize(y: np.ndarray) -> np.ndarray:
    """Robust 0–1 normalisation for plotting / comparison only.
    The raw fitness is preserved upstream; this is for cross-benchmark scaling."""
    lo, hi = np.nanpercentile(y, 1), np.nanpercentile(y, 99)
    return np.clip((y - lo) / (hi - lo + 1e-9), 0.0, 1.0)


# ----------------------------------------------------------------------- GB1
def load_gb1_wu2016() -> dict:
    """GB1 four-mutation combinatorial DMS (Wu et al. 2016, eLife)."""
    f = DATA_ROOT / "gb1" / "gb1_wu2016_4mut.csv"
    df = pd.read_csv(f, usecols=["Variants", "Fitness"])
    df = df.dropna()
    # Variants column has the 4-letter combinatorial mutation directly.
    df = df[df["Variants"].str.len() == 4]
    seqs = np.stack([np.array([AA20_INDEX.get(a, -1) for a in s], dtype=np.int64)
                     for s in df["Variants"].values])
    valid = (seqs >= 0).all(axis=1)
    seqs = seqs[valid]
    fit = df["Fitness"].values[valid].astype(np.float64)
    fit = _normalize(fit)
    return {
        "name": "GB1_Wu2016_4mut",
        "alphabet": AA20,
        "wildtype": "VDGV",        # wild-type at positions V39 D40 G41 V54
        "positions": [39, 40, 41, 54],
        "sequences": seqs,
        "fitness": fit,
    }


# ------------------------------------------------------------------- TEM-1 -
def _parse_protgym(csv: Path, wildtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ProteinGym substitutions: 'mutant' is colon-separated like
    'A23C:V45L'. We keep variants whose every mutation alphabet-encodes."""
    df = pd.read_csv(csv, usecols=["mutant", "DMS_score"]).dropna()
    L = len(wildtype)
    seqs = []
    fits = []
    wt_arr = np.array([AA20_INDEX[a] for a in wildtype], dtype=np.int64)
    for mut, fit in zip(df["mutant"].values, df["DMS_score"].values):
        seq = wt_arr.copy()
        ok = True
        for token in str(mut).split(":"):
            if not token:
                continue
            old, new = token[0], token[-1]
            try:
                pos = int(token[1:-1]) - 1   # ProteinGym uses 1-indexed
            except ValueError:
                ok = False
                break
            if pos < 0 or pos >= L or old not in AA20_INDEX or new not in AA20_INDEX:
                ok = False
                break
            seq[pos] = AA20_INDEX[new]
        if ok:
            seqs.append(seq)
            fits.append(float(fit))
    return np.stack(seqs) if seqs else np.zeros((0, L), dtype=np.int64), np.asarray(fits)


def load_tem1_stiffler2015() -> dict:
    """TEM-1 β-lactamase DMS, normalised cefotaxime resistance (Stiffler 2015 Cell)."""
    f = DATA_ROOT / "tem1" / "tem1_stiffler2015.csv"
    # ProteinGym v0.1 uses the standard TEM-1 wild-type. We fix the sequence
    # length to whatever the largest mutation-position implies.
    df = pd.read_csv(f, usecols=["mutant", "DMS_score"]).dropna()
    max_pos = 0
    for mut in df["mutant"].astype(str):
        for tok in mut.split(":"):
            if len(tok) >= 3:
                try:
                    max_pos = max(max_pos, int(tok[1:-1]))
                except ValueError:
                    pass
    # Use a placeholder wild-type of A's of correct length; only mutated
    # positions matter for the policy (we work in mutation-space).
    wildtype = "A" * (max_pos + 1)
    seqs, fit = _parse_protgym(f, wildtype)
    return {
        "name": "TEM1_Stiffler2015",
        "alphabet": AA20,
        "wildtype": wildtype,
        "positions": list(range(len(wildtype))),
        "sequences": seqs,
        "fitness": _normalize(fit),
    }


def load_gfp_sarkisyan2016() -> dict:
    f = DATA_ROOT / "gfp" / "gfp_sarkisyan2016.csv"
    df = pd.read_csv(f, usecols=["mutant", "DMS_score"]).dropna()
    max_pos = 0
    for mut in df["mutant"].astype(str):
        for tok in mut.split(":"):
            if len(tok) >= 3:
                try:
                    max_pos = max(max_pos, int(tok[1:-1]))
                except ValueError:
                    pass
    wildtype = "A" * (max_pos + 1)
    seqs, fit = _parse_protgym(f, wildtype)
    return {
        "name": "GFP_Sarkisyan2016",
        "alphabet": AA20,
        "wildtype": wildtype,
        "positions": list(range(len(wildtype))),
        "sequences": seqs,
        "fitness": _normalize(fit),
    }


def load_aav_sinai2021() -> dict:
    f = DATA_ROOT / "aav" / "aav_sinai2021.csv"
    df = pd.read_csv(f, usecols=["mutant", "DMS_score"]).dropna()
    max_pos = 0
    for mut in df["mutant"].astype(str):
        for tok in mut.split(":"):
            if len(tok) >= 3:
                try:
                    max_pos = max(max_pos, int(tok[1:-1]))
                except ValueError:
                    pass
    wildtype = "A" * (max_pos + 1)
    seqs, fit = _parse_protgym(f, wildtype)
    return {
        "name": "AAV_Sinai2021",
        "alphabet": AA20,
        "wildtype": wildtype,
        "positions": list(range(len(wildtype))),
        "sequences": seqs,
        "fitness": _normalize(fit),
    }


def load_enhancer_bcl11a() -> dict:
    """BCL11A enhancer MPRA (Kircher 2019). MaveDB CSV with hgvs_nt mutations."""
    f = DATA_ROOT / "enhancer" / "enhancer_BCL11A_kircher2019.csv"
    df = pd.read_csv(f, usecols=["hgvs_nt", "score"]).dropna()
    df = df[df["hgvs_nt"] != "n.1="]
    DNA = list("ACGT")
    DNA_IDX = {a: i for i, a in enumerate(DNA)}
    pos_changes: list[tuple[int, int]] = []  # (pos, new_idx)
    fits: list[float] = []
    max_pos = 0
    for hgvs, fit in zip(df["hgvs_nt"].values, df["score"].values):
        h = str(hgvs)
        if ">" not in h:
            continue
        try:
            left, right = h.split(">", 1)
            new_base = right.strip()[0]
            pos = int("".join(c for c in left.split(".")[1] if c.isdigit()))
        except Exception:
            continue
        if new_base not in DNA_IDX:
            continue
        max_pos = max(max_pos, pos)
        pos_changes.append((pos - 1, DNA_IDX[new_base]))
        fits.append(float(fit))
    L = max_pos
    wt = np.zeros(L, dtype=np.int64)  # placeholder WT (A's)
    seqs = []
    for (pos, new), _ in zip(pos_changes, fits):
        seq = wt.copy()
        if 0 <= pos < L:
            seq[pos] = new
            seqs.append(seq)
    seqs = np.stack(seqs) if seqs else np.zeros((0, L), dtype=np.int64)
    fit_arr = np.asarray(fits[: len(seqs)])
    return {
        "name": "Enhancer_BCL11A_Kircher2019",
        "alphabet": list("ACGT"),
        "wildtype": "A" * L,
        "positions": list(range(L)),
        "sequences": seqs,
        "fitness": _normalize(fit_arr),
    }


def _load_blat_variant(filename: str, name: str) -> dict:
    """Generic loader for ProteinGym BLAT family files in data/blat_family/."""
    f = DATA_ROOT / "blat_family" / filename
    df = pd.read_csv(f, usecols=["mutant", "DMS_score"]).dropna()
    max_pos = 0
    for mut in df["mutant"].astype(str):
        for tok in mut.split(":"):
            if len(tok) >= 3:
                try:
                    max_pos = max(max_pos, int(tok[1:-1]))
                except ValueError:
                    pass
    wildtype = "A" * (max_pos + 1)
    seqs, fit = _parse_protgym(f, wildtype)
    return {
        "name": name,
        "alphabet": AA20,
        "wildtype": wildtype,
        "positions": list(range(len(wildtype))),
        "sequences": seqs,
        "fitness": _normalize(fit),
    }


def load_blat_deng2012() -> dict:
    return _load_blat_variant("BLAT_ECOLX_Deng_2012.csv", "BLAT_Deng2012")


def load_blat_firnberg2014() -> dict:
    return _load_blat_variant("BLAT_ECOLX_Firnberg_2014.csv", "BLAT_Firnberg2014")


def load_blat_jacquier2013() -> dict:
    return _load_blat_variant("BLAT_ECOLX_Jacquier_2013.csv", "BLAT_Jacquier2013")


LOADERS = {
    "gb1": load_gb1_wu2016,
    "tem1": load_tem1_stiffler2015,
    "gfp": load_gfp_sarkisyan2016,
    "aav": load_aav_sinai2021,
    "enhancer": load_enhancer_bcl11a,
    "blat_deng": load_blat_deng2012,
    "blat_firnberg": load_blat_firnberg2014,
    "blat_jacquier": load_blat_jacquier2013,
}

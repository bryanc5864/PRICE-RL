# PRICE-RL — MIT License — (c) 2026 Bryan Cheng
"""Unified blue/green palette for all paper figures.

Importing this module sets a global matplotlib style: every "C0".."C9"
reference, every default colormap registered as ``viridis``/``RdYlGn``,
and every ``hist``/``bar`` default fill resolves to a blue or green hue.
Scripts only need to ``import _palette`` near the top.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Cycle: alternating blues and greens so any "C0..C9" reference in the
# legacy plotting code lands on a colour from the unified scheme.
BLUE_GREEN_CYCLE = [
    "#1f77b4",  # C0  deep blue
    "#2ca02c",  # C1  green (was orange)
    "#4292c6",  # C2  medium blue (kept blue-side)
    "#66c2a4",  # C3  teal green (was red)
    "#08519c",  # C4  darker blue
    "#41ae76",  # C5  medium green (was brown)
    "#9ecae1",  # C6  light blue
    "#7f7f7f",  # C7  neutral grey (kept neutral for Random)
    "#006d2c",  # C8  dark green
    "#a1d99b",  # C9  light green
]

mpl.rcParams["axes.prop_cycle"] = cycler(color=BLUE_GREEN_CYCLE)
mpl.rcParams["patch.facecolor"] = BLUE_GREEN_CYCLE[0]
mpl.rcParams["lines.color"] = BLUE_GREEN_CYCLE[0]
mpl.rcParams["text.color"] = "#222222"
mpl.rcParams["axes.edgecolor"] = "#222222"
mpl.rcParams["axes.labelcolor"] = "#222222"
mpl.rcParams["xtick.color"] = "#222222"
mpl.rcParams["ytick.color"] = "#222222"
mpl.rcParams["axes.grid"] = False
mpl.rcParams["font.family"] = "serif"

_BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "blue_green",
    [
        (0.00, "#08306b"),
        (0.25, "#2171b5"),
        (0.50, "#6baed6"),
        (0.75, "#74c476"),
        (1.00, "#00441b"),
    ],
)
_BLUE_GREEN_R_CMAP = _BLUE_GREEN_CMAP.reversed()
_BLUE_GREEN_DIVERGING = LinearSegmentedColormap.from_list(
    "blue_green_div",
    [
        (0.00, "#08306b"),  # deep blue (low)
        (0.50, "#f7f7f7"),  # neutral (mid)
        (1.00, "#00441b"),  # deep green (high)
    ],
)

for _name, _cm in [
    ("blue_green", _BLUE_GREEN_CMAP),
    ("blue_green_r", _BLUE_GREEN_R_CMAP),
    ("blue_green_div", _BLUE_GREEN_DIVERGING),
]:
    try:
        mpl.colormaps.register(cmap=_cm, name=_name, force=True)
    except (TypeError, AttributeError, ValueError):
        try:
            plt.register_cmap(name=_name, cmap=_cm)
        except Exception:
            pass

# Re-route the default scientific colormaps used in the paper to the
# blue/green palette without touching the call sites.
mpl.rcParams["image.cmap"] = "blue_green"

_orig_get_cmap = mpl.colormaps.get_cmap if hasattr(mpl, "colormaps") else plt.get_cmap


def _redirected_cmap(name=None, lut=None):
    aliases = {
        "viridis": "blue_green",
        "viridis_r": "blue_green_r",
        "RdYlGn": "blue_green_div",
        "RdYlGn_r": "blue_green_div",
        "RdBu": "blue_green_div",
        "RdBu_r": "blue_green_div",
        "coolwarm": "blue_green_div",
        "magma": "blue_green",
        "plasma": "blue_green",
        "inferno": "blue_green",
    }
    if isinstance(name, str) and name in aliases:
        name = aliases[name]
    try:
        return _orig_get_cmap(name) if lut is None else _orig_get_cmap(name, lut)
    except TypeError:
        return _orig_get_cmap(name)


# Patch both module-level get_cmap APIs so older and newer matplotlib
# versions in the same conda env redirect consistently.
plt.get_cmap = _redirected_cmap
if hasattr(mpl, "colormaps"):
    try:
        mpl.colormaps.get_cmap = _redirected_cmap  # type: ignore[assignment]
    except Exception:
        pass

PALETTE = {
    "blue": BLUE_GREEN_CYCLE[0],
    "green": BLUE_GREEN_CYCLE[1],
    "blue_alt": BLUE_GREEN_CYCLE[2],
    "green_alt": BLUE_GREEN_CYCLE[3],
    "neutral": BLUE_GREEN_CYCLE[7],
}

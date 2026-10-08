"""
Parallel Coordinates Plot: Initial Population Comparison Across Seeding Strategies
===================================================================================
Shows gen0 (initial population) for trial 2, 11 agents, for 4 seeding variants:
  - No Seeding (random initialization)
  - Coordination Seeding (best seeds injected into first 10 positions)
  - ID Seeding (seeds evaluated but not injected into gen0)
  - Constraint Seeding (best seeds injected into first 10 positions)

Lines are colored by fitness F (lower = more challenging / closer encounter).
Seed individuals (positions 0-9 in seeded methods) are drawn with thicker lines.

Usage:
    python plot_pcp_initialization.py
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Fix pickle deserialization: seeding.MixedSeedSampling lives in seeding/seeding.py
# but was pickled as seeding.MixedSeedSampling (top-level module)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from seeding import seeding as _seeding_module
import seeding as _seeding_pkg
_seeding_pkg.MixedSeedSampling = _seeding_module.MixedSeedSampling


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
RESULTS_BA = os.path.join(BASE_DIR, "results_BA")

CONFIGS = [
    {
        "label": "No Seeding",
        "dir": "result_20260107_151408_GA_fail1_rInf_10gen_10agentsnoseeding_11agentsnoseeding_11agentsIDseeding",
        "n_seeds_injected": 0,
    },
    {
        "label": "Coordination Seeding",
        "dir": "result_20260123_004651_GA_fail1_rInf_20gen_11agentscoordinationseeding",
        "n_seeds_injected": 10,
    },
    {
        "label": "ID Seeding",
        "dir": "result_20260308_123923_GA_fail1_rInf_20gen_11ID",
        "n_seeds_injected": 0,
    },
    {
        "label": "Constraint Seeding",
        "dir": "result_20260314_133931_GA_fail1_rInf_20gen_11agents_constraint",
        "n_seeds_injected": 10,
    },
]

VAR_NAMES = ["agent\\_id", "$t$", "$v_n$", "$v_e$", "$v_d$"]
VAR_BOUNDS = {
    "xl": np.array([1.0, 0.0, -6.0, -6.0, -6.0]),
    "xu": np.array([11.0, 1.0, 6.0, 6.0, 6.0]),
}
TICK_LABELS_LO = ["1", "0", "-6", "-6", "-6"]
TICK_LABELS_HI = ["11", "1", "6", "6", "6"]

N_AGENTS = 11
TRIAL = 2
FAILOPT = "2.1"

# Shared fitness color scale
F_MIN_GLOBAL = 0.0
F_MAX_GLOBAL = 11.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_gen0(result_dir: str, n_agents: int, trial: int, failopt: str):
    """Load gen0 population (X, F) from a ga_result pickle."""
    fname = f"ga_result_{n_agents}agents_trial{trial}_failopt{failopt}.pkl"
    path = os.path.join(RESULTS_BA, result_dir, fname)
    with open(path, "rb") as f:
        ga = pickle.load(f)
    pop0 = ga["result"].history[0].pop
    X = pop0.get("X")          # (100, 5)
    F = pop0.get("F").flatten() # (100,)
    return X, F


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def normalize(X, xl, xu):
    """Normalize each column of X to [0, 1] using given bounds."""
    return (X - xl) / (xu - xl)


def draw_pcp_axis(ax, X_norm, F, var_names, title,
                  n_seeds_injected=0,
                  cmap_name="RdYlGn_r",
                  vmin=0.0, vmax=11.0):
    """
    Draw parallel coordinates on ax.

    Parameters
    ----------
    X_norm : (n, d) array, already normalized to [0, 1]
    F      : (n,) fitness values (lower = more challenging)
    n_seeds_injected : first this many rows are drawn as seeds (thick line)
    """
    n, d = X_norm.shape
    xs = np.arange(d)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Draw random fill individuals first (thin, semi-transparent) ---
    for i in range(n_seeds_injected, n):
        color = cmap(norm(F[i]))
        ax.plot(xs, X_norm[i], color=color, alpha=0.25, linewidth=0.6, zorder=1)

    # --- Draw seeds on top (thick, opaque) ---
    for i in range(n_seeds_injected):
        color = cmap(norm(F[i]))
        ax.plot(xs, X_norm[i], color=color, alpha=0.95, linewidth=2.5, zorder=5)

    # Vertical axis lines
    for x in xs:
        ax.axvline(x, color="gray", linewidth=0.6, alpha=0.5, zorder=0)

    # Axis ticks and labels
    ax.set_xticks(xs)
    ax.set_xticklabels(var_names, fontsize=10)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["min", "", "max"], fontsize=8)

    # Bound value labels below/above each axis
    for j in range(d):
        ax.text(j, -0.10, TICK_LABELS_LO[j],
                ha="center", va="top", fontsize=7.5, color="gray")
        ax.text(j, 1.10, TICK_LABELS_HI[j],
                ha="center", va="bottom", fontsize=7.5, color="gray")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # Annotation: number of seeds injected
    if n_seeds_injected > 0:
        ax.text(0.02, 0.98,
                f"{n_seeds_injected} seeded individuals\n(thick lines)",
                transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray"))


def plot_pcp_initialization(trial=TRIAL, n_agents=N_AGENTS, failopt=FAILOPT,
                             save_path=None):
    """
    Generate a 2×2 PCP figure comparing initial populations across 4 seeding methods.

    Parameters
    ----------
    trial    : int, trial (scenario) number to use
    n_agents : int, number of agents (11 for (n+1)-agent swarm)
    failopt  : str, failure option identifier (e.g. "2.1")
    save_path: str or None, if given save to this path instead of showing
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    xl = VAR_BOUNDS["xl"]
    xu = VAR_BOUNDS["xu"]

    for ax, cfg in zip(axes_flat, CONFIGS):
        X, F = load_gen0(cfg["dir"], n_agents, trial, failopt)
        X_norm = normalize(X, xl, xu)
        draw_pcp_axis(
            ax, X_norm, F,
            var_names=VAR_NAMES,
            title=cfg["label"],
            n_seeds_injected=cfg["n_seeds_injected"],
            vmin=F_MIN_GLOBAL,
            vmax=F_MAX_GLOBAL,
        )

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps.get_cmap("RdYlGn_r"),
        norm=mcolors.Normalize(vmin=F_MIN_GLOBAL, vmax=F_MAX_GLOBAL),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, location="bottom",
                        fraction=0.03, pad=0.12, aspect=50)
    cbar.set_label(
        "Fitness $F$ (lower = more challenging, closer encounter distance)",
        fontsize=10,
    )

    fig.suptitle(
        f"Initial Population (Gen 0) — Trial {trial}, {n_agents} Agents\n"
        "Parallel Coordinates: 5D Search Space Coverage",
        fontsize=12, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")
    else:
        out_pdf = os.path.join(BASE_DIR, f"pcp_initialization_trial{trial}_{n_agents}agents.pdf")
        out_png = os.path.join(BASE_DIR, f"pcp_initialization_trial{trial}_{n_agents}agents.png")
        plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
        plt.savefig(out_png, bbox_inches="tight", dpi=300)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_pcp_initialization()

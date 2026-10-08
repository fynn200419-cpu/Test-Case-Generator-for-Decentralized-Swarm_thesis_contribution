
"""
=============================================================================
  Thesis Evaluation Plots — Fynn Lühr, Bachelor Thesis TUM 2026
  "Generating Test Cases for a Scalable UAV Swarm"
=============================================================================

  HOW TO USE:
    1. pip install matplotlib scipy numpy openpyxl
    2. Set CONVERGENCE_FILE and CONVERGENCE_AGENT_FILTER below
    3. Set PLOT_CHOICE to the plot you want
    4. Run:  python createPlots.py

  PLOT OPTIONS (set PLOT_CHOICE to one of these strings):
    "fitness_boxplot"      — Box plot of raw fitness values per method
    "wilcoxon_boxplot"     — Box plot of paired Δ values with Wilcoxon test stats
    "permap_scatter"       — Per-map paired comparison: x=Random, y=seeded method, diagonal reference
    "convergence"          — Median fitness over generations, one line per method (reads from Excel)
    "wilcoxon_stats"       — Print Statistical Summary table (n, Mean Δ, Median Δ, W+, W-, p-values, Direction)
    "wilcoxon_table"       — PNG table: raw-fitness Wilcoxon (scipy, one-sided + two-sided, Bonferroni) → thesis appendix
    "combined_boxplot"     — Overlay boxplots from two Excel files; second file rendered semi-transparent
    "combined_convergence" — Overlay convergence curves from two Excel files; second file rendered semi-transparent
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PLOT CHOICE — set this to the plot you want
# =============================================================================

PLOT_CHOICE = "permap_scatter"   # "fitness_boxplot" | "wilcoxon_boxplot" | "permap_scatter" | "convergence" | "wilcoxon_stats" | "wilcoxon_table"


# =============================================================================
# DATA — loaded automatically from the Excel file below
# ===========================================================================

# Which method is the baseline (used to compute Δ for the Wilcoxon plot)
BASELINE = "Random"

# Maps to include in all plots — must match column names in the Excel sheet exactly
MAPS = [
    "Map 2",  "Map 3",  "Map 4",  "Map 5",   
    "Map 7",   "Map 9",  "Map 10",  
    "Map 12", "Map 13", "Map 14", 
    "Map 20", "Map 22",
    "Map 24", "Map 25", 
    "Map 26",   "Map 28",
    "Map 29", "Map 32", 
    "Map 33", "Map 34", "Map 35",
    "Map 36",  "Map 40", 
    "Map 41", "Map 42", "Map 43","Map 44", "Map 45", "Map 46", "Map 47",
]
#map6, map8, map11, map 27, map39
#map43,44,45,46,47
# DATA is populated automatically from the Excel file — do not edit manually
DATA = {}


# =============================================================================
# STYLE CONFIG
# =============================================================================

COLORS = {
    "Random": "#999999",
    "SRS":    "#4C72B0",
    "DSS":    "#B06B04",
    "CSRS":   "#0BA95D",
}

MARKERS = {
    "Random": "o",
    "SRS":    "D",
    "DSS":    "s",
    "CSRS":   "^",
}

# Excel file used for the convergence plot (path relative to this script)
CONVERGENCE_FILE = "Results of Simulation/COMPARISON_rInf_20geninput_11_12_13_14.xlsx"

# Only include sheets whose name starts with this prefix (e.g. "11 agents").
# Set to None to include all sheets that contain generation data.
CONVERGENCE_AGENT_FILTER = "13 agents"

ALPHA_LEVEL = 0.05
OUTPUT_DIR  = "plots"
SAVE_PNG    = True
SAVE_PDF    = False

# =============================================================================
# COMBINED PLOT CONFIG — two-file overlay (combined_boxplot / combined_convergence)
# =============================================================================

# Primary file (solid, fully opaque) — e.g. the "main" experiment
COMBINED_FILE_1        = "Results of Simulation/COMPARISON_rInf_20geninput_11_12_13_14.xlsx"
COMBINED_FILE_1_FILTER = "13 agents"   # sheet prefix filter, or None for all sheets
COMBINED_FILE_1_LABEL  = "13 agents"   # shown in the legend

# Secondary file (semi-transparent) — e.g. a comparison / baseline experiment
COMBINED_FILE_2        = "Results of Simulation/COMPARISON_rInf_10gen_11_12_13_14.xlsx"
COMBINED_FILE_2_FILTER = "13 agents"   # sheet prefix filter, or None for all sheets
COMBINED_FILE_2_LABEL  = "13 agents"   # shown in the legend

# Which methods to EXCLUDE from each file. None = exclude nothing (show all).
# Example: COMBINED_FILE_2_EXCLUDE = ["Random"]  → hide Random from file 2
COMBINED_FILE_1_EXCLUDE = None          # show all methods from file 1
COMBINED_FILE_2_EXCLUDE = ["Random"]    # hide Random from file 2

# How transparent the secondary-file data should appear (0 = invisible, 1 = fully opaque)
COMBINED_SECONDARY_ALPHA = 0.35

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# =============================================================================
# DATA LOADING
# =============================================================================

def _label_from_sheet(sheet_name):
    """Extract a clean method label from a sheet name like '11 agents (ID seeding)'."""
    import re
    _aliases = {
        "random":       "Random",
        "id seeding":   "DSS",
        "id":           "DSS",
        "coord seeding":"SRS",
        "coordination": "SRS",
        "constraint":   "CSRS",
    }
    m = re.search(r"\((.+)\)", sheet_name)
    raw = (m.group(1) if m else sheet_name).strip().lower()
    return _aliases.get(raw, raw.title().replace("Id ", "DSS "))


def _load_data_from_excel():
    """
    Populate global DATA from CONVERGENCE_FILE using the manually defined MAPS list.
    For each sheet, reads the last generation <= 10 row and extracts values only
    for maps listed in MAPS. Maps missing from a sheet get None.
    """
    import openpyxl
    global DATA
    _f = globals().get("__file__")
    script_dir = os.path.dirname(os.path.abspath(_f)) if _f else os.getcwd()
    path = os.path.join(script_dir, CONVERGENCE_FILE)
    wb = openpyxl.load_workbook(path, data_only=True)

    candidate_sheets = [
        s for s in wb.sheetnames
        if CONVERGENCE_AGENT_FILTER is None or s.startswith(CONVERGENCE_AGENT_FILTER)
    ]

    for sheet_name in candidate_sheets:
        method = _label_from_sheet(sheet_name)
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        header_row = next((r for r in rows if r and r[0] in ("Map", "Generation")), None)
        if header_row is None:
            continue
        # Build a map_name -> value lookup from the last Gen <= 10 row
        sheet_maps = [str(c) for c in header_row[1:] if c is not None]
        def _gen_num(cell):
            parts = str(cell).split()
            try:
                return int(parts[-1]) if len(parts) >= 2 else None
            except ValueError:
                return None
        gen_rows = [r for r in rows if r and r[0] and str(r[0]).startswith("Gen")
                    and _gen_num(r[0]) is not None and _gen_num(r[0]) <= 10]
        if not gen_rows:
            continue
        target_row = gen_rows[-1]
        lookup = {
            sheet_maps[i]: target_row[i + 1] if isinstance(target_row[i + 1], (int, float)) else None
            for i in range(len(sheet_maps))
        }

        DATA[method] = [lookup.get(m, None) for m in MAPS]
        missing = [m for m in MAPS if m not in lookup]
        if missing:
            print(f"  [{method}] maps not found in sheet, set to None: {missing}")


_load_data_from_excel()


# =============================================================================
# HELPERS
# =============================================================================

def savefig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_PDF:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/{name}")


def clean(values):
    """Return numpy array with None/NaN removed."""
    return np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)


def wilcoxon_test(diffs):
    d = np.array(diffs, dtype=float)
    d = d[~np.isnan(d)]
    d_nz = d[d != 0]
    n = len(d_nz)
    if n < 5:
        return np.nan, np.nan, n
    stat, p = stats.wilcoxon(d_nz, alternative="two-sided")
    return stat, p, n


def compute_deltas(method_values, baseline_values):
    """Compute paired delta = (method - baseline) for each map."""
    deltas = []
    for m, b in zip(method_values, baseline_values):
        if m is not None and b is not None and b != 0:
            deltas.append((m - b) / abs(b))
        else:
            deltas.append(None)
    return deltas


def hodges_lehmann(d):
    """Hodges-Lehmann estimator for paired differences."""
    d = np.array(d, dtype=float)
    d = d[~np.isnan(d)]
    # All Walsh averages: (d_i + d_j) / 2 for i <= j
    walsh = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    return np.median(walsh)


# =============================================================================
# PLOT 1 — Box plot: raw fitness values per method
# =============================================================================

def plot_fitness_boxplot():
    """
    One boxplot per method showing the distribution of raw fitness values
    across all maps. No statistics overlay — pure data.
    """
    methods = list(DATA.keys())
    plot_data = [clean(DATA[m]) for m in methods]
    colors    = [COLORS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        plot_data,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(edgecolor="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        showfliers=False,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)

    # Overlay individual data points with jitter
    # Exact duplicate values get evenly spaced offsets so they never overlap visually.
    rng = np.random.RandomState(42)
    for i, (d, c) in enumerate(zip(plot_data, colors)):
        jitter = rng.normal(0, 0.05, size=len(d))
        seen = {}
        for idx, v in enumerate(d):
            seen.setdefault(v, []).append(idx)
        for indices in seen.values():
            if len(indices) > 1:
                spread = np.linspace(-0.06, 0.06, len(indices))
                for k, idx in enumerate(indices):
                    jitter[idx] = spread[k]
        ax.scatter(
            i + 1 + jitter, d,
            alpha=0.7, s=30, color=c,
            edgecolors="black", linewidths=0.5, zorder=3,
        )

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods])
    ax.set_ylabel("Fitness Value")
    ax.set_title("Fitness Distribution per Method", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    print("\nFitness boxplot stats:")
    for m, d in zip(methods, plot_data):
        arr = np.array(d)
        print(f"  {m:<15}  median: {np.median(arr):.4f}   Q1 (lower bound): {np.percentile(arr, 25):.4f}")

    plt.tight_layout()
    savefig(fig, "plot_fitness_boxplot")
    plt.show()


# =============================================================================
# PLOT 2 — Box plot: Wilcoxon rank test (paired Δ values)
# =============================================================================

def plot_wilcoxon_boxplot():
    """
    One boxplot per non-baseline method showing the distribution of paired
    Δ = (method - baseline) / |baseline| values, with Wilcoxon test stats.
    """
    baseline_vals = DATA[BASELINE]
    methods = [m for m in DATA.keys() if m != BASELINE]

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        deltas = compute_deltas(DATA[method], baseline_vals)
        d = clean(deltas)
        color = COLORS.get(method, "#888888")

        W, p, n = wilcoxon_test(d)

        bp = ax.boxplot(
            d, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            showfliers=False,
        )

        rng = np.random.RandomState(42)
        jitter = rng.normal(0, 0.04, size=len(d))
        ax.scatter(
            1 + jitter, d,
            alpha=0.6, s=25, color=color,
            edgecolors="black", linewidths=0.5, zorder=3,
        )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

        if not np.isnan(W):
            title = f"$W$ = {W:.0f},  $p$ = {p:.4f}\n$n$ = {n}"
        else:
            title = f"$n$ = {n} (too few for test)"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(method.replace(" ", "\n"), fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([""])

    axes[0].set_ylabel(r"Relative $\Delta$  (method $-$ baseline) / |baseline|")

    fig.suptitle(
        f"Paired Differences vs. {BASELINE} (Wilcoxon Signed-Rank Test)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "plot_wilcoxon_boxplot")
    plt.show()


# =============================================================================
# PLOT 3 — Per-map scatter: all fitness values, maps on x-axis
# =============================================================================

def plot_permap_scatter():
    """
    Per-map paired comparison: one subplot per seeded method.
    x = Random fitness, y = seeded method fitness.
    Points above the diagonal = seeding won; below = Random won.
    """
    baseline = np.array([v if v is not None else np.nan for v in DATA[BASELINE]])
    seeded_methods = [m for m in DATA if m != BASELINE]

    fig, axes = plt.subplots(1, len(seeded_methods), figsize=(7 * len(seeded_methods), 7),
                             sharey=False)
    if len(seeded_methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, seeded_methods):
        y = np.array([v if v is not None else np.nan for v in DATA[method]])
        color = COLORS.get(method, "#4C72B0")
        mark  = MARKERS.get(method, "o")

        # Diagonal reference line
        all_vals = np.concatenate([baseline, y])
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="black", linewidth=1.2, linestyle="--", zorder=1, label="y = x")

        # Shade regions
        ax.fill_between([lo - pad, hi + pad], [lo - pad, hi + pad], hi + pad,
                        color=color, alpha=0.06, zorder=0)
        ax.fill_between([lo - pad, hi + pad], lo - pad, [lo - pad, hi + pad],
                        color="#999999", alpha=0.06, zorder=0)

        # Scatter points
        ax.scatter(baseline, y,
                   color=color, marker=mark, s=100,
                   edgecolors="black", linewidths=0.5, zorder=3)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{BASELINE} Fitness", fontsize=20)
        ax.set_ylabel(f"{method} Fitness", fontsize=20)
        ax.set_title(f"{method} vs {BASELINE}", fontweight="bold", fontsize=21)
        ax.tick_params(axis='both', labelsize=17)

        # Win/loss counts (lower fitness = better)
        seeding_better = int(np.sum(y < baseline))   # seeded method lower (better)
        random_better  = int(np.sum(y > baseline))   # random lower (better)
        ties           = int(np.sum(y == baseline))
        mask = (y <= baseline) & ~np.isnan(y) & ~np.isnan(baseline)
        better_maps = [MAPS[i] for i in range(len(MAPS)) if mask[i]]
        print(f"\n  {method} <= Random on ({len(better_maps)}): {better_maps}")
        ax.text(0.03, 0.97,
                f"Above line: Random lower F ({random_better})\n"
                f"Below line: {method} lower F ({seeding_better})\n"
                f"On line: equal ({ties})",
                transform=ax.transAxes, va="top", ha="left", fontsize=16,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.grid(linestyle="--", alpha=0.3, zorder=0)

    fig.suptitle("Per-Map Paired Comparison",
                 fontweight="bold", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "plot_permap_scatter")
    plt.show()


# =============================================================================
# PLOT 4 — Convergence: median fitness over generations
# =============================================================================

def plot_convergence():
    """
    Reads per-generation, per-map fitness values from CONVERGENCE_FILE.
    Plots median fitness across all maps over generations for each method.
    Lower fitness = better performance.
    """
    import openpyxl, os
    _f = globals().get("__file__")
    script_dir = os.path.dirname(os.path.abspath(_f)) if _f else os.getcwd()
    path = os.path.join(script_dir, CONVERGENCE_FILE)
    wb = openpyxl.load_workbook(path, data_only=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_medians = []

    # Auto-detect sheets: filter by agent prefix if set, then pick sheets with Gen data
    candidate_sheets = [
        s for s in wb.sheetnames
        if (CONVERGENCE_AGENT_FILTER is None or s.startswith(CONVERGENCE_AGENT_FILTER))
    ]
    # Derive a clean label from the sheet name, e.g. "11 agents (ID seeding)" -> "ID Seeding"
    def _label(sheet_name):
        import re
        _aliases = {"coord seeding": "SRS", "coordination": "SRS",
                    "random": "Random", "id seeding": "DSS", "id": "DSS",
                    "constraint": "CSRS"}
        m = re.search(r"\((.+)\)", sheet_name)
        raw = (m.group(1) if m else sheet_name).strip().lower()
        return _aliases.get(raw, raw.title())

    for sheet_name in candidate_sheets:
        method = _label(sheet_name)
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        # Find header row and restrict to columns listed in MAPS
        header_row = next(
            (r for r in rows if r and r[0] in ("Map", "Generation")), None
        )
        if header_row is not None:
            map_col_indices = [
                i for i, h in enumerate(header_row)
                if h in MAPS
            ]
        else:
            map_col_indices = None  # fallback: use all columns

        # Data rows are any row whose first cell starts with "Gen"
        data_rows = [r for r in rows if r and r[0] and str(r[0]).startswith("Gen")]

        generations = []
        medians = []
        for row in data_rows:
            gen_label = str(row[0])  # e.g. "Gen 1" or "Generation 1"
            parts = gen_label.split()
            try:
                gen_num = int(parts[-1]) if len(parts) >= 2 else None
            except ValueError:
                continue
            if gen_num is None or gen_num > 10:
                continue
            if map_col_indices is not None:
                values = [row[i] for i in map_col_indices if i < len(row) and isinstance(row[i], (int, float))]
            else:
                values = [v for v in row[1:] if isinstance(v, (int, float))]
            if not values:
                continue
            generations.append(gen_num)
            medians.append(float(np.median(values)))

        all_medians.extend(medians)
        gen_median = dict(zip(generations, medians))
        g1  = gen_median.get(1,  float("nan"))
        g10 = gen_median.get(10, float("nan"))
        print(f"  {method:<15}  Gen 1: {g1:.4f}   Gen 10: {g10:.4f}")
        color = COLORS.get(method, "#888888")
        mark  = MARKERS.get(method, "o")
        ax.plot(generations, medians,
                color=color, marker=mark, markersize=6,
                linewidth=1.8, markeredgecolor="black", markeredgewidth=0.5,
                label=method, zorder=3)

    # Y-axis: zoom to actual data range with a small margin, let matplotlib pick ticks
    if all_medians:
        lo, hi = min(all_medians), max(all_medians)
        pad = (hi - lo) * 0.15 or 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    ax.set_xlabel("Generation")
    ax.set_ylabel("Median Fitness (lower = better)")
    ax.set_title("Convergence: Median Fitness over Generations", fontweight="bold")
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4, zorder=0)

    plt.tight_layout()
    savefig(fig, "plot_convergence")
    plt.show()


# =============================================================================
# FUNCTION 5 — Wilcoxon Signed-Rank Statistical Summary (console output)
# =============================================================================
# NOTE: This function uses normalised Δ = (method − baseline) / |baseline| for
# ranking, matching the Excel sheet methodology.  Use FUNCTION 6 below for the
# raw-fitness version (scipy-exact, with Bonferroni correction) as a PNG table.
# =============================================================================

def print_wilcoxon_stats():
    """
    Prints the full Statistical Summary table for the Wilcoxon Signed-Rank Test,
    mirroring the layout of the 'Comparison (Best per Map)' Excel sheet
    (rows 41–53 of that sheet).

    For each non-baseline method the following are computed and printed:
      n          — number of map pairs used in the test (non-zero Δ pairs only)
      Mean Δ     — arithmetic mean of all valid Δ values (descriptive, sensitive to outliers)
      Median Δ   — robust central-tendency estimate; negative = method typically better
      W+         — sum of ranks where method Δ > 0  (large W+ → method tends to be WORSE)
      W−         — sum of ranks where method Δ < 0  (large W− → method tends to be BETTER)
      p (two-tailed)            — H₀: median Δ = 0.  Reject at p < ALPHA_LEVEL
      p (one-tailed: WORSE)     — valid when W+ > W−
      p (one-tailed: BETTER)    — valid when W− > W+
      Direction  — which rank sum dominates

    Δ = (method − baseline) / |baseline|, computed per map (same formula as compute_deltas).
    Ties (Δ = 0) and missing pairs are excluded from ranking, matching Excel methodology.
    """
    from scipy.stats import rankdata as _rankdata

    baseline_vals = DATA[BASELINE]
    methods = [m for m in DATA.keys() if m != BASELINE]

    # ── Compute statistics for every method ──────────────────────────────────
    results = {}
    for method in methods:
        deltas = compute_deltas(DATA[method], baseline_vals)
        # All valid (non-None) Δ values — used for Mean and Median
        d_all = np.array([v for v in deltas if v is not None], dtype=float)
        # Non-zero subset — used for the rank test (matches Excel: Δ=0 excluded)
        d_nz  = d_all[d_all != 0]
        n     = len(d_nz)

        mean_d = float(np.mean(d_all))   if len(d_all) > 0 else np.nan
        med_d  = float(np.median(d_all)) if len(d_all) > 0 else np.nan

        if n >= 5:
            # Rank |Δ| with average-tie handling (matches Excel RANK.AVG)
            abs_nz = np.abs(d_nz)
            ranks  = _rankdata(abs_nz)           # scipy default = average
            Wp = float(np.sum(ranks[d_nz > 0]))  # W+ : ranks where method > baseline
            Wm = float(np.sum(ranks[d_nz < 0]))  # W− : ranks where method < baseline

            # scipy.stats.wilcoxon returns T = min(W+, W−) and the two-tailed p-value
            _, p_two = stats.wilcoxon(d_nz, alternative="two-sided")

            # One-tailed p-values derived from the two-tailed result:
            #   when W− ≥ W+ the method tends to be BETTER  → p_better = p_two / 2
            #   when W+ > W− the method tends to be WORSE   → p_worse  = p_two / 2
            if Wm >= Wp:
                direction = f"W- dominant -> method BETTER than {BASELINE}"
                p_better  = p_two / 2
                p_worse   = 1.0 - p_two / 2
            else:
                direction = f"W+ dominant -> method WORSE than {BASELINE}"
                p_worse   = p_two / 2
                p_better  = 1.0 - p_two / 2
        else:
            Wp, Wm, p_two, p_worse, p_better = (np.nan,) * 5
            direction = "n too small for test"

        results[method] = dict(
            n=n, mean_d=mean_d, med_d=med_d,
            Wp=Wp, Wm=Wm,
            p_two=p_two, p_worse=p_worse, p_better=p_better,
            direction=direction,
        )

    # ── Formatting helpers ────────────────────────────────────────────────────
    col_w  = 22   # width of each method column
    lbl_w  = 44   # width of the statistic-label column
    sep    = "=" * (lbl_w + col_w * len(methods) + 2)
    thin   = "-" * (lbl_w + col_w * len(methods) + 2)

    def _fmt(v, fmt):
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        if isinstance(v, str):
            return v
        return fmt.format(v)

    def _row(label, key, fmt):
        line = f"  {label:<{lbl_w}}"
        for m in methods:
            cell = _fmt(results[m][key], fmt)
            line += f"{cell:<{col_w}}"
        print(line)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(sep)
    print(f"  Statistical Summary — Wilcoxon Signed-Rank Test")
    print(f"  Baseline: {BASELINE}  |  "
          f"Δ = (method − {BASELINE}) / |{BASELINE}|  |  "
          f"Δ=0 ties excluded from ranking  |  α = {ALPHA_LEVEL}")
    print(sep)

    # Header row
    header = f"  {'Statistic':<{lbl_w}}"
    for m in methods:
        header += f"{m:<{col_w}}"
    print(header)
    print(thin)

    _row("n  (map pairs in test)",                     "n",        "{:d}")
    _row("Mean Δ  (all valid pairs)",                  "mean_d",   "{:.6f}")
    _row("Median Δ  (all valid pairs)",                "med_d",    "{:.6e}")
    _row("W+  (ranks where method > baseline)",        "Wp",       "{:.0f}")
    _row("W−  (ranks where method < baseline)",        "Wm",       "{:.0f}")
    _row("p-value  (two-tailed)",                      "p_two",    "{:.6f}")
    _row("p  (one-tailed: method WORSE than baseline)","p_worse",  "{:.6f}")
    _row("p  (one-tailed: method BETTER than baseline)","p_better","{:.6f}")

    print(thin)
    # Direction row (strings — may be long, printed individually)
    print(f"  {'Direction':<{lbl_w}}")
    for m in methods:
        print(f"    {m:<{lbl_w - 2}}{results[m]['direction']}")

    print(sep)
    print()
    print("  NOTES:")
    print("    H₀: median Δ = 0 (non-parametric, no normality assumption).")
    print(f"    Reject H₀ at p < {ALPHA_LEVEL} (two-tailed).")
    print("    Large W− → method typically achieves lower (better) fitness than baseline.")
    print("    Large W+ → method typically achieves higher (worse) fitness than baseline.")
    print("    Prefer Median Δ over Mean Δ as effect-size estimate (robust to outliers).")
    print()
    print("  Per-method significance summary:")
    for m in methods:
        r = results[m]
        if np.isnan(r["p_two"]):
            verdict = "n too small — test not performed"
        elif r["p_two"] < ALPHA_LEVEL:
            verdict = f"SIGNIFICANT  (p = {r['p_two']:.4f} < {ALPHA_LEVEL})"
        else:
            verdict = f"not significant  (p = {r['p_two']:.4f} ≥ {ALPHA_LEVEL})"
        print(f"    {m:<18}  {verdict}")
    print()


# =============================================================================
# FUNCTION 6 — Wilcoxon Signed-Rank Table  →  PNG for thesis appendix
# =============================================================================

def plot_wilcoxon_table():
    """
    Renders a publication-ready PNG table of Wilcoxon Signed-Rank Test results
    for all non-baseline methods vs. BASELINE, saved to OUTPUT_DIR.

    Statistical method (matches wilcoxon_table.py exactly):
      - scipy.stats.wilcoxon(method_fitness, random_fitness, ...)
      - H₁ (one-sided):  seeding achieves LOWER (better) fitness than random
                          → alternative='less'
      - H₀ (two-sided):  median paired difference = 0
                          → alternative='two-sided'
      - Ties (diff = 0) are discarded:  zero_method='wilcox'
      - Bonferroni correction applied across all non-baseline methods (k = number of methods)
      - Test uses raw fitness values, NOT normalised Δ, to match wilcoxon_table.py

    Data source:
      - Uses the already-loaded DATA dict (Gen ≤ 10 best-of values from Excel)
      - Only map pairs where BOTH baseline and method have a numeric value are used

    Table columns:
      Method | n | Seed better | Rand. better | Ties |
      p (one-sided) | p (two-sided) | Median Δ (raw) | Sig. α=.05 | Sig. Bonf.

    Colour coding:
      green  = significant after Bonferroni correction
      yellow = significant at α = 0.05 (uncorrected only)
      white  = not significant
    """
    import scipy

    baseline_vals = DATA[BASELINE]
    methods = [m for m in DATA.keys() if m != BASELINE]
    k = len(methods)                        # number of comparisons for Bonferroni
    alpha_bonf = ALPHA_LEVEL / k

    # ── Compute statistics for every method ──────────────────────────────────
    results = []
    for method in methods:
        # Build paired arrays — skip any map where either value is missing
        pairs = [
            (b, m)
            for b, m in zip(baseline_vals, DATA[method])
            if b is not None and m is not None
        ]
        n = len(pairs)

        if n == 0:
            results.append({
                "method": method, "n": 0,
                "n_better": 0, "n_worse": 0, "n_tie": 0,
                "p_one": np.nan, "p_two": np.nan, "T": np.nan,
                "med_diff": np.nan, "hl_est": np.nan, "sig_005": False, "sig_bonf": False,
            })
            continue

        rand_arr   = np.array([p[0] for p in pairs])
        method_arr = np.array([p[1] for p in pairs])
        diff       = method_arr - rand_arr          # negative = seeding better

        n_better = int(np.sum(diff < 0))   # method achieved lower fitness (better)
        n_worse  = int(np.sum(diff > 0))   # random achieved lower fitness (better)
        n_tie    = int(np.sum(diff == 0))
        med_diff = float(np.median(diff))
        hl_est   = float(hodges_lehmann(diff))

        # Need at least 5 non-zero differences to run the test
        nonzero = diff[diff != 0]
        if len(nonzero) < 5:
            results.append({
                "method": method, "n": n,
                "n_better": n_better, "n_worse": n_worse, "n_tie": n_tie,
                "p_one": np.nan, "p_two": np.nan, "T": np.nan,
                "med_diff": med_diff, "hl_est": hl_est, "sig_005": False, "sig_bonf": False,
            })
            continue

        # ── THE ACTUAL SCIPY WILCOXON CALLS ──────────────────────────────────
        # wilcoxon(x, y) internally computes d = x - y, then tests d against 0.
        # alternative='less'      → H₁: method < random  (seeding is better)
        # alternative='two-sided' → H₀: median diff = 0
        # zero_method='wilcox'    → discard pairs where diff == 0
        T_two, p_two = stats.wilcoxon(
            method_arr, rand_arr,
            alternative="two-sided",
            zero_method="wilcox",
        )
        _, p_one = stats.wilcoxon(
            method_arr, rand_arr,
            alternative="less",
            zero_method="wilcox",
        )

        results.append({
            "method":   method,
            "n":        n,
            "n_better": n_better,
            "n_worse":  n_worse,
            "n_tie":    n_tie,
            "p_one":    float(p_one),
            "p_two":    float(p_two),
            "T":        float(T_two),
            "med_diff": med_diff,
            "hl_est":   hl_est,
            "sig_005":  bool(p_one < ALPHA_LEVEL),
            "sig_bonf": bool(p_one < alpha_bonf),
        })

    # ── Build table content ───────────────────────────────────────────────────
    col_labels = [
        "Method",
        "n",
        "Seed\nbetter",
        "Rand.\nbetter",
        "Ties",
        f"p  (one-sided)\nH₁: seed < {BASELINE}",
        "p  (two-sided)",
        "Median Δ\n(raw fitness)",
        "HL Est.\n(raw fitness)",
        f"Sig.\nα={ALPHA_LEVEL}",
        f"Sig.\nBonf. k={k}",
    ]

    cell_text   = []
    cell_colors = []

    for r in results:
        p1   = f"{r['p_one']:.5f}"  if not np.isnan(r["p_one"])  else "—"
        p2   = f"{r['p_two']:.5f}"  if not np.isnan(r["p_two"])  else "—"
        md   = f"{r['med_diff']:+.5f}" if not np.isnan(r["med_diff"]) else "—"
        hl   = f"{r['hl_est']:+.5f}"  if not np.isnan(r["hl_est"])  else "—"
        sig1 = "Yes *"  if r["sig_005"]  else "No"
        sig2 = "Yes **" if r["sig_bonf"] else "No"

        cell_text.append([
            r["method"],
            str(r["n"]),
            str(r["n_better"]),
            str(r["n_worse"]),
            str(r["n_tie"]),
            p1, p2, md, hl, sig1, sig2,
        ])

        if r["sig_bonf"]:
            cell_colors.append(["#d4edda"] * len(col_labels))
        elif r["sig_005"]:
            cell_colors.append(["#fff3cd"] * len(col_labels))
        else:
            cell_colors.append(["white"] * len(col_labels))

    # ── Render table as PNG ───────────────────────────────────────────────────
    n_rows     = len(cell_text)
    fig_height = 1.8 + 0.65 * n_rows
    fig, ax    = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText    = cell_text,
        colLabels   = col_labels,
        cellColours = cell_colors,
        colColours  = ["#d0d0d0"] * len(col_labels),
        cellLoc     = "center",
        loc         = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Left-align the method-name column
    for row_idx in range(1, n_rows + 1):
        cell = table.get_celld().get((row_idx, 0))
        if cell:
            cell._loc = "left"
            cell.PAD  = 0.04

    # Bold & slightly larger header row
    for col_idx in range(len(col_labels)):
        cell = table.get_celld().get((0, col_idx))
        if cell:
            cell.set_text_props(fontweight="bold", fontsize=9)

    # Title
    fig.suptitle(
        f"Wilcoxon Signed-Rank Test  —  {BASELINE} vs. Seeding Methods\n"
        f"(best fitness at Gen ≤ 10, raw values, scipy {scipy.__version__})",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # Footer methodology note
    fig.text(
        0.5, -0.02,
        f"H₁ (one-sided): seeding < {BASELINE}  |  "
        f"zero_method='wilcox' (ties discarded)  |  "
        f"Bonferroni α = {ALPHA_LEVEL}/{k} = {alpha_bonf:.4f}  |  "
        f"Green = Bonf. sig.  |  Yellow = α={ALPHA_LEVEL} sig. (uncorrected only)",
        ha="center", fontsize=8, style="italic", color="#555555",
    )

    plt.tight_layout()
    savefig(fig, "plot_wilcoxon_table")
    plt.show()

    # Also print a compact console summary
    print(f"\n  Wilcoxon Table  (Bonferroni k={k}, α_bonf={alpha_bonf:.4f})")
    print(f"  {'Method':<18}  {'n':>3}  {'p_one':>9}  {'p_two':>9}  {'Med.Δ':>9}  {'HL Est.':>9}  {'α=.05':>6}  {'Bonf.':>6}")
    print(f"  {'-'*82}")
    for r in results:
        p1  = f"{r['p_one']:.5f}"  if not np.isnan(r["p_one"])  else "  N/A  "
        p2  = f"{r['p_two']:.5f}"  if not np.isnan(r["p_two"])  else "  N/A  "
        md  = f"{r['med_diff']:+.5f}" if not np.isnan(r["med_diff"]) else "  N/A  "
        hl  = f"{r['hl_est']:+.5f}"  if not np.isnan(r["hl_est"])  else "  N/A  "
        s1  = "Yes *"  if r["sig_005"]  else "No"
        s2  = "Yes **" if r["sig_bonf"] else "No"
        print(f"  {r['method']:<18}  {r['n']:>3}  {p1:>9}  {p2:>9}  {md:>9}  {hl:>9}  {s1:>6}  {s2:>6}")
    print()


# =============================================================================
# HELPER — parameterised data loader (used by combined plots)
# =============================================================================

def _load_data_file(filepath, agent_filter):
    """
    Like _load_data_from_excel but for an arbitrary file + agent filter.
    Returns a {method: [values per MAPS]} dict without touching the global DATA.
    """
    import openpyxl
    _f = globals().get("__file__")
    script_dir = os.path.dirname(os.path.abspath(_f)) if _f else os.getcwd()
    path = os.path.join(script_dir, filepath)
    wb = openpyxl.load_workbook(path, data_only=True)

    candidate_sheets = [
        s for s in wb.sheetnames
        if agent_filter is None or s.startswith(agent_filter)
    ]

    data = {}
    for sheet_name in candidate_sheets:
        method = _label_from_sheet(sheet_name)
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        header_row = next((r for r in rows if r and r[0] in ("Map", "Generation")), None)
        if header_row is None:
            continue
        sheet_maps = [str(c) for c in header_row[1:] if c is not None]

        def _gen_num(cell):
            parts = str(cell).split()
            try:
                return int(parts[-1]) if len(parts) >= 2 else None
            except ValueError:
                return None

        gen_rows = [r for r in rows if r and r[0] and str(r[0]).startswith("Gen")
                    and _gen_num(r[0]) is not None and _gen_num(r[0]) <= 10]
        if not gen_rows:
            continue
        target_row = gen_rows[-1]
        lookup = {
            sheet_maps[i]: target_row[i + 1] if isinstance(target_row[i + 1], (int, float)) else None
            for i in range(len(sheet_maps))
        }
        data[method] = [lookup.get(m, None) for m in MAPS]
    return data


def _load_convergence_file(filepath, agent_filter):
    """
    Like the data-loading portion of plot_convergence but parameterised.
    Returns {method: (generations_list, medians_list)}.
    """
    import openpyxl
    _f = globals().get("__file__")
    script_dir = os.path.dirname(os.path.abspath(_f)) if _f else os.getcwd()
    path = os.path.join(script_dir, filepath)
    wb = openpyxl.load_workbook(path, data_only=True)

    candidate_sheets = [
        s for s in wb.sheetnames
        if agent_filter is None or s.startswith(agent_filter)
    ]

    result = {}
    for sheet_name in candidate_sheets:
        method = _label_from_sheet(sheet_name)
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        header_row = next(
            (r for r in rows if r and r[0] in ("Map", "Generation")), None
        )
        map_col_indices = (
            [i for i, h in enumerate(header_row) if h in MAPS]
            if header_row is not None else None
        )

        data_rows = [r for r in rows if r and r[0] and str(r[0]).startswith("Gen")]

        generations, medians = [], []
        for row in data_rows:
            parts = str(row[0]).split()
            try:
                gen_num = int(parts[-1]) if len(parts) >= 2 else None
            except ValueError:
                continue
            if gen_num is None or gen_num > 10:
                continue
            if map_col_indices is not None:
                values = [row[i] for i in map_col_indices if i < len(row) and isinstance(row[i], (int, float))]
            else:
                values = [v for v in row[1:] if isinstance(v, (int, float))]
            if not values:
                continue
            generations.append(gen_num)
            medians.append(float(np.median(values)))

        if generations:
            result[method] = (generations, medians)
    return result


# =============================================================================
# PLOT 7 — Combined boxplot: overlay two Excel files
# =============================================================================

def plot_combined_boxplot():
    """
    Same as plot_fitness_boxplot but overlays data from two Excel files.
    Primary file (COMBINED_FILE_1) is rendered at full opacity.
    Secondary file (COMBINED_FILE_2) is rendered at COMBINED_SECONDARY_ALPHA.
    Boxes from the two files are placed side-by-side within each method group.
    """
    data1 = _load_data_file(COMBINED_FILE_1, COMBINED_FILE_1_FILTER)
    data2 = _load_data_file(COMBINED_FILE_2, COMBINED_FILE_2_FILTER)

    if COMBINED_FILE_1_EXCLUDE:
        data1 = {m: v for m, v in data1.items() if m not in COMBINED_FILE_1_EXCLUDE}
    if COMBINED_FILE_2_EXCLUDE:
        data2 = {m: v for m, v in data2.items() if m not in COMBINED_FILE_2_EXCLUDE}

    # Use all methods present in either file, preserving COLORS key order where possible
    all_methods = list(dict.fromkeys(list(data1.keys()) + list(data2.keys())))
    n = len(all_methods)

    # x positions: each method gets a group of width 1; two boxes at ±0.22
    offset = 0.22
    pos1 = [i + 1 - offset for i in range(n)]
    pos2 = [i + 1 + offset for i in range(n)]
    box_width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, 2 * n), 5))

    rng = np.random.RandomState(42)

    def _draw_boxes(data_dict, positions, alpha, label_suffix):
        plot_data = [clean(data_dict.get(m, [])) for m in all_methods]
        colors    = [COLORS.get(m, "#888888") for m in all_methods]

        bp = ax.boxplot(
            plot_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(edgecolor="black", linewidth=1.2),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            showfliers=False,
            manage_ticks=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(alpha * 0.4 + (1 - alpha) * 0.1)  # scale box fill too

        for i, (d, c, pos) in enumerate(zip(plot_data, colors, positions)):
            if len(d) == 0:
                continue
            jitter = rng.normal(0, 0.04, size=len(d))
            seen = {}
            for idx, v in enumerate(d):
                seen.setdefault(v, []).append(idx)
            for indices in seen.values():
                if len(indices) > 1:
                    spread = np.linspace(-0.05, 0.05, len(indices))
                    for k, idx in enumerate(indices):
                        jitter[idx] = spread[k]
            ax.scatter(
                pos + jitter, d,
                alpha=alpha * 0.7, s=25, color=c,
                edgecolors="black", linewidths=0.5, zorder=3,
            )

    _draw_boxes(data1, pos1, alpha=1.0,                    label_suffix=COMBINED_FILE_1_LABEL)
    _draw_boxes(data2, pos2, alpha=COMBINED_SECONDARY_ALPHA, label_suffix=COMBINED_FILE_2_LABEL)

    # Legend proxies: solid patches for file 1, faded patches for file 2
    from matplotlib.patches import Patch
    legend_handles = []
    for m in data1:
        c = COLORS.get(m, "#888888")
        legend_handles.append(Patch(facecolor=c, edgecolor="black", alpha=0.8, label=m))
    for m in data2:
        c = COLORS.get(m, "#888888")
        legend_handles.append(Patch(facecolor=c, edgecolor="black", alpha=COMBINED_SECONDARY_ALPHA,
                                    label=f"{m} (10gen)"))

    ax.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.9)

    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_xticks([i + 1 for i in range(n)] )
    ax.set_xticklabels([m.replace(" ", "\n") for m in all_methods])
    ax.set_ylabel("Fitness Value")
    ax.set_title(
        f"Fitness Distribution: {COMBINED_FILE_1_LABEL} (SO-20) vs {COMBINED_FILE_2_LABEL} (SO-10)",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    plt.tight_layout()
    savefig(fig, "plot_combined_boxplot")
    plt.show()


# =============================================================================
# PLOT 8 — Combined convergence: overlay two Excel files
# =============================================================================

def plot_combined_convergence():
    """
    Same as plot_convergence but overlays data from two Excel files.
    Primary file (COMBINED_FILE_1): solid lines, full opacity.
    Secondary file (COMBINED_FILE_2): dashed lines, COMBINED_SECONDARY_ALPHA opacity.
    Colour coding and markers are identical for both.
    """
    conv1 = _load_convergence_file(COMBINED_FILE_1, COMBINED_FILE_1_FILTER)
    conv2 = _load_convergence_file(COMBINED_FILE_2, COMBINED_FILE_2_FILTER)

    if COMBINED_FILE_1_EXCLUDE:
        conv1 = {m: v for m, v in conv1.items() if m not in COMBINED_FILE_1_EXCLUDE}
    if COMBINED_FILE_2_EXCLUDE:
        conv2 = {m: v for m, v in conv2.items() if m not in COMBINED_FILE_2_EXCLUDE}

    fig, ax = plt.subplots(figsize=(8, 5))
    all_medians = []

    def _draw_lines(conv_data, alpha, linestyle, label_suffix=None):
        for method, (generations, medians) in conv_data.items():
            color = COLORS.get(method, "#888888")
            mark  = MARKERS.get(method, "o")
            all_medians.extend(medians)
            lbl = f"{method} ({label_suffix})" if label_suffix else method
            ax.plot(
                generations, medians,
                color=color, marker=mark, markersize=6,
                linewidth=1.8, markeredgecolor="black", markeredgewidth=0.5,
                linestyle=linestyle, alpha=alpha,
                label=lbl,
                zorder=3,
            )

    _draw_lines(conv1, alpha=1.0,                     linestyle="-")
    _draw_lines(conv2, alpha=COMBINED_SECONDARY_ALPHA, linestyle="--", label_suffix="10gen")

    if all_medians:
        lo, hi = min(all_medians), max(all_medians)
        pad = (hi - lo) * 0.15 or 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    ax.set_xlabel("Generation")
    ax.set_ylabel("Median Fitness (lower = better)")
    ax.set_title(
        f"Convergence: {COMBINED_FILE_1_LABEL} (SO-20) vs {COMBINED_FILE_2_LABEL} (SO-10)",
        fontweight="bold",
    )
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4, zorder=0)

    plt.tight_layout()
    savefig(fig, "plot_combined_convergence")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    options = {
        "fitness_boxplot":      plot_fitness_boxplot,
        "wilcoxon_boxplot":     plot_wilcoxon_boxplot,
        "permap_scatter":       plot_permap_scatter,
        "convergence":          plot_convergence,
        "wilcoxon_stats":       print_wilcoxon_stats,
        "wilcoxon_table":       plot_wilcoxon_table,
        "combined_boxplot":     plot_combined_boxplot,
        "combined_convergence": plot_combined_convergence,
    }

    if PLOT_CHOICE not in options:
        print(f"Unknown PLOT_CHOICE: '{PLOT_CHOICE}'")
        print(f"Valid options: {list(options.keys())}")
        return

    print(f"Generating: {PLOT_CHOICE}")
    options[PLOT_CHOICE]()


if __name__ == "__main__":
    main()

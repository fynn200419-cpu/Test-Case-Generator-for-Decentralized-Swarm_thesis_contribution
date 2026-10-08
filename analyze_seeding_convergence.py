"""
Analyze seeding method convergence speed comparison.
Compares when seeding methods achieve results as good as random sampling.

Output: Clean tables showing per-trial data and proper aggregate speedup calculation.
"""
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats

# --- Robust Loading ---
class BetterStub:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)

class GracefulUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try: return super().find_class(module, name)
        except (ImportError, AttributeError): return BetterStub

def load_robust(file_path):
    with open(file_path, 'rb') as f:
        return GracefulUnpickler(f).load()

def extract_convergence_data(entry):
    """
    Extract convergence data from a result entry.
    Returns: (n_evals_array, best_so_far_array, final_best)
    """
    res = entry.get('result')
    history = getattr(res, "history", None)
    if not history:
        return None, None, None

    n_evals = []
    opt_vals = []

    for j, e in enumerate(history):
        evaluator = getattr(e, "evaluator", None)
        n_eval = getattr(evaluator, "n_eval", j) if evaluator else j
        n_evals.append(n_eval)

        opt = getattr(e, "opt", None)
        if opt and hasattr(opt[0], "F"):
            F = np.array(opt[0].F).flatten()[0]
        else:
            F = np.nan
        opt_vals.append(float(F))

    n_evals = np.array(n_evals)
    opt_vals = np.array(opt_vals)
    best_so_far = np.minimum.accumulate(opt_vals)

    return n_evals, best_so_far, best_so_far[-1] if len(best_so_far) > 0 else np.nan

def find_first_match(n_evals, best_so_far, target_value, tolerance=0.01):
    """
    Find the first evaluation count where the best-so-far reaches target_value (with tolerance).
    Returns: (n_eval when target reached, or None if never reached)
    """
    for i, (n_eval, val) in enumerate(zip(n_evals, best_so_far)):
        if val <= target_value * (1 + tolerance):  # within tolerance
            return n_eval
    return None

def calculate_overall_speedup(df, rand_col, seeding_col, max_evals_col=None):
    """
    Calculate overall speedup using sum of evaluations (proper method).

    Two calculations:
    1. OPTIMISTIC: Only considers trials where seeding reached target
       Speedup = sum(random_evals) / sum(seeding_evals) for successful trials

    2. PENALIZED: For failed trials, uses max_evals as penalty
       Speedup = sum(random_evals) / sum(seeding_evals_with_penalty) for ALL trials
       This properly accounts for failures.
    """
    # Filter to trials where seeding reached target (OPTIMISTIC)
    valid = df[[rand_col, seeding_col]].dropna()
    n_success = len(valid)
    n_total = len(df)

    if n_success == 0:
        return None, None, None, 0, None

    rand_total_success = valid[rand_col].sum()
    seeding_total_success = valid[seeding_col].sum()

    # OPTIMISTIC: Only successful trials
    optimistic_speedup = rand_total_success / seeding_total_success if seeding_total_success > 0 else None

    # PENALIZED: Include failures with max_evals as penalty
    if max_evals_col and max_evals_col in df.columns:
        # For failed trials, use max_evals as the "time to reach"
        seeding_with_penalty = df[seeding_col].copy()
        failed_mask = df[seeding_col].isna()
        seeding_with_penalty[failed_mask] = df.loc[failed_mask, max_evals_col]

        rand_total_all = df[rand_col].sum()
        seeding_total_penalized = seeding_with_penalty.sum()

        penalized_speedup = rand_total_all / seeding_total_penalized if seeding_total_penalized > 0 else None
    else:
        penalized_speedup = None

    # Individual speedups for geometric mean (only successful)
    individual_speedups = valid[rand_col] / valid[seeding_col]
    geometric_mean = stats.gmean(individual_speedups) if len(individual_speedups) > 0 else None

    return optimistic_speedup, geometric_mean, n_success, n_total, penalized_speedup

def print_table(df, columns, headers, title):
    """Print a nicely formatted table."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    # Header
    header_str = " | ".join(f"{h:>12}" for h in headers)
    print(f"  {header_str}")
    print(f"  {'-'*len(header_str)}")

    # Data rows
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                values.append(f"{'---':>12}")
            elif isinstance(val, float):
                values.append(f"{val:>12.2f}")
            else:
                values.append(f"{val:>12}")
        print(f"  {' | '.join(values)}")

def analyze_rInf():
    """Analyze rInf experiments: random vs ID seeding vs coordination seeding"""

    # Paths
    path_noseeding_and_idseeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260107_151408_GA_fail1_rInf_10gen_10agentsnoseeding_11agentsnoseeding_11agentsIDseeding\all_results.pkl")
    path_coord_seeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260123_004651_GA_fail1_rInf_20gen_11agentscoordinationseeding\all_results.pkl")

    print("Loading rInf data...")
    data_combined = load_robust(path_noseeding_and_idseeding)
    data_coord = load_robust(path_coord_seeding)

    FILTER_NUM_AGENTS = 11

    no_seeding_by_trial = {}
    id_seeding_by_trial = {}

    for entry in data_combined:
        if entry.get('num_agents') == FILTER_NUM_AGENTS:
            trial = int(entry.get('trial'))
            if trial not in no_seeding_by_trial:
                no_seeding_by_trial[trial] = entry
            else:
                id_seeding_by_trial[trial] = entry

    coord_seeding_by_trial = {}
    for entry in data_coord:
        if entry.get('num_agents') == FILTER_NUM_AGENTS:
            trial = int(entry.get('trial'))
            coord_seeding_by_trial[trial] = entry

    common_trials = sorted(
        set(no_seeding_by_trial.keys()) &
        set(id_seeding_by_trial.keys()) &
        set(coord_seeding_by_trial.keys())
    )

    results = []

    for trial in common_trials:
        rand_evals, rand_best, rand_final = extract_convergence_data(no_seeding_by_trial[trial])
        id_evals, id_best, id_final = extract_convergence_data(id_seeding_by_trial[trial])
        coord_evals, coord_best, coord_final = extract_convergence_data(coord_seeding_by_trial[trial])

        if rand_evals is None or id_evals is None or coord_evals is None:
            continue

        target = rand_final
        rand_reach_final = find_first_match(rand_evals, rand_best, target, tolerance=0.001)
        id_reach_target = find_first_match(id_evals, id_best, target, tolerance=0.01)
        coord_reach_target = find_first_match(coord_evals, coord_best, target, tolerance=0.01)

        # CAP: For fair comparison, cap coordination seeding at 1000 evals (10 generations)
        # Same budget as ID seeding
        FAIR_CAP = 1000
        if coord_reach_target is not None and coord_reach_target > FAIR_CAP:
            coord_reach_target = None  # Treat as failure if it took more than 10 gen

        # Calculate individual speedups
        id_speedup = rand_reach_final / id_reach_target if id_reach_target and rand_reach_final else np.nan
        coord_speedup = rand_reach_final / coord_reach_target if coord_reach_target and rand_reach_final else np.nan

        # Max evals for penalty = 1000 (fair common budget for both methods)
        id_max_evals = FAIR_CAP
        coord_max_evals = FAIR_CAP

        results.append({
            'Trial': trial,
            'Rand@Eval': rand_reach_final,
            'ID@Eval': id_reach_target,
            'ID_MaxEvals': id_max_evals,
            'ID_Speedup': id_speedup,
            'Coord@Eval': coord_reach_target,
            'Coord_MaxEvals': coord_max_evals,
            'Coord_Speedup': coord_speedup,
            'Rand_Final': rand_final,
            'ID_Final': id_final,
            'Coord_Final': coord_final,
        })

    df = pd.DataFrame(results)

    # Print main comparison table
    print_table(
        df,
        ['Trial', 'Rand@Eval', 'ID@Eval', 'ID_Speedup', 'Coord@Eval', 'Coord_Speedup'],
        ['Trial', 'Random@', 'ID@', 'ID Speedup', 'Coord@', 'Coord Speedup'],
        "rInf: CONVERGENCE SPEED COMPARISON (11 Agents)"
    )

    # Print final objective values table
    print_table(
        df,
        ['Trial', 'Rand_Final', 'ID_Final', 'Coord_Final'],
        ['Trial', 'Random', 'ID Seed', 'Coord Seed'],
        "rInf: FINAL OBJECTIVE VALUES"
    )

    return df

def analyze_r20():
    """Analyze r20 experiments: random vs coordination seeding"""

    path_random = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260124_045836_GA_fail1_r20_20gen_\10agentsrandom_11agentsrandom\all_results.pkl")
    path_coord = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260125_163000_GA_fail1_r20_200gen_10agentsrandom_11agentscoordination\all_results.pkl")

    print("\nLoading r20 data...")
    data_random = load_robust(path_random)
    data_coord = load_robust(path_coord)

    FILTER_NUM_AGENTS = 11

    random_by_trial = {}
    for entry in data_random:
        if entry.get('num_agents') == FILTER_NUM_AGENTS:
            trial = int(entry.get('trial'))
            random_by_trial[trial] = entry

    coord_by_trial = {}
    for entry in data_coord:
        if entry.get('num_agents') == FILTER_NUM_AGENTS:
            trial = int(entry.get('trial'))
            coord_by_trial[trial] = entry

    common_trials = sorted(set(random_by_trial.keys()) & set(coord_by_trial.keys()))

    results = []

    for trial in common_trials:
        rand_evals, rand_best, rand_final = extract_convergence_data(random_by_trial[trial])
        coord_evals, coord_best, coord_final = extract_convergence_data(coord_by_trial[trial])

        if rand_evals is None or coord_evals is None:
            continue

        target = rand_final
        rand_reach_final = find_first_match(rand_evals, rand_best, target, tolerance=0.001)
        coord_reach_target = find_first_match(coord_evals, coord_best, target, tolerance=0.01)

        coord_speedup = rand_reach_final / coord_reach_target if coord_reach_target and rand_reach_final else np.nan

        # Max evals (total budget) for penalty calculation
        coord_max_evals = coord_evals[-1] if len(coord_evals) > 0 else np.nan

        results.append({
            'Trial': trial,
            'Rand@Eval': rand_reach_final,
            'Coord@Eval': coord_reach_target,
            'Coord_MaxEvals': coord_max_evals,
            'Coord_Speedup': coord_speedup,
            'Rand_Final': rand_final,
            'Coord_Final': coord_final,
        })

    df = pd.DataFrame(results)

    # Print main comparison table
    print_table(
        df,
        ['Trial', 'Rand@Eval', 'Coord@Eval', 'Coord_Speedup'],
        ['Trial', 'Random@', 'Coord@', 'Speedup'],
        "r20: CONVERGENCE SPEED COMPARISON (11 Agents)"
    )

    # Print final objective values table
    print_table(
        df,
        ['Trial', 'Rand_Final', 'Coord_Final'],
        ['Trial', 'Random', 'Coord Seed'],
        "r20: FINAL OBJECTIVE VALUES"
    )

    return df

def print_final_summary(df_rInf, df_r20):
    """Print the final summary with proper speedup calculations."""

    print("\n")
    print("=" * 100)
    print("  FINAL SUMMARY: OVERALL SPEEDUP CALCULATIONS")
    print("=" * 100)

    print("""
  Two Speedup Calculations:
  -------------------------
  1. OPTIMISTIC: SUM(rand_evals) / SUM(seed_evals)  [only for successful trials]
     - Ignores failures, may overestimate speedup

  2. PENALIZED:  SUM(rand_evals) / SUM(seed_evals_with_penalty)  [ALL trials]
     - For failed trials: uses max_evals (full budget) as penalty
     - Fair comparison that accounts for failures
  """)

    # rInf Summary
    print("\n" + "-" * 100)
    print("  rInf (Infinite Awareness Radius) - 11 Agents")
    print("-" * 100)

    # ID Seeding
    opt_id, geom_id, n_id, n_total_id, pen_id = calculate_overall_speedup(
        df_rInf, 'Rand@Eval', 'ID@Eval', 'ID_MaxEvals')
    print(f"\n  ID SEEDING:")
    print(f"    Trials reaching target: {n_id}/{n_total_id}")
    print(f"    Total random evals (all trials): {df_rInf['Rand@Eval'].sum():.0f}")
    print(f"    Total ID seeding evals (success only): {df_rInf['ID@Eval'].sum():.0f}")
    print(f"    ---------------------------------------------")
    print(f"    OPTIMISTIC SPEEDUP (success only):  {opt_id:.2f}x" if opt_id else "    OPTIMISTIC: N/A")
    print(f"    PENALIZED SPEEDUP  (all trials):    {pen_id:.2f}x" if pen_id else "    PENALIZED: N/A")
    print(f"    Geometric mean (success only):      {geom_id:.2f}x" if geom_id else "    Geometric mean: N/A")

    # Coord Seeding
    opt_coord, geom_coord, n_coord, n_total_coord, pen_coord = calculate_overall_speedup(
        df_rInf, 'Rand@Eval', 'Coord@Eval', 'Coord_MaxEvals')
    print(f"\n  COORDINATION SEEDING:")
    print(f"    Trials reaching target: {n_coord}/{n_total_coord}")
    print(f"    Total random evals (all trials): {df_rInf['Rand@Eval'].sum():.0f}")
    print(f"    Total coord seeding evals (success only): {df_rInf['Coord@Eval'].sum():.0f}")
    print(f"    ---------------------------------------------")
    print(f"    OPTIMISTIC SPEEDUP (success only):  {opt_coord:.2f}x" if opt_coord else "    OPTIMISTIC: N/A")
    print(f"    PENALIZED SPEEDUP  (all trials):    {pen_coord:.2f}x" if pen_coord else "    PENALIZED: N/A")
    print(f"    Geometric mean (success only):      {geom_coord:.2f}x" if geom_coord else "    Geometric mean: N/A")

    # r20 Summary
    print("\n" + "-" * 100)
    print("  r20 (Awareness Radius = 20) - 11 Agents")
    print("-" * 100)

    opt_coord_r20, geom_coord_r20, n_coord_r20, n_total_r20, pen_coord_r20 = calculate_overall_speedup(
        df_r20, 'Rand@Eval', 'Coord@Eval', 'Coord_MaxEvals')
    print(f"\n  COORDINATION SEEDING:")
    print(f"    Trials reaching target: {n_coord_r20}/{n_total_r20}")
    print(f"    Total random evals (all trials): {df_r20['Rand@Eval'].sum():.0f}")
    print(f"    Total coord seeding evals (success only): {df_r20['Coord@Eval'].sum():.0f}")
    print(f"    ---------------------------------------------")
    print(f"    OPTIMISTIC SPEEDUP (success only):  {opt_coord_r20:.2f}x" if opt_coord_r20 else "    OPTIMISTIC: N/A")
    print(f"    PENALIZED SPEEDUP  (all trials):    {pen_coord_r20:.2f}x" if pen_coord_r20 else "    PENALIZED: N/A")
    print(f"    Geometric mean (success only):      {geom_coord_r20:.2f}x" if geom_coord_r20 else "    Geometric mean: N/A")

    # Final comparison table
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    print(f"""
  +-----------------+--------------------+----------+--------------------+--------------------+
  | Awareness       | Seeding Method     | Success  | OPTIMISTIC Speedup | PENALIZED Speedup  |
  | Radius          |                    | Rate     | (ignores failures) | (with max_evals)   |
  +-----------------+--------------------+----------+--------------------+--------------------+
  | rInf            | ID Seeding         | {n_id:>2}/{n_total_id:<2}    | {opt_id:>6.2f}x             | {pen_id:>6.2f}x             |
  | rInf            | Coordination       | {n_coord:>2}/{n_total_coord:<2}    | {opt_coord:>6.2f}x             | {pen_coord:>6.2f}x             |
  | r20             | Coordination       | {n_coord_r20:>2}/{n_total_r20:<2}    | {opt_coord_r20:>6.2f}x             | {pen_coord_r20:>6.2f}x             |
  +-----------------+--------------------+----------+--------------------+--------------------+

  Note: PENALIZED speedup is the fairer metric as it accounts for failures.
  """)

if __name__ == "__main__":
    df_rInf = analyze_rInf()
    df_r20 = analyze_r20()
    print_final_summary(df_rInf, df_r20)

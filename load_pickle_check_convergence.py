import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- 1. ROBUSTES LADEN ---
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

# --- 2. VERGLEICHS-PLOT FUNKTION ---
def plot_comparison(results_list, labels, num_agents, trial, save_path):
    """
    Plottet mehrere Convergence-Kurven in ein einziges Diagramm.
    results_list: Liste von result dicts
    labels: Liste von Labels für jede Kurve
    """
    plt.figure(figsize=(10, 6))

    for i, res_dict in enumerate(results_list):
        res = res_dict.get('result')
        history = getattr(res, "history", None)
        if not history:
            continue

        # Daten extrahieren
        n_evals = np.array([getattr(getattr(e, "evaluator", None), "n_eval", j) for j, e in enumerate(history)])
        opt_vals = []
        for e in history:
            opt = getattr(e, "opt", None)
            if opt and hasattr(opt[0], "F"):
                F = np.array(opt[0].F).flatten()[0]
            else:
                F = np.nan
            opt_vals.append(float(F))

        # Best-so-far Kurve (Kumulatives Minimum)
        opt_vals = np.minimum.accumulate(np.array(opt_vals))

        label_name = labels[i] if i < len(labels) else f"Dataset {i+1}"
        plt.plot(n_evals, opt_vals, label=label_name, linewidth=2)

    plt.title(f"Comparison: Swarm Size {num_agents} - Trial {trial}")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective (Best so far)")
    plt.yscale("linear")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# --- 3. KONFIGURATION ---
# =============================================================================
# MODUS: "two_paths" oder "three_paths_rInf"
# =============================================================================
MODE = "three_paths_rInf"  # <-- HIER UMSCHALTEN

# --- Für MODE = "two_paths" (z.B. r20) ---
path_no_seeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260124_045836_GA_fail1_r20_20gen_\10agentsrandom_11agentsrandom\all_results.pkl")
path_seeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260125_163000_GA_fail1_r20_200gen_10agentsrandom_11agentscoordination\all_results.pkl")

# --- Für MODE = "three_paths_rInf" ---
# Dieser Pfad enthält BEIDE: no seeding UND ID seeding für 11 agents (je 30 trials, abwechselnd)
path_noseeding_and_idseeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260107_151408_GA_fail1_rInf_10gen_10agentsnoseeding_11agentsnoseeding_11agentsIDseeding\all_results.pkl")
# Dieser Pfad enthält coordination seeding
path_coord_seeding = Path(r"C:\Users\fynnl\OneDrive\Desktop\TUM\BachelorThesis\experiment-targeted-failure-modes-fork\results_to_save\exp_swarmsize\result_20260123_004651_GA_fail1_rInf_20gen_11agentscoordinationseeding\all_results.pkl")

# Nur für diese Anzahl Agents filtern
FILTER_NUM_AGENTS = 11

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":

    if MODE == "two_paths":
        # =====================================================================
        # MODUS: Zwei Pfade (no seeding vs seeding)
        # =====================================================================
        output_dir = path_seeding.parent / "comparison_plots_11agents"
        output_dir.mkdir(exist_ok=True)

        print(f"Lade no seeding Daten von: {path_no_seeding}")
        data_no_seeding = load_robust(path_no_seeding)

        print(f"Lade seeding Daten von: {path_seeding}")
        data_seeding = load_robust(path_seeding)

        no_seeding_by_trial = {}
        for entry in data_no_seeding:
            if entry.get('num_agents') == FILTER_NUM_AGENTS:
                trial = int(entry.get('trial'))
                no_seeding_by_trial[trial] = entry

        seeding_by_trial = {}
        for entry in data_seeding:
            if entry.get('num_agents') == FILTER_NUM_AGENTS:
                trial = int(entry.get('trial'))
                seeding_by_trial[trial] = entry

        print(f"\nGefunden für {FILTER_NUM_AGENTS} Agents:")
        print(f"  No Seeding: {len(no_seeding_by_trial)} Trials")
        print(f"  Seeding: {len(seeding_by_trial)} Trials")

        common_trials = sorted(set(no_seeding_by_trial.keys()) & set(seeding_by_trial.keys()))
        print(f"  Gemeinsame Trials: {len(common_trials)}")

        for trial in common_trials:
            results_list = [no_seeding_by_trial[trial], seeding_by_trial[trial]]
            labels = ["no seeding", "seeding"]

            filename = f"comp_{FILTER_NUM_AGENTS}agents_trial{trial}.png"
            save_path = output_dir / filename

            plot_comparison(results_list, labels, FILTER_NUM_AGENTS, trial, save_path)
            print(f"Plot erstellt: {filename}")

        print(f"\nFertig! Alle Vergleiche sind hier: {output_dir}")

    elif MODE == "three_paths_rInf":
        # =====================================================================
        # MODUS: Drei Methoden (no seeding, ID seeding, coordination seeding)
        # =====================================================================
        output_dir = path_coord_seeding.parent / "comparison_plots_3methods"
        output_dir.mkdir(exist_ok=True)

        print(f"Lade no seeding + ID seeding Daten von: {path_noseeding_and_idseeding}")
        data_combined = load_robust(path_noseeding_and_idseeding)

        print(f"Lade coordination seeding Daten von: {path_coord_seeding}")
        data_coord = load_robust(path_coord_seeding)

        # Aus data_combined: Für 11 agents gibt es 2 Einträge pro Trial
        # Erster Eintrag = no seeding, Zweiter Eintrag = ID seeding
        no_seeding_by_trial = {}
        id_seeding_by_trial = {}

        for entry in data_combined:
            if entry.get('num_agents') == FILTER_NUM_AGENTS:
                trial = int(entry.get('trial'))
                if trial not in no_seeding_by_trial:
                    no_seeding_by_trial[trial] = entry  # Erster Eintrag = no seeding
                else:
                    id_seeding_by_trial[trial] = entry  # Zweiter Eintrag = ID seeding

        # Coordination seeding
        coord_seeding_by_trial = {}
        for entry in data_coord:
            if entry.get('num_agents') == FILTER_NUM_AGENTS:
                trial = int(entry.get('trial'))
                coord_seeding_by_trial[trial] = entry

        print(f"\nGefunden für {FILTER_NUM_AGENTS} Agents:")
        print(f"  No Seeding: {len(no_seeding_by_trial)} Trials")
        print(f"  ID Seeding: {len(id_seeding_by_trial)} Trials")
        print(f"  Coord Seeding: {len(coord_seeding_by_trial)} Trials")

        # Gemeinsame Trials finden (alle drei müssen vorhanden sein)
        common_trials = sorted(
            set(no_seeding_by_trial.keys()) &
            set(id_seeding_by_trial.keys()) &
            set(coord_seeding_by_trial.keys())
        )
        print(f"  Gemeinsame Trials: {len(common_trials)}")

        for trial in common_trials:
            results_list = [
                no_seeding_by_trial[trial],
                id_seeding_by_trial[trial],
                coord_seeding_by_trial[trial]
            ]
            labels = ["no seeding", "ID seeding", "coord seeding"]

            filename = f"comp_{FILTER_NUM_AGENTS}agents_trial{trial}_3methods.png"
            save_path = output_dir / filename

            plot_comparison(results_list, labels, FILTER_NUM_AGENTS, trial, save_path)
            print(f"Plot erstellt: {filename}")

        print(f"\nFertig! Alle Vergleiche sind hier: {output_dir}")
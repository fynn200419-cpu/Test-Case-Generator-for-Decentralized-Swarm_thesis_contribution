"""
Re-run top-10 GA solutions with trajectory data saving enabled.

This script takes the top 10 solutions from seeds pickle files and re-runs them with
output_all_states=True and save=True to generate trajectory data needed for coordination seeding.
"""

import pickle
import numpy as np
from pathlib import Path
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR


def rerun_top10_solutions_with_trajectories(ga_results_dir,
                                            num_agents_list=None,
                                            trial_list=None,
                                            failure_option=2.1,
                                            n_seed=20,
                                            output_subdir='top10_trajectories'):
    """
    Re-run the top-10 GA solutions with trajectory data saving enabled.

    This function loads the top 10 solutions from seeds pickle files (sorted by fitness),
    re-runs them with MATLAB's output_all_states=True and save=True, and saves the trajectory
    data to .mat files for use in coordination seeding.

    Parameters
    ----------
    ga_results_dir : str or Path
        Path to the GA results directory containing seeds_*.pkl files
    num_agents_list : list of int, optional
        List of agent counts to process. If None, processes all found.
    trial_list : list of int, optional
        List of trial numbers to process. If None, processes all found.
    failure_option : float
        Failure option used in the GA run (e.g., 2.1 for velocity single)
    n_seed : int
        Number of top solutions to re-run (default: 10)
    output_subdir : str
        Name of subdirectory to create for trajectory outputs (default: 'top10_trajectories')

    Returns
    -------
    dict
        Dictionary mapping (num_agents, trial_num) to list of trajectory file paths

    Example
    -------
    >>> trajectory_paths = rerun_top10_solutions_with_trajectories(
    ...     ga_results_dir='results_to_save/GA_fail_TIMESTAMP',
    ...     num_agents_list=[10],
    ...     trial_list=[1, 2, 3]
    ... )
    >>> print(trajectory_paths[(10, 1)])  # List of 10 trajectory paths for trial 1
    """

    ga_results_dir = Path(ga_results_dir)

    if not ga_results_dir.exists():
        raise FileNotFoundError(f"GA results directory not found: {ga_results_dir}")

    # Create output directory for trajectories
    trajectory_output_dir = ga_results_dir / output_subdir
    trajectory_output_dir.mkdir(exist_ok=True)

    print(f"Re-running top-{n_seed} GA solutions with trajectory saving...")
    print(f"GA results dir: {ga_results_dir}")
    print(f"Output dir: {trajectory_output_dir}")

    # Find all seeds pickle files
    seeds_files = list(ga_results_dir.glob(f"seeds_*agents_trial*_failopt{failure_option}.pkl"))

    if len(seeds_files) == 0:
        print(f"WARNING: No seeds files found matching pattern: seeds_*agents_trial*_failopt{failure_option}.pkl")
        return {}

    print(f"Found {len(seeds_files)} seeds files")

    # Initialize simulator
    simulator = SimulatorSwarmlab(str(SIM_DIR))

    trajectory_paths = {}

    # Process each seeds file
    for seeds_file in seeds_files:
        # Parse filename to extract num_agents and trial_num
        filename = seeds_file.stem  # e.g., "seeds_10agents_trial5_failopt2.1"
        parts = filename.split('_')

        try:
            num_agents = int(parts[1].replace('agents', ''))
            trial_num = int(parts[2].replace('trial', ''))
        except (IndexError, ValueError) as e:
            print(f"WARNING: Could not parse filename {seeds_file.name}: {e}")
            continue

        # Filter by num_agents and trial if specified
        if num_agents_list is not None and num_agents not in num_agents_list:
            continue
        if trial_list is not None and trial_num not in trial_list:
            continue

        print(f"\nProcessing: {num_agents} agents, trial {trial_num}")

        # Load seeds file (contains X and F arrays from GA run)
        try:
            with open(seeds_file, 'rb') as f:
                seeds_data = pickle.load(f)
        except Exception as e:
            print(f"  ERROR: Could not load seeds file: {e}")
            continue

        X_prev = seeds_data.get('X')
        F_prev = seeds_data.get('F')

        if X_prev is None or F_prev is None:
            print(f"  WARNING: Missing 'X' or 'F' in seeds file")
            continue

        X_prev = np.asarray(X_prev, dtype=float)
        F_prev = np.asarray(F_prev, dtype=float)

        # Ensure 2D
        if X_prev.ndim == 1:
            X_prev = np.atleast_2d(X_prev)
        if F_prev.ndim != 1:
            F_prev = F_prev.flatten()

        # Select top n_seed by best (lowest) fitness
        best_idx = np.argsort(F_prev)[:n_seed]
        seed_solutions = X_prev[best_idx]
        seed_fitness = F_prev[best_idx]

        print(f"  Re-running top {len(seed_solutions)} solutions")
        print(f"  Best fitness values: {seed_fitness}")

        # Load corresponding ga_result file to get simulation parameters
        ga_result_file = ga_results_dir / f"ga_result_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"

        if not ga_result_file.exists():
            print(f"  WARNING: GA result file not found: {ga_result_file.name}")
            print(f"  Using default simulation parameters")
            end_time = 300.0
            start_x = 0.0
            end_x = 100.0
            range_aware = float('inf')
            v_ref = 6.0
            env_config_type = 1
        else:
            with open(ga_result_file, 'rb') as f:
                ga_result = pickle.load(f)
            end_time = ga_result.get('end_time', 300.0)
            start_x = ga_result.get('start_x', 0.0)
            end_x = ga_result.get('end_x', 100.0)
            range_aware = ga_result.get('range_aware', float('inf'))
            v_ref = ga_result.get('v_ref', 6.0)
            env_config_type = ga_result.get('env_config_type', 1)

        # Create subdirectory for this specific trial
        trial_output_dir = trajectory_output_dir / f"{num_agents}agents_trial{trial_num}"
        trial_output_dir.mkdir(exist_ok=True)

        # Store paths for this trial
        trial_trajectory_paths = []

        # Re-run each of the top solutions
        for seed_idx, (solution, fitness) in enumerate(zip(seed_solutions, seed_fitness)):
            print(f"  Solution {seed_idx+1}/{len(seed_solutions)}: fitness={fitness:.6f}")

            # Parse solution based on failure_option
            if failure_option == 2.1:
                # Velocity single failure: [agent_id, fail_time, v_n, v_e, v_d]
                fail_agent_id = int(solution[0])
                fail_time = solution[1]
                fail_v_n = solution[2]
                fail_v_e = solution[3]
                fail_v_d = solution[4]

                # Build failure config
                # IMPORTANT: Use numpy arrays (not lists) for fail_agent_num, fail_time, etc.
                # This matches the format used in failure_modes_problem.py
                sim_fail_config = {
                    'failure_option': [2],  # Velocity failure
                    'fail_agent_num': [np.array([fail_agent_id])],
                    'fail_time': [np.array([fail_time])],
                    'fail_v_n': [np.array([fail_v_n])],
                    'fail_v_e': [np.array([fail_v_e])],
                    'fail_v_d': [np.array([fail_v_d])],
                    'fail_in_swarm': [True]
                }

            elif failure_option in [1, 1.1]:
                # Power reduction/cutoff: [agent_id, fail_time, fail_power]
                fail_agent_id = int(solution[0])
                fail_time = solution[1]
                fail_power = solution[2]

                # IMPORTANT: Use numpy arrays (not lists) for fail_agent_num, fail_time, etc.
                # This matches the format used in failure_modes_problem.py
                sim_fail_config = {
                    'failure_option': [int(failure_option)],
                    'fail_agent_num': [np.array([fail_agent_id])],
                    'fail_time': [np.array([fail_time])],
                    'fail_power': [np.array([fail_power])],
                    'fail_in_swarm': [True]
                }
            else:
                print(f"    WARNING: Unsupported failure_option {failure_option}")
                continue

            # Build simulation config
            sim_config = {
                'end_time': [end_time],
                'start_x': [start_x],
                'end_x': [end_x],
                'env_config_type': [env_config_type],
                'env_maps': [trial_num],
                'env_starts': [trial_num]
            }

            swarm_config = {
                'num_agents': int(num_agents),
                'd_ref': 10.0,
                'v_ref': v_ref,
                'r': range_aware
            }

            # Create subdirectory for this specific solution
            solution_output_dir = trial_output_dir / f"seed{seed_idx:02d}"
            solution_output_dir.mkdir(exist_ok=True)

            # Configure flags with trajectory saving enabled
            flags = {
                'output_all_states': True,   # Enable trajectory data
                'save': True,                 # Save .mat files
                'viz': False,
                'viz_video': False,
                'dynamic_end': True,
                'run_parallel': False,
                'cooperating': True,
                'ACTIVE_ARENA_WALLS': True
            }

            # Run simulation
            try:
                result = simulator.failure_modes_simulate(
                    sim_config=sim_config,
                    swarm_config=swarm_config,
                    sim_fail_config=sim_fail_config,
                    results_dir=str(solution_output_dir),
                    flags=flags
                )

                # The .mat file should be saved to solution_output_dir/iter001/state_var.mat
                mat_file_path = solution_output_dir / "iter001" / "state_var.mat"

                if mat_file_path.exists():
                    print(f"    ✓ Trajectory saved to: {mat_file_path}")
                    trial_trajectory_paths.append(mat_file_path)
                else:
                    print(f"    WARNING: Expected trajectory file not found: {mat_file_path}")
                    trial_trajectory_paths.append(None)

            except Exception as e:
                print(f"    ERROR: Failed to re-run simulation: {e}")
                trial_trajectory_paths.append(None)

        # Store paths for this trial
        trajectory_paths[(num_agents, trial_num)] = trial_trajectory_paths

    # Shutdown simulator
    simulator.shutdown_engine()

    print(f"\n✓ Re-run complete! Generated trajectory files for {len(trajectory_paths)} trials")
    print(f"  Trajectories saved to: {trajectory_output_dir}")

    return trajectory_paths


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rerun_top10_with_trajectories.py <ga_results_dir>")
        print("Example: python rerun_top10_with_trajectories.py results_to_save/result_20260107_151408_GA_fail1_velocity_single")
        sys.exit(1)

    ga_results_dir = sys.argv[1]

    trajectory_paths = rerun_top10_solutions_with_trajectories(
        ga_results_dir=ga_results_dir,
        failure_option=2.1
    )

    print("\nGenerated trajectory paths:")
    for (num_agents, trial_num), paths in sorted(trajectory_paths.items()):
        print(f"\n  {num_agents} agents, trial {trial_num}:")
        for i, path in enumerate(paths):
            if path:
                print(f"    Seed {i}: {path}")
            else:
                print(f"    Seed {i}: FAILED")
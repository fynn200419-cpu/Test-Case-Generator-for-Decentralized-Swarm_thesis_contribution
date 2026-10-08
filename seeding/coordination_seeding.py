#coordination seeding

#plan is the following:
#AI -check
#look where the data gets stored of drones position
#write function to dtermine the point in space when drone fails
#write function to check the size n2 functional run, which drone is the closest to that point at the time of failure
#assign the failure to that drone
#give that new X vector to initial population

"""
Coordination-based seeding for genetic algorithms.

This module implements spatial coordination-based seeding, which allows
transferring good solutions across different swarm sizes by matching
agents based on their spatial locations rather than their IDs.
"""

import numpy as np
import pickle
from pathlib import Path
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from trajectory_utils import load_trajectory_data, get_num_agents_from_trajectory


def get_failure_spatial_location(pos_ned_history, time_history, agent_id, fail_time):
    """
    Find the (x, y, z) position where a specific agent was when failure occurred.

    Parameters
    ----------
    pos_ned_history : ndarray
        Position trajectory array, shape (num_timesteps, 3*num_agents)
        Format: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
    time_history : ndarray
        Time vector, shape (num_timesteps,)
    agent_id : int
        Which agent failed (1-indexed from MATLAB)
    fail_time : float
        When the failure occurred (in seconds)

    Returns
    -------
    tuple
        (x, y, z) - spatial coordinates at failure time

    Raises
    ------
    ValueError
        If agent_id is out of valid range or time_history is empty

    Example
    -------
    >>> pos_ned = np.random.rand(1000, 30)  # 10 agents, 1000 timesteps
    >>> time = np.linspace(0, 100, 1000)
    >>> xyz = get_failure_spatial_location(pos_ned, time, agent_id=5, fail_time=10.5)
    >>> print(xyz)  # (x, y, z) coordinates
    """
    if time_history.size == 0:
        raise ValueError("time_history cannot be empty")

    num_agents = get_num_agents_from_trajectory(pos_ned_history)

    if agent_id < 1 or agent_id > num_agents:
        raise ValueError(
            f"agent_id must be between 1 and {num_agents}, got {agent_id}"
        )

    # Find timestep index closest to fail_time (nearest neighbor, no interpolation)
    time_idx = np.argmin(np.abs(time_history - fail_time))

    # Convert agent_id to 0-indexed
    agent_idx = agent_id - 1 #Matlab has agent 1,2,3... Pythn uses idx which starts at 0,1,2,3...

    # Extract position columns for this agent: [x, y, z]
    col_indices = [3 * agent_idx, 3 * agent_idx + 1, 3 * agent_idx + 2] #this is to get the exact columns that belong to the agent id. Eg. Agentid =3, idx=2 and we have to have [3*2, 3*2+1. 3*2+2] = [6,7,8] which makes sense
    position = pos_ned_history[time_idx, col_indices]

    return tuple(position)


def find_closest_agent_at_time(pos_ned_history, time_history, target_xyz, target_time):
    """
    Find which agent is closest to target spatial location at target time.

    Parameters
    ----------
    pos_ned_history : ndarray
        Position trajectory array for NEW swarm, shape (num_timesteps, 3*num_agents)
    time_history : ndarray
        Time vector for NEW swarm, shape (num_timesteps,)
    target_xyz : tuple or array-like
        Target position (x, y, z) - failure location from original run
    target_time : float
        When the failure occurred (in seconds)

    Returns
    -------
    int
        Agent ID (1-indexed) that is closest to target location at target time

    Raises
    ------
    ValueError
        If time_history is empty or target_xyz doesn't have 3 elements

    Example
    -------
    >>> pos_ned = np.random.rand(1000, 33)  # 11 agents, 1000 timesteps
    >>> time = np.linspace(0, 100, 1000)
    >>> target_xyz = (50.0, 25.0, -10.0)
    >>> closest_agent = find_closest_agent_at_time(pos_ned, time, target_xyz, 10.5)
    >>> print(closest_agent)  # e.g., 6
    """
    if time_history.size == 0:
        raise ValueError("time_history cannot be empty")

    target_xyz = np.array(target_xyz)
    if target_xyz.shape != (3,):
        raise ValueError(f"target_xyz must have 3 elements, got {target_xyz.shape}")

    # Find timestep closest to target_time (nearest neighbor, no interpolation)
    time_idx = np.argmin(np.abs(time_history - target_time))

    # Get number of agents
    num_agents = get_num_agents_from_trajectory(pos_ned_history)

    # Extract all agent positions at this timestep
    positions_flat = pos_ned_history[time_idx, :]  # Shape: (3*num_agents,)

    # Reshape to (num_agents, 3) for easier distance calculation
    positions = positions_flat.reshape(num_agents, 3)

    # Calculate Euclidean distance from each agent to target
    distances = np.linalg.norm(positions - target_xyz, axis=1)

    # Find agent with minimum distance
    closest_agent_idx = np.argmin(distances)

    # Convert to 1-indexed (MATLAB convention)
    closest_agent_id = closest_agent_idx + 1

    return closest_agent_id


def transform_seed_to_coordination(seed_solution, old_traj_data, new_traj_data,
                                   failure_option, trial_idx=0):
    """
    Transform a seed solution from one swarm size to another using coordination.

    Takes a seed solution with agent IDs from an old swarm size, finds the
    spatial locations where failures occurred, and maps them to the closest
    agents in a new swarm size.

    Parameters
    ----------
    seed_solution : ndarray
        Seed solution from previous run, shape depends on failure_option
        For velocity single (2.1): [agent_id, fail_time, v_n, v_e, v_d]
        For power (1): [agent_id, fail_time, fail_power]
    old_traj_data : dict
        Trajectory data from load_trajectory_data() for old swarm size
    new_traj_data : dict
        Trajectory data from load_trajectory_data() for new swarm size
    failure_option : float
        Type of failure (1, 1.1, 2.1, etc.) to determine solution structure
    trial_idx : int
        Which trial/map to use from trajectory data (default: 0)

    Returns
    -------
    ndarray
        Transformed seed solution with new agent IDs for new swarm size

    Raises
    ------
    ValueError
        If trial_idx is out of range or failure_option is not supported

    Example
    -------
    >>> old_seed = np.array([5, 10.5, -3.2, 4.1, -2.0])  # velocity single failure
    >>> new_seed = transform_seed_to_coordination(
    ...     old_seed, old_traj_10agents, new_traj_11agents, failure_option=2.1
    ... )
    >>> print(new_seed)  # e.g., [6, 10.5, -3.2, 4.1, -2.0] (agent ID changed)
    """
    # Validate trial_idx
    if trial_idx >= old_traj_data['num_runs'] or trial_idx >= new_traj_data['num_runs']:
        raise ValueError(
            f"trial_idx {trial_idx} out of range. "
            f"Old has {old_traj_data['num_runs']}, new has {new_traj_data['num_runs']} runs"
        )

    # Extract trajectory data for this trial
    old_pos_ned = old_traj_data['pos_ned'][trial_idx]
    old_time = old_traj_data['time'][trial_idx]
    new_pos_ned = new_traj_data['pos_ned'][trial_idx]
    new_time = new_traj_data['time'][trial_idx]

    # Create copy of seed solution to modify
    new_solution = seed_solution.copy()

    # Determine structure based on failure_option
    if failure_option == 1 or failure_option == 1.1:
        # Power reduction/cutoff: [agent_id, fail_time, fail_power]
        old_agent_id = int(seed_solution[0])
        fail_time = seed_solution[1]

        # Find spatial location of failure in old swarm
        failure_xyz = get_failure_spatial_location(
            old_pos_ned, old_time, old_agent_id, fail_time
        )

        # Find closest agent in new swarm
        new_agent_id = find_closest_agent_at_time(
            new_pos_ned, new_time, failure_xyz, fail_time
        )

        # Update agent ID in solution
        new_solution[0] = new_agent_id

    elif failure_option == 2.1:
        # Velocity change single: [agent_id, fail_time, v_n, v_e, v_d]
        old_agent_id = int(seed_solution[0])
        fail_time = seed_solution[1]

        # Find spatial location of failure in old swarm
        failure_xyz = get_failure_spatial_location(
            old_pos_ned, old_time, old_agent_id, fail_time
        )

        # Find closest agent in new swarm
        new_agent_id = find_closest_agent_at_time(
            new_pos_ned, new_time, failure_xyz, fail_time
        )

        # Update agent ID in solution (keep velocity components unchanged)
        new_solution[0] = new_agent_id

    elif failure_option in [2.2, 2.3]:
        # Multiple velocity changes: more complex handling needed
        # For now, assume first agent is at index 0, second at index 5, etc.
        # This would need refinement based on actual problem structure
        raise NotImplementedError(
            f"Coordination seeding for failure_option {failure_option} "
            "with multiple failures not yet implemented"
        )

    else:
        raise ValueError(f"Unknown failure_option: {failure_option}")

    return new_solution


class CoordinationSeedSampling(Sampling):
    """
    Seeds GA population using spatial coordination instead of agent IDs.

    This class enables transferring good solutions from one swarm size to another
    by matching agents based on their spatial locations during failure events,
    rather than by their ID numbers.

    Workflow:
    1. Load seed solutions (agent_id, fail_time, ...) from previous runs
    2. Load trajectory data for both old and new swarm sizes
    3. Convert seed agent IDs to spatial coordinates using old trajectories
    4. Find matching agents in new swarm based on spatial proximity
    5. Create transformed seed solutions with new agent IDs
    6. Mix with random solutions to create initial population (34% seeds, 66% random)

    Parameters
    ----------
    seed_solutions : ndarray
        Seed solutions from previous run, shape (n_seeds, n_var)
    seed_fitness : ndarray, optional
        Fitness values for each seed, shape (n_seeds,)
    old_traj_path : str or Path
        Path to .mat file with trajectory data for old swarm size
    new_traj_path : str or Path
        Path to .mat file with trajectory data for new swarm size
    failure_option : float
        Type of failure (1, 1.1, 2.1, etc.)
    trial_idx : int, optional
        Which trial/map to use from trajectory data (default: 0)

    Attributes
    ----------
    transformed_seeds : ndarray
        Seed solutions with agent IDs transformed to new swarm size
    transformed_fitness : ndarray or None
        Fitness values (unchanged from input)
    """

    def __init__(self, seed_solutions, seed_fitness=None,
                 old_traj_data_list=None, new_traj_path=None,
                 failure_option=None, trial_idx=0):
        super().__init__()

        # Store original seeds
        self.seed_solutions = np.asarray(seed_solutions, dtype=float)
        self.seed_fitness = None
        if seed_fitness is not None:
            self.seed_fitness = np.asarray(seed_fitness, dtype=float).flatten()

        # Load trajectory data
        if old_traj_data_list is None or new_traj_path is None:
            raise ValueError("Both old_traj_data_list and new_traj_path must be provided")

        if failure_option is None:
            raise ValueError("failure_option must be provided")

        if len(old_traj_data_list) != len(seed_solutions):
            raise ValueError(
                f"Number of trajectory files ({len(old_traj_data_list)}) must match "
                f"number of seeds ({len(seed_solutions)})"
            )

        print(f"[COORDINATION SEEDING] Loading functional trajectory data...")
        print(f"  New swarm (functional): {new_traj_path}")

        self.old_traj_data_list = old_traj_data_list
        self.new_traj_data = load_trajectory_data(new_traj_path)
        self.failure_option = failure_option
        self.trial_idx = trial_idx

        # Transform all seeds
        print(f"[COORDINATION SEEDING] Transforming {len(seed_solutions)} seeds...")
        self.transformed_seeds = self._transform_all_seeds()

        # Random sampler for non-seeded population
        self.random_sampler = FloatRandomSampling()

    def _transform_all_seeds(self):
        """Transform all seed solutions to new swarm size.

        Each seed uses its corresponding trajectory file from old_traj_data_list.
        """
        transformed = []

        for i, seed in enumerate(self.seed_solutions):
            try:
                # Use the trajectory file corresponding to this seed
                old_traj_data = self.old_traj_data_list[i] #e.g. i=3, get the trajectory of seed03/iter001/state_var.mat

                new_seed = transform_seed_to_coordination(
                    seed,
                    old_traj_data,
                    self.new_traj_data,
                    self.failure_option,
                    trial_idx=0  # Always use first run since each .mat file has only 1 run
                )
                transformed.append(new_seed)

                if i == 0:
                    # Print first transformation for debugging
                    print(f"    Example transformation (seed 0):")
                    print(f"      Old agent ID: {int(seed[0])}")
                    print(f"      New agent ID: {int(new_seed[0])}")
                    print(f"      Fail time: {seed[1]:.2f}s")

            except Exception as e:
                print(f"    Warning: Failed to transform seed {i}: {e}")
                print(f"    Keeping original seed")
                transformed.append(seed)

        return np.array(transformed)

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate initial population with coordination-transformed seeds.

        Parameters
        ----------
        problem : pymoo Problem
            The optimization problem
        n_samples : int
            Total population size

        Returns
        -------
        ndarray
            Initial population, shape (n_samples, n_var)
        """
        n_seed = self.transformed_seeds.shape[0]

        # Validate problem dimensions
        if problem.n_var != self.transformed_seeds.shape[1]:
            raise ValueError(
                f"Problem dimension mismatch: problem.n_var={problem.n_var} "
                f"vs seed columns={self.transformed_seeds.shape[1]}"
            )

        # Cap seeds to max 34% of population (ensure >=66% random)
        max_seed_pct = 0.34
        max_seeds = int(np.floor(max_seed_pct * n_samples))
        seeds_to_use = min(n_seed, max_seeds)

        # Generate full population of random samples first
        X = self.random_sampler._do(problem, n_samples, **kwargs) #generates problem related random samples for intial population
        if X.ndim == 1:
            X = np.atleast_2d(X)

        # If no seeds to use, return random population
        if seeds_to_use <= 0:
            return X

        # Select best seeds if fitness data available
        if self.seed_fitness is not None:
            # Sort by fitness ascending (lower = better for minimization)
            best_idx = np.argsort(self.seed_fitness)[:seeds_to_use]
            seed_rows = self.transformed_seeds[best_idx]
            best_fitness = self.seed_fitness[best_idx]
            print(f"[COORDINATION SEEDING] Using {seeds_to_use} best seeds (fitness: {best_fitness})")
        else:
            # No fitness info: take first seeds_to_use rows
            seed_rows = self.transformed_seeds[:seeds_to_use]
            print(f"[COORDINATION SEEDING] Using {seeds_to_use} seeds (no fitness ranking)")

        # Overwrite first part of random population with seeds
        X[:seeds_to_use] = seed_rows

        return X


def build_coordination_seeded_sampler(seed_pickle_path, old_traj_dir, new_traj_path,
                                      problem, failure_option, trial_idx=0, n_seed=10, n_pool=20):
    """
    Factory function to create CoordinationSeedSampling from saved seeds and trajectories.

    Parameters
    ----------
    seed_pickle_path : str or Path
        Path to pickle file with {'X': ndarray, 'F': ndarray}
    old_traj_dir : str or Path
        Path to DIRECTORY containing trajectory files for old swarm size
        Expected structure: old_traj_dir/seed00/iter001/state_var.mat, seed01/iter001/state_var.mat, etc.
    new_traj_path : str or Path
        Path to .mat file with trajectory data for new swarm size (functional baseline)
    problem : pymoo Problem
        Current problem instance (for validation)
    failure_option : float
        Type of failure (1, 1.1, 2.1, etc.)
    trial_idx : int, optional
        Which trial/map to use from new_traj (default: 0) because we loop through the maps beforehand and this makes sure that the only run in this file gets selected
    n_seed : int, optional
        Max number of seeds to use (default: 10)

    Returns
    -------
    CoordinationSeedSampling or FloatRandomSampling
        Coordination seeding sampler if successful, else random sampler
    """
    # Validate inputs
    if seed_pickle_path is None:
        print("[COORDINATION SEEDING] No seed path provided, using random sampling")
        return FloatRandomSampling()

    if old_traj_dir is None or new_traj_path is None:
        print("[COORDINATION SEEDING] Missing trajectory paths, using random sampling")
        return FloatRandomSampling()

    # Load seed data
    try:
        with open(seed_pickle_path, 'rb') as f:
            seed_data = pickle.load(f)
    except Exception as e:
        print(f"[COORDINATION SEEDING] Failed to load seeds: {e}")
        return FloatRandomSampling()

    # Extract X and F
    X_prev = seed_data.get('X')
    F_prev = seed_data.get('F')

    if X_prev is None or F_prev is None:
        print("[COORDINATION SEEDING] Seed data missing 'X' or 'F', using random sampling")
        return FloatRandomSampling()

    X_prev = np.asarray(X_prev, dtype=float)
    F_prev = np.asarray(F_prev, dtype=float)

    # Ensure 2D
    if X_prev.ndim == 1:
        X_prev = np.atleast_2d(X_prev)
    if F_prev.ndim != 1:
        F_prev = F_prev.flatten()

    # Validate dimensions
    if X_prev.shape[1] != problem.n_var:
        print(
            f"[COORDINATION SEEDING] ERROR: Seed has {X_prev.shape[1]} columns "
            f"but problem.n_var={problem.n_var}"
        )
        return FloatRandomSampling()

    # Ensure fitness and X match
    if F_prev.shape[0] != X_prev.shape[0]:
        print(f"[COORDINATION SEEDING] Warning: F and X size mismatch, trimming")
        min_rows = min(F_prev.shape[0], X_prev.shape[0])
        X_prev = X_prev[:min_rows]
        F_prev = F_prev[:min_rows]

    # Sort all available seeds by fitness (best first) and collect the first n_seed
    # that are within the current problem bounds. This handles constraint seeding,
    # which may have tightened problem.xu after the previous run was saved:
    # seeds whose fail_time now exceeds the bound are skipped, and the next-best
    # valid seed is tried instead. seed{rank:02d}/ folders must exist up to the
    # highest rank that is actually selected.
    sorted_ranks = np.argsort(F_prev)[:n_pool]
    all_X_sorted = X_prev[sorted_ranks]
    all_F_sorted = F_prev[sorted_ranks]

    valid_seeds   = []
    valid_fitness = []
    valid_ranks   = []  # sorted rank → maps to seed{rank:02d}/ trajectory folder

    for rank in range(len(all_X_sorted)):
        if len(valid_seeds) >= n_seed:
            break
        seed    = all_X_sorted[rank]
        fitness = all_F_sorted[rank]
        if np.all(seed >= problem.xl) and np.all(seed <= problem.xu):
            valid_seeds.append(seed)
            valid_fitness.append(fitness)
            valid_ranks.append(rank)
        else:
            print(f"[COORDINATION SEEDING] Skipping seed rank {rank} (fitness={fitness:.4f}): variables out of current bounds")

    if not valid_seeds:
        print("[COORDINATION SEEDING] No valid seeds within bounds, using random sampling")
        return FloatRandomSampling()

    seed_solutions = np.array(valid_seeds)
    seed_fitness   = np.array(valid_fitness)

    print(f"[COORDINATION SEEDING] Selected {len(valid_seeds)}/{n_seed} seeds within bounds from {seed_pickle_path}")

    # Load trajectory files for the selected seeds using their sorted rank as the
    # folder index (seed{rank:02d}/iter001/state_var.mat)
    old_traj_dir = Path(old_traj_dir)
    old_traj_data_list = []

    print(f"[COORDINATION SEEDING] Loading {len(seed_solutions)} trajectory files from {old_traj_dir}")

    for rank in valid_ranks:
        seed_dir = old_traj_dir / f"seed{rank:02d}" / "iter001" / "state_var.mat"

        if not seed_dir.exists():
            print(f"[COORDINATION SEEDING] ERROR: Trajectory file not found: {seed_dir}")
            print(f"[COORDINATION SEEDING] Falling back to random sampling")
            return FloatRandomSampling()

        try:
            traj_data = load_trajectory_data(seed_dir)
            old_traj_data_list.append(traj_data)
        except Exception as e:
            print(f"[COORDINATION SEEDING] ERROR: Failed to load trajectory {seed_dir}: {e}")
            print(f"[COORDINATION SEEDING] Falling back to random sampling")
            return FloatRandomSampling()

    print(f"[COORDINATION SEEDING] Successfully loaded {len(old_traj_data_list)} trajectory files")

    # Create coordination seeding sampler
    try:
        sampler = CoordinationSeedSampling(
            seed_solutions=seed_solutions,
            seed_fitness=seed_fitness,
            old_traj_data_list=old_traj_data_list,
            new_traj_path=new_traj_path,
            failure_option=failure_option,
            trial_idx=trial_idx
        )
        return sampler
    except Exception as e:
        print(f"[COORDINATION SEEDING] Failed to create sampler: {e}")
        return FloatRandomSampling()
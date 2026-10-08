"""
Constraint-based initialization that tightens the GA time bounds
to the obstacle clearance region.

Steps:
1. Load obstacle map data from .mat file, find max obstacle x-position
2. Load functional trajectory, find when last drone crosses that x
3. Update problem.xu for time variables to that normalized value
"""

import numpy as np
from scipy.io import loadmat

from src.constants import SIM_DIR
from src.trajectory_utils import load_trajectory_data


# Map file paths relative to SIM_DIR/data/
MAP_FILES = {
    1: '20250822_144327_predefined_maps_obst_1sph_1cyl_100trials_50cross.mat',
}


def get_max_obstacle_x(env_config_type, trial_num):
    """
    Get the maximum x-extent (trailing edge) of all obstacles for a given map.

    Parameters
    ----------
    env_config_type : int
        Environment config type (1 = obstacles, 2 = no obstacles)
    trial_num : int
        Trial/map index (1-indexed, as used in MATLAB)

    Returns
    -------
    float
        Maximum x-coordinate of the trailing edge of all obstacles.
        Returns 0.0 if no obstacles (env_config_type=2).
    """
    if env_config_type == 2:
        return 0.0

    if env_config_type not in MAP_FILES:
        raise ValueError(
            f"env_config_type={env_config_type} not supported for constraint seeding. "
            f"Supported types: {list(MAP_FILES.keys())} and 2 (no obstacles)."
        )

    # Load the .mat map file
    map_file = SIM_DIR / 'data' / MAP_FILES[env_config_type]
    mat_data = loadmat(str(map_file), squeeze_me=False, struct_as_record=False)
    save_maps = mat_data['save_maps']

    # Access the map struct for this trial (MATLAB 1-indexed → Python 0-indexed)
    # Each cell in the MATLAB cell array is wrapped in a (1,1) object array, so unwrap with [0,0]
    map_struct = save_maps[0, 0].maps[0, trial_num - 1][0, 0]

    trailing_edges = []

    # Cylinders: buildings_width is the diameter, buildings_north is center
    n_cyl = int(np.asarray(map_struct.n_cyl).flat[0])
    if n_cyl > 0:
        buildings_north = np.asarray(map_struct.buildings_north).flatten()
        buildings_width = np.asarray(map_struct.buildings_width).flatten()
        cyl_edges = buildings_north + buildings_width / 2.0
        trailing_edges.extend(cyl_edges.tolist())

    # Spheres: spheres_north is center, spheres_r is radius
    n_spheres = int(np.asarray(map_struct.n_spheres).flat[0])
    if n_spheres > 0:
        spheres_north = np.asarray(map_struct.spheres_north).flatten()
        spheres_r = np.asarray(map_struct.spheres_r).flatten()
        sph_edges = spheres_north + spheres_r
        trailing_edges.extend(sph_edges.tolist())

    if not trailing_edges:
        return 0.0

    return max(trailing_edges)


def apply_obstacle_time_constraint(problem, env_config_type, trial_num, functional_traj_path):
    """
    Constrain the GA time variable upper bound to when the last drone
    clears the last obstacle in the functional (no-failure) run.

    Modifies problem.xu in-place. Returns nothing — the caller is responsible
    for constructing the sampling strategy afterwards.

    Parameters
    ----------
    problem : FailureModesProblem
        The GA problem instance (xu will be modified in-place)
    env_config_type : int
        Environment config type
    trial_num : int
        Trial/map index (1-indexed)
    functional_traj_path : str or Path
        Path to functional trajectory .mat file (state_var.mat)
    """
    # Step 1: Get max obstacle x
    max_obs_x = get_max_obstacle_x(env_config_type, trial_num)

    if max_obs_x <= 0:
        print(f"[CONSTRAINT] No obstacles found (env_config_type={env_config_type}), skipping constraint")
        return

    # Step 2: Load functional trajectory and find when last drone crosses max_obs_x
    traj_data = load_trajectory_data(functional_traj_path)
    pos_ned = traj_data['pos_ned'][0]   # shape (num_timesteps, 3*num_agents)
    time_arr = traj_data['time'][0]     # shape (num_timesteps,)

    # Extract x (north) positions: columns 0, 3, 6, ...
    pos_x = pos_ned[:, 0::3]
    num_agents = pos_x.shape[1]

    crossing_times = []
    for agent_idx in range(num_agents):
        agent_x = pos_x[:, agent_idx]
        crossing_indices = np.where(agent_x > max_obs_x)[0]
        if len(crossing_indices) > 0:
            crossing_times.append(time_arr[crossing_indices[0]])

    if not crossing_times:
        print(f"[CONSTRAINT] WARNING: No drone crossed obstacle x={max_obs_x:.1f}, skipping constraint")
        return

    clearance_time = max(crossing_times)

    # Step 3: Compute normalized bound and update problem.xu
    flight_start = problem.flight_start
    flight_end = problem.flight_end
    time_range = flight_end - flight_start

    if time_range <= 0:
        print(f"[CONSTRAINT] WARNING: Invalid time range, skipping constraint")
        return

    normalized_bound = (clearance_time - flight_start) / time_range
    normalized_bound = np.clip(normalized_bound, 0.01, 1.0)

    # Determine time variable indices in xu based on failure_option
    num_failures = problem.num_failures
    if problem.failure_option in [1, 1.1, 2]:
        # [agent_ids (num_failures), fail_times (num_failures), ...]
        time_start_idx = num_failures
        time_end_idx = 2 * num_failures
    elif problem.failure_option == 2.1:
        # [single_agent_id (1), fail_times (num_failures), ...]
        time_start_idx = 1
        time_end_idx = 1 + num_failures
    else:
        print(f"[CONSTRAINT] WARNING: Unknown failure_option={problem.failure_option}, skipping constraint")
        return

    # Update upper bounds for time variables
    problem.xu[time_start_idx:time_end_idx] = normalized_bound

    print(f"[CONSTRAINT] Map {trial_num}: max obstacle x = {max_obs_x:.1f}")
    print(f"[CONSTRAINT]   Last drone clearance time = {clearance_time:.2f}s")
    print(f"[CONSTRAINT]   Time bound: {normalized_bound:.3f} (was 1.0)")
    print(f"[CONSTRAINT]   Effective time range: [{flight_start:.2f}, {flight_start + normalized_bound * time_range:.2f}]s")

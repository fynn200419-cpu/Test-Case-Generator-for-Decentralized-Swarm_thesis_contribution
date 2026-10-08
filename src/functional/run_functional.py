# Run a "functional" test to:
# 1. Check if safety violation occurs, and if not...
# 2. Check how long it takes the group to reach goal

import numpy as np
import pandas as pd
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR
from src.utils import make_dataframe, create_results_dir


def run_trial(vect_agents, run_type="functional", num_neig=10, rad_neig=150, map_option=0, cooperating=True,
              flag_parallel=False):
    """
    Run a functional trial of the swarm simulation system.

    This function configures and executes a set of simulation runs to assess system performance under different
    swarm sizes.

    Parameters:
        :param flag_parallel: (bool) Whether to run the simulation in parallel mode. Default is False.
        :param num_neig: (int) Number of neighbors for the swarm. Default is 10.
        :param run_type: (str) Name of the run type.
        :param vect_agents: (np.array) Array of swarm sizes to run.

    Procedure:
    1. Configures simulation parameters, including agent sizes, failure conditions, and time constraints.
    2. Initializes the simulator and organizes inputs for each trial.
    3. Runs the simulation for each configuration, collecting outputs such as collision results and flight times.
    4. Saves the results to a dataframe, exporting the data in both pickle and CSV formats.

    Key Simulation Settings:
        - Number of agents: Ranges from 100 to 10 (in descending order, by step size of 5).
        - End time: Fixed at 30 seconds.
        - Failure conditions: None (functional test only).

    Returns:
        None

    Outputs:
        - Results are saved as a dataframe, including simulation parameters, collision data, and timing metrics.
        - Data is exported to files in the specified output directory.

    Notes:
        - The dataframe is created using the `make_dataframe` utility.
        - Results are saved using the `save_dataframe` utility, which outputs files in both pickle and CSV formats.
    """

    ##############
    #  Options   #
    ##############

    # Flags (dictionary -> struct)
    flags = {'run_parallel': flag_parallel, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig':num_neig,
             'map_option': map_option, 'rad_neig': rad_neig, 'cooperating': cooperating}

    # Number of agents - functional test will be run once per agent size
    num_agents = vect_agents
    num_agents = np.flip(num_agents)  # Flip to go in descending order, see if that helps with runtime
    # num_agents = np.arange(10, 30, 10)  # for debugging

    # Number of simulation runs based on swarm sizes
    num_runs = len(num_agents)

    # End time
    max_end_time = np.full(num_runs, 30)

    # Start and end planes
    start_x = np.full(num_runs, 0)
    end_x = np.full(num_runs, 100)

    ################
    #  End Options #
    ################

    # Initialize the simulator object
    sim = SimulatorSwarmlab(str(SIM_DIR))

    # Failure conditions - no failures in functional test
    fail_agent_num = np.full(num_runs, np.nan)
    fail_time = np.full(num_runs, np.nan)
    fail_in_swarm = np.full(num_runs, np.nan)

    # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
    x = np.column_stack((max_end_time, start_x, end_x, num_agents, fail_agent_num, fail_time, fail_in_swarm))

    # Run the simulation
    coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end = sim.simulate(x, flags)

    # Trial number (0 for functional tests; deterministic with no failures)
    trial = np.full(num_runs, 0, dtype=int)

    # Seed number (nan for functional tests; no failures)
    seed_num = np.full(num_runs, np.nan)

    # Set generation and population to 0
    save_pop_size = np.full(num_runs, 0)  # For random search, save iterations to "pop_size" field
    save_num_gen = np.full(num_runs, 0)

    # Save results to new dataframe
    df = make_dataframe(trial, seed_num, save_pop_size, save_num_gen, x, coll_result,
                        flight_time, flight_start, flight_end, existing_df=None)

    # Create directory and save data (pickle and csv)
    folder_path = create_results_dir(run_type)
    pickle_path = folder_path / "dataframe.pkl"
    df.to_pickle(pickle_path)
    csv_path = folder_path / "dataframe.csv"
    df.to_csv(csv_path, index=False, na_rep='NaN')

    # # Test out saving a second set of data
    # folder_path = save_dataframe(df, run_type, 0, 1, exist_folder_path=folder_path, to_csv=True)

    return folder_path


def run_trial_new_wrapper(vect_agents, run_type="functional", swarm_r=float('inf'),
                          env_config_type=0, num_trials=1,
                         cooperating=True, flag_parallel=False, end_time=300.0, start_x=0.0, end_x=100.0):
    """
    Run a functional trial using the new MATLAB wrapper_run_generation_exp_failure_modes function.
    
    This function configures and executes simulation runs with no failures to assess baseline system performance 
    under different swarm sizes using the new wrapper infrastructure.
    
    Parameters:
        vect_agents (np.array): Array of swarm sizes to test
        run_type (str): Name identifier for the run type
        num_neig (int): Number of neighbors for the swarm (currently unused in new wrapper)
        rad_neig (float): Neighborhood radius (currently unused in new wrapper) 
        map_option (int): Map selection option (0=default, 1=obsL, 2=obsR)
        cooperating (bool): Whether agents cooperate
        flag_parallel (bool): Whether to run simulations in parallel
        end_time (float): Maximum simulation time
        start_x (float): Start plane x-coordinate
        end_x (float): End plane x-coordinate
        
    Returns:
        Path: Path to the results directory
    """
    
    # Initialize the simulator object
    sim = SimulatorSwarmlab(str(SIM_DIR))
    
    # Number of agents - functional test will be run once per agent size
    num_agents = np.flip(vect_agents)  # Flip to descending order for potential runtime benefits
    num_runs = len(num_agents) * num_trials
    
    # Configure simulation parameters for each run (must be lists within the dictionary for MATLAB
    sim_config = {
        'end_time': [end_time] * num_trials,
        'start_x': [start_x] * num_trials,
        'end_x': [end_x] * num_trials,
        'env_config_type': [env_config_type] * num_trials,
        'env_maps': np.arange(1, num_trials+1).tolist(),  # MATLAB indexing starts at 1
        'env_starts': np.arange(1, num_trials+1).tolist()  # Use first start configuration
    }
    
    # Configure swarm parameters (same for all runs, will be set per individual run)
    # Note: num_agents will be set individually per run since it varies
    swarm_config = {
        'num_agents': int(num_agents[0]),  # Will be overridden per run
        'd_ref': 10.0,
        'v_ref': 6.0, 
        'r': swarm_r
    }
    
    # Configure failure parameters (no failures for functional test)
    sim_fail_config = {
        'failure_option': [0] * num_trials,  # No failures
        'fail_agent_num': [float('nan')] * num_trials,
        'fail_time': [float('nan')] * num_trials,
        'fail_power': [float('nan')] * num_trials,
        'fail_in_swarm': [True] * num_trials  # Not used when no failures
    }
    
    # Configure execution flags
    flags = {
        'output_all_states': True,
        'save': True,  # Save MATLAB results (needed for trajectory data for coordination seeding)
        'viz': False,
        'viz_video': False,
        'dynamic_end': True,
        'run_parallel': flag_parallel,
        'cooperating': cooperating,
        'ACTIVE_ARENA_WALLS': True
    }
    
    # Create results directory
    folder_path = create_results_dir(run_type)

    # Create subdirectory for MATLAB trajectory files to keep main folder clean
    trajectory_dir = folder_path / "trajectory_data"
    trajectory_dir.mkdir(exist_ok=True)

    # Collect results for each agent count
    all_results = []

    for i, n_agents in enumerate(num_agents):
        print(f"Running functional test {i+1}/{len(num_agents)} with {n_agents} agents")

        # Create separate subdirectory for each agent count to avoid overwriting
        # Structure: trajectory_data/{n_agents}agents/iter001/, iter002/, etc.
        agent_trajectory_dir = trajectory_dir / f"{int(n_agents)}agents"
        agent_trajectory_dir.mkdir(exist_ok=True)
        results_dir = str(agent_trajectory_dir)

        # Update swarm config for current agent count
        current_swarm_config = swarm_config.copy()
        current_swarm_config['num_agents'] = int(n_agents)

        # Run simulation
        result = sim.failure_modes_simulate(
            sim_config=sim_config,
            swarm_config=current_swarm_config,
            sim_fail_config=sim_fail_config,
            results_dir=results_dir,
            flags=flags
        )
        
        all_results.append({
            'num_agents': n_agents,
            'result': result,
            'sim_params': {
                'end_time': end_time,
                'start_x': start_x, 
                'end_x': end_x,
                'env_config_type': env_config_type
            }
        })
    
    # Process results into dataframe format
    dataframe_rows = []
    
    for i, res_data in enumerate(all_results):
        n_agents = res_data['num_agents']
        result = res_data['result']
        sim_params = res_data['sim_params']
        
        # Process each trial within this agent count result
        for trial_idx in range(num_trials):
            
            # Extract timing data from MATLAB result for this trial
            if ('times_start' in result) and ('times_end' in result):
                times_start = np.array(result['times_start'][trial_idx]).flatten()
                times_end = np.array(result['times_end'][trial_idx]).flatten()
                
                # Calculate flight time as difference between first over the start and last over the end
                flight_start_val = np.min(times_start) if len(times_start) > 0 else np.nan
                flight_end_val = np.max(times_end) if len(times_end) > 0 else np.nan
                flight_time_val = flight_end_val - flight_start_val if not (np.isnan(flight_start_val) or np.isnan(flight_end_val)) else np.nan
            else:
                flight_time_val = np.nan
                flight_start_val = np.nan 
                flight_end_val = np.nan
            
            # Extract collision data from MATLAB result for this trial
            collision_occurred = False
            collision_type = 'not_specified'
            
            if 'collision' in result and trial_idx < len(result['collision']):
                collision_array = result['collision'][trial_idx]
                # Convert MATLAB logical array to numpy array and check for any True values
                collision_data = np.array(collision_array)
                if collision_data.size > 0 and np.any(collision_data):
                    collision_occurred = True
                    collision_type = 'detected'
            
            # Create row for dataframe
            row = {
                'trial': sim_config['env_maps'][trial_idx],
                'seed_num': np.nan,  # No randomization
                'pop_size': 0,  # Not applicable
                'num_gen': 0,  # Not applicable
                'end_time': sim_params['end_time'],
                'start_x': sim_params['start_x'],
                'end_x': sim_params['end_x'],
                'num_agents': int(n_agents),
                'fail_agent_num': 0,  # No failures
                'fail_time': np.nan,  # No failures
                'fail_in_swarm': False,  # No failures
                'coll_result_occur': collision_occurred,  # Based on actual collision data
                'coll_result_type': collision_type,  # Based on actual collision detection
                'flight_time': round(flight_time_val,2),
                'flight_start': round(flight_start_val,2),
                'flight_end': round(flight_end_val,2)
            }
            
            dataframe_rows.append(row)
    
    # Create dataframe manually
    df = pd.DataFrame(dataframe_rows)
    
    # Ensure proper data types
    df = df.astype({
        'trial': int,
        'seed_num': float,
        'pop_size': int,
        'num_gen': int,
        'end_time': float,
        'start_x': float,
        'end_x': float,
        'num_agents': int,
        'fail_agent_num': int,
        'fail_time': float,
        'fail_in_swarm': bool,
        'coll_result_occur': bool,
        'coll_result_type': str,
        'flight_time': float,
        'flight_start': float,
        'flight_end': float
    })
    
    # Save results
    pickle_path = folder_path / "dataframe.pkl"
    df.to_pickle(pickle_path)
    csv_path = folder_path / "dataframe.csv"
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    # Shutdown simulator
    sim.shutdown_engine()
    
    return folder_path

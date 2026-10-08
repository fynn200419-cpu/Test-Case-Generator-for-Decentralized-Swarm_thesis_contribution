import numpy as np
import src.functional.run_functional as run_functional
import src.multifail.run_ga_failure_modes as ga_failure_modes


def load_and_display_results(results_path, experiment_name=""):
    """
    Load GA results and display summary information.

    Parameters:
        results_path (Path): Path to results directory
        experiment_name (str): Name of experiment for display
    """
    try:
        results_data = ga_failure_modes.load_ga_results(results_path)
        df = results_data['dataframe']

        print(f"\n{experiment_name} Results Summary:")
        print(f"Number of runs: {len(df)}")

        # Extract scalar fitness values (handle both scalar and array formats)
        fitness_values = []
        for f in df['fitness']:
            if isinstance(f, (list, np.ndarray)):
                fitness_values.append(f[0] if len(f) > 0 else np.nan)
            else:
                fitness_values.append(f)
        print(f"Best fitness values: {fitness_values}")

        # Display dynamic columns based on what's available in the dataframe
        failure_cols = ['num_agents', 'num_failures']
        failure_cols.extend([col for col in df.columns if col.startswith('fail_agent_num_')])
        failure_cols.extend([col for col in df.columns if col.startswith('fail_time_')])
        print(f"Agent failure patterns:\n{df[failure_cols].to_string()}")

    except Exception as e:
        print(f"Could not load {experiment_name} results summary: {e}")

#The simulator (Swarmlab) already has a set of maps/seeded scenarios. When you ask for 50 trials it hands you map #1..#50 (these are the “maps”).
#Step 1: run the functional baseline on those maps with NO failures. That produces a CSV with one row per trial (map) telling you if the swarm crashed or finished, how long it flew, etc.
#Step 2: you pick the maps that had no crash in the functional CSV and tell the GA to use those maps.
#Step 3: the GA does NOT change maps. It tries lots of different failure settings (which agent fails, when, how) ON THE SAME MAPS to find the failure that makes the swarm do worst. The GA returns the best failure it found per map. Those best failures are what you see in the GA dataframe.
# ========================================
# EXPERIMENT CONFIGURATIONS
# ========================================

# Functional baseline configurations; must update data_path and vect_maps manually after inspection
FUNCTIONAL_CONFIGS = {
    'rInf': {
        'run': False,
        'run_type': 'functional_rInf',
        'swarm_r': float('inf'),
        'env_config_type': 1,  # with obstacles
        'data_path': 'results_to_save/exp_swarmsize/result_20260223_173643_functional_rInf/dataframe.csv',
        'vect_maps': np.array([2,3,4,5,7,9,10,12,13,14,20,22,24,25,26,28,29,32,33,34,35,36,40,41,42,43,44,45,46,47]) 
         
                              # 30 collision-free trials. After the functional run it only includes the maps that worked for failure injection.
    },
     'rInfc': {
        'run': False,
        'run_type': 'functional_rInf',
        'swarm_r': float('inf'),
        'env_config_type': 1,  # with obstacles
        'data_path': 'results_to_save/exp_swarmsize/result_20260223_173643_functional_rInf/dataframe.csv',
        'vect_maps': np.array([2,3,4])
        #5,6,7,8,9,10,11,12,13,14,20,22,24,25,26,27,28,29,32,33,34,35,36,39,40,41,42,43,44,45,46,47])        
                              # 30 collision-free trials. After the functional run it only includes the maps that worked for failure injection.
    },
    'rInfnew': {
        'run': False,
        'run_type': 'functional_rInf',
        'swarm_r': float('inf'),
        'env_config_type': 1,  # with obstacles
        #'data_path': 'results_BA/result_20260223_173643_functional_rInf/dataframe.csv',
        'data_path': 'results_to_save/exp_swarmsize/result_20260310_143954_functional_rInf_10_14agents/dataframe.csv', 
        'vect_maps': np.array([2,3,4,5,7,9,10,12,13,14,20,22,24,25,26,28,29,32,33,34,35,36,40,41,42,43,44,45,46,47])
    },  #2,3,4,5,7,9,10,12,13,20,22,24,26,28,29,33,34,35,36,41,42,
            
}

# Common functional test parameters
DEFAULT_FUNCTIONAL_PARAMS = {
    'vect_agents': np.array([10, 14]),
    'num_trials': 50,  # At least 30; can generate extras in case of collisions
    'cooperating': True,
    'flag_parallel': False,
    'end_time': 300.0,
    'start_x': 0.0,
    'end_x': 100.0
}

# Experiment definitions: each entry specifies failure_option, failure_type, and num_failures
EXPERIMENT_CONFIGS = {
    'exp2_velocity_single_rInf_random_id': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity Coordination ',
        'seeding_method': 'coordination',
        'baseline_results_path': 'results_to_save/exp_swarmsize/result_20260303_073651_GA_fail1_rInf_10gen_10random_11coordination',  # set to folder path to skip re-running first agent size
    },
     'exp2_velocity_single_rInf_random': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity id ',
        'seeding_method': 'id',
        'baseline_results_path': 'results_to_save/exp_swarmsize/result_20260303_073651_GA_fail1_rInf_10gen_10random_11coordination',  # set to folder path to skip re-running first agent size
    },
    'exp2_velocity_single_rInf_id10input': {
        'run': False,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity Coordination ',
        'seeding_method': 'id',
        'baseline_results_path': 'results_to_save/exp_swarmsize/result_20260303_073651_GA_fail1_rInf_10gen_10random_11coordination',  # maps must be done 
    },
    'exp2_velocity_single_rInf_id20input': {
        'run': False,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity Coordination ',
        'seeding_method': 'id',
        'baseline_results_path': None,  # maps must be done
    },
    'exp2_velocity_single_rInf_coord10input': {
        'run': False,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity Coordination ',
        'seeding_method': 'coordination',
        'baseline_results_path': 'results_to_save/exp_swarmsize/result_20260303_073651_GA_fail1_rInf_10gen_10random_11coordination',  # maps must be done
    },
    'exp2_velocity_single_rInf_coord20input': {
        'run': False,
        'functional_path': FUNCTIONAL_CONFIGS['rInfnew']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInfnew']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInfnew']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInfnew']['swarm_r'],
        'vect_agents': np.array([10,14]),
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity Coordination ',
        'seeding_method': 'coordination',
        'baseline_results_path': None,  # maps must be done
    },
    
}

#Common GA parameters (can be overridden per experiment if needed)
DEFAULT_GA_PARAMS = {
    'population_size': 100,
    'end_time_mult': 2.0, # multiplier for functional end time to set GA trial end time
    'f_min_points': 0.5, # min points within r_aware to consider success
    'v_ref': 6.0, # target veclocity for fitness calculation when speed deviations happen
    'flag_parallel': True, # whether to run trials in parallel
    'flag_eng_persist': True, # whether to persist engine instances across evaluations
    'terminate': {
        'num_gen': 10, #20
        'max_time': None,  # maximum time allowed PER TRIAL/SEARCH [sec]
        'gen_stagnation': None,
        'gen_stagnation_tol': None,
        'min_f': 1e-6  # stop if fitness reaches 0 (some tolerance for machine error)
    }
}
# DEFAULT_GA_PARAMS = {  # FOR DEBUGGING
#     'population_size': 3,
#     'end_time_mult': 2.0,
#     'f_min_points': 0.5,
#     'v_ref': 6.0,
#     'flag_parallel': False,
#     'flag_eng_persist': True,
#     'terminate': {
#         'num_gen': 300,
#         'max_time': 100,  # maximum time allowed PER TRIAL/SEARCH [sec]
#         'gen_stagnation': None,
#         'gen_stagnation_tol': None,
#         'min_f': 1e-6  # stop if fitness reaches 0 (some tolerance for machine error)
#     }
# }


# ========================================
# HELPER FUNCTIONS
# ========================================

def run_functional_baseline(functional_key, **override_params):
    """
    Run a functional baseline test if enabled.

    Parameters:
        functional_key (str): Key from FUNCTIONAL_CONFIGS
        **override_params: Any parameters to override from defaults
    """
    func_config = FUNCTIONAL_CONFIGS[functional_key]

    # Check if this functional test should run
    if not func_config.get('run', False):
        print(f"Skipping functional baseline: {func_config['run_type']} (run=False)")
        return

    params = {**DEFAULT_FUNCTIONAL_PARAMS, **override_params}

    print(f"\n{'='*60}")
    print(f"Running Functional Baseline: {func_config['run_type']}")
    print(f"  Swarm awareness range: {func_config['swarm_r']}")
    print(f"  Environment config: {func_config['env_config_type']} ({'with obstacles' if func_config['env_config_type'] == 0 else 'no obstacles'})")
    print(f"  Number of trials: {params['num_trials']}")
    print(f"{'='*60}")

    run_functional.run_trial_new_wrapper(
        vect_agents=params['vect_agents'],
        run_type=func_config['run_type'],
        swarm_r=func_config['swarm_r'],
        env_config_type=func_config['env_config_type'],
        num_trials=params['num_trials'],
        cooperating=params['cooperating'],
        flag_parallel=params['flag_parallel'],
        end_time=params['end_time'],
        start_x=params['start_x'],
        end_x=params['end_x']
    )
    #builds list of map indices (num_trials), seeds and simulation inputs
    #Simulator runs each trial on each agent-count, simulator responsible for map generation and internal radomness
    #collects results and saves in dataframe at results 
    #then CSV gets inspected and updates FUNCTIONAL_CONFIGS and vect_maps to the choosen trials

    print(f"Completed functional baseline: {func_config['run_type']}")
    print(f"Update FUNCTIONAL_CONFIG['{functional_key}'] with the new data_path and vect_maps after reviewing results.\n")


def run_ga_experiment(exp_key, **override_params):
    """
    Run a GA experiment if enabled.

    Parameters:
        exp_key (str): Key from EXPERIMENT_CONFIGS
        **override_params: Any parameters to override from defaults
    """
    exp_config = EXPERIMENT_CONFIGS[exp_key]

    # Check if this experiment should run
    if not exp_config.get('run', False):
        print(f"Skipping experiment: {exp_config['description']} (run=False)")
        return

    # Merge default params with experiment-specific config and overrides
    # Priority: override_params > exp_config > DEFAULT_GA_PARAMS
    params = {**DEFAULT_GA_PARAMS, **exp_config, **override_params}

    # Get some experiment parameters - will use in printout
    vect_trials = params['vect_trials']
    vect_agents = params['vect_agents']
    env_config_type = params['env_config_type']
    terminate_config = params['terminate']

    # Build run_type string from description_short
    run_type = f"GA_fail{params['num_failures']}_{params['description_short']}"

    # Environment label for display
    env_label = 'No obstacles' if env_config_type == 2 else 'With obstacles'

    print(f"\n{'='*60}")
    print(f"Running Experiment: {params['description']}")
    print(f"  Failure Option: {params['failure_option']}")
    print(f"  Num Failures: {params['num_failures']}")
    print(f"  Awareness Range: {params['r_aware']}")
    print(f"  Environment: {env_label} (config_type={env_config_type})")
    print(f"  Num Agents: {vect_agents}")
    print(f"  Num Trials: {len(vect_trials)}")
    print(f"  Population Size: {params['population_size']}")
    print(f"  Max Time/Trial: {terminate_config.get('max_time')} sec")
    print(f"{'='*60}")

    # Run GA search
    results_path = ga_failure_modes.run_ga_failure_modes(
        vect_agents=vect_agents,
        vect_trials=vect_trials,
        failure_option=params['failure_option'],
        num_failures=params['num_failures'],
        functional_data_path=params['functional_path'],
        run_type=run_type,
        population_size=params['population_size'],
        range_aware=params['r_aware'],
        end_time_mult=params['end_time_mult'],
        f_min_points=params['f_min_points'],
        v_ref=params['v_ref'],
        env_config_type=env_config_type,
        terminate_config=terminate_config,
        flag_parallel=params['flag_parallel'],
        flag_eng_persist=params['flag_eng_persist'],
        functional_trajectory_dir=params.get('functional_trajectory_dir'),
        seeding_method=params['seeding_method'],
        baseline_results_path=params.get('baseline_results_path')
    )
    #Creates a results folder.
    #For each combination (num_agents × each trial in vect_trials) it:
    #Sets up a FailureModesProblem instance (this defines GA decision variables = failure parameters).
    #Configures a pymoo GA (population, crossover, mutation, termination).
    #The GA minimizes the fitness function defined in FailureModesProblem.
    #For each candidate (individual) the problem uses SimulatorSwarmlab to run the simulator with injected failures (fail agent IDs, fail times, etc.) on the given map/seed.
    #The GA evolves candidate failure-parameters across generations via selection/crossover/mutation. These are parameter searches only (no map modification).
    #Saves detailed results to all_results.pkl and a summary dataframe.pkl / dataframe.csv in the created results folder.

    load_and_display_results(results_path, params['description'])
    return results_path


# ========================================
# MAIN FUNCTION
# ========================================

def main():
   
    # Control which functional baselines to run via 'run' flag in FUNCTIONAL_CONFIGS

    for func_key in FUNCTIONAL_CONFIGS.keys():
        run_functional_baseline(func_key)

    #50 maps were run functionally. Some crashed due to collisions. 

    # TESTING/DEBUGGING: Override to run only 2 trials
    test_trials = np.array([4, 5])

    #vect_maps uses 30 maps that worked in functional run without collisions.
    #These maps are used for all GA experiments. The final dataframe contains 50 results
    #Index 1-30 has failure injections; index 31-50 are copies of functional results with no failures injected.

    for exp_key in EXPERIMENT_CONFIGS.keys():
        # run_ga_experiment(exp_key, vect_trials=test_trials)  # TESTING/DEBUGGING
        run_ga_experiment(exp_key)


if __name__ == "__main__":
    main()

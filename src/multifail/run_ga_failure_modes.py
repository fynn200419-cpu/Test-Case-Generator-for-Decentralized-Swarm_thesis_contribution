# Streamlined genetic algorithm runner for failure mode search
import pickle
import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.termination import TerminateIfAny
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination import get_termination
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from src.utils import create_results_dir
from src.multifail.failure_modes_problem import FailureModesProblem
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR, SEED_RANGE


def run_ga_failure_modes(vect_agents,
                         vect_trials,
                         failure_option=1,
                         num_failures=1,
                         functional_data_path=None,
                         run_type="GA_failure_modes",
                         population_size=20,
                         range_aware=float('inf'),
                         end_time_mult=2.0,
                         f_min_points=0.5,
                         v_ref=6.0,
                         env_config_type = 0,
                         terminate_config=None,
                         flag_parallel=False,
                         flag_eng_persist=True):
    """
    Run genetic algorithm search for failure scenarios in UAV swarms.
    
    Parameters:
        vect_agents (np.array): Array of agent counts to test
        vect_trials (np.array): Array of trial numbers (maps) to test
        failure_option (int/float): 1=power reduction, 1.1=power cutoff, 2=velocity multi-agent, 2.1=velocity single-agent multi-event
        num_failures (int): Number of failures per simulation
        functional_data_path (str): Path to functional test data CSV
        run_type (str): Run identifier for results directory
        population_size (int): GA population size
        num_generations (int): Number of GA generations
        v_ref (float): Reference velocity for velocity failure bounds
        flag_parallel (bool): Whether to use parallel simulation
        flag_eng_persist (bool): Whether to keep MATLAB engine persistent
        
    Returns:
        Path: Directory where results were saved
    """

    # Some hard-coded parameters
    start_x = 0.0
    end_x = 100.0

    # Default termination parameters
    if terminate_config is None:
        terminate_config = {
            'num_gen': 10,
            'max_time': None,
            'gen_stagnation': None,
            'gen_stagnation_tol' : 1e-6,
            'min_f': None
        }

    # Create results directory
    folder_path = create_results_dir(run_type)
    
    # Process each combination of agent count and trial
    all_results = []

    # Startup simulator if persistent engine is requested
    simulator = None
    if flag_eng_persist:
        simulator = SimulatorSwarmlab(str(SIM_DIR))
        if flag_parallel:
            simulator.start_parallel_pool()
    
    for num_agents in vect_agents:
        for trial_num in vect_trials:
            print(f"Running GA search: {num_agents} agents, trial {trial_num}")
            
            # Configure simulation parameters
            sim_config = {
                'end_time': np.nan,  # should be set in problem object
                'start_x': start_x,
                'end_x': end_x,
                'env_config_type': env_config_type,  # Default environment
                'env_maps': trial_num,
                'env_starts': trial_num
            }
            
            swarm_config = {
                'num_agents': int(num_agents),
                'd_ref': 10.0,
                'v_ref': v_ref,
                'r': range_aware
            }
            
            flags = {
                'output_all_states': False,
                'save': False,
                'viz': False,
                'viz_video': False,
                'dynamic_end': True,
                'run_parallel': flag_parallel,
                'cooperating': True,
                'ACTIVE_ARENA_WALLS': True,
                'eng_persist': flag_eng_persist
            }
            
            # Initialize simulator if not using persistent engine
            if not flag_eng_persist:
                simulator = SimulatorSwarmlab(str(SIM_DIR))
                if flag_parallel:
                    simulator.start_parallel_pool()
            
            # Create problem instance
            problem = FailureModesProblem(
                failure_option=failure_option,
                num_failures=num_failures,
                num_agents=num_agents,
                trial_num=trial_num,
                functional_data_path=functional_data_path,
                end_time_mult=end_time_mult,
                f_min_points=f_min_points,
                v_ref=v_ref,
                sim_config=sim_config,
                swarm_config=swarm_config,
                flags=flags,
                simulator=simulator
            )
            
            # Configure GA algorithm
            algorithm = GA(
                pop_size=population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1.0/problem.n_var, eta=20)
            )
            
            # Define termination
            termination_criteria = []
            # Time or max number of generations
            if terminate_config['max_time'] is not None:
                termination_criteria.append(get_termination("time", terminate_config['max_time']))
            else:
                termination_criteria.append(get_termination("n_gen", terminate_config['num_gen']))
            # Minimum fitness
            if terminate_config['min_f'] is not None:
                termination_criteria.append(get_termination("fmin", terminate_config['min_f']))
            # Stagnation
            if terminate_config['gen_stagnation'] is not None:
                termination_criteria.append(SingleObjectiveSpaceTermination(tol=terminate_config['gen_stagnation_tol'],
                                                                            n_last=terminate_config['gen_stagnation']))
            # Assemble
            if len(termination_criteria) > 1:
                termination = TerminateIfAny(*termination_criteria)
            elif len(termination_criteria) == 1:
                termination = termination_criteria[0]
            else:
                raise ValueError(
                    f"Termination criteria not defined correctly for trial: {trial_num} with num agents: {num_agents}"
                )

            # Run optimization
            result = minimize(
                problem,
                algorithm,
                termination,
                seed=int(SEED_RANGE[trial_num - 1]),
                verbose=True,
                save_history=True
            )

            # Remove simulator from results object to enable pickling later
            result.problem.simulator = None
            
            # Process results
            print(f"GA completed. Best fitness: {result.F[0]:.3f}")
            
            # Extract best solution
            best_x = result.X[0] if result.X.ndim > 1 else result.X
            best_fitness = result.F[0] if result.F.ndim > 1 else result.F
            
            # Decode best solution
            agent_ids, fail_times, fail_powers, fail_v_n, fail_v_e, fail_v_d = problem.decode_variables(
                best_x.reshape(1, -1))
            
            # Store result data
            result_data = {
                'num_agents': num_agents,
                'trial': trial_num,
                'start_time': problem.flight_start,
                'end_time' : round(problem.sim_end_time, 2),
                'end_time_mult': end_time_mult,
                'seed_num': int(SEED_RANGE[trial_num - 1]),
                'failure_option': failure_option,
                'num_failures': num_failures,
                'population_size': population_size,
                'num_generations': result.history[-1].n_gen,
                'best_fitness': best_fitness,
                'best_agent_ids': agent_ids,
                'best_fail_times': fail_times,
                'convergence_history': [gen.opt[0].F[0] for gen in result.history],
                'problem_bounds': {
                    'xl': problem.xl,
                    'xu': problem.xu
                },
                'result': result
            }
            
            if failure_option in [1, 1.1]:  # Power failures
                result_data['best_fail_powers'] = fail_powers
            elif failure_option in [2, 2.1]:  # Velocity failures (both multi-agent and single-agent multi-event)
                result_data['best_fail_v_n'] = fail_v_n
                result_data['best_fail_v_e'] = fail_v_e
                result_data['best_fail_v_d'] = fail_v_d
            
            all_results.append(result_data)
            
            # Save individual result
            result_filename = f"ga_result_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"
            result_path = folder_path / result_filename
            with open(result_path, 'wb') as f:
                pickle.dump(result_data, f)
            
            # Shutdown simulator for this run if not persistent
            if not flag_eng_persist and simulator:
                simulator.shutdown_engine()
    
    # Create summary dataframe
    df_rows = []
    for result_data in all_results:
        # Create a row for the best solution found
        row = {
            'trial': result_data['trial'],
            'seed_num': result_data['seed_num'],
            'pop_size': result_data['population_size'],
            'num_gen': result_data['num_generations'],
            'end_time': result_data['end_time'],
            'start_x': start_x,
            'end_x': end_x,
            'num_agents': result_data['num_agents'],
            'fail_in_swarm': True,
            'fitness': result_data['best_fitness'],
            'failure_option': result_data['failure_option'],
            'num_failures': result_data['num_failures']
        }
        
        # Add multi-failure columns dynamically based on num_failures
        num_failures = result_data['num_failures']
        agent_ids = result_data['best_agent_ids'][0]  # Extract from (1, num_failures) array
        fail_times = result_data['best_fail_times'][0]  # Extract from (1, num_failures) array
        
        # Flatten agent IDs and failure times into individual columns
        for i in range(num_failures):
            row[f'fail_agent_num_{i+1}'] = agent_ids[i] if i < len(agent_ids) else np.nan
            row[f'fail_time_{i+1}'] = fail_times[i] if i < len(fail_times) else np.nan
        
        # Add failure-specific parameters with individual columns
        if result_data['failure_option'] in [1, 1.1]:  # Power failures
            fail_powers = result_data['best_fail_powers'][0]  # Extract from (1, num_failures) array
            for i in range(num_failures):
                row[f'fail_power_{i+1}'] = fail_powers[i] if i < len(fail_powers) else np.nan
                
        elif result_data['failure_option'] in [2, 2.1]:  # Velocity failures
            fail_v_n = result_data['best_fail_v_n'][0]  # Extract from (1, num_failures) array
            fail_v_e = result_data['best_fail_v_e'][0]  # Extract from (1, num_failures) array
            fail_v_d = result_data['best_fail_v_d'][0]  # Extract from (1, num_failures) array
            for i in range(num_failures):
                row[f'fail_v_n_{i+1}'] = fail_v_n[i] if i < len(fail_v_n) else np.nan
                row[f'fail_v_e_{i+1}'] = fail_v_e[i] if i < len(fail_v_e) else np.nan
                row[f'fail_v_d_{i+1}'] = fail_v_d[i] if i < len(fail_v_d) else np.nan
        
        df_rows.append(row)
    
    # Save summary dataframe
    df = pd.DataFrame(df_rows)
    pickle_path = folder_path / "dataframe.pkl"
    df.to_pickle(pickle_path)
    csv_path = folder_path / "dataframe.csv"
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    # Save all results summary
    summary_path = folder_path / "all_results.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Cleanup persistent simulator
    if flag_eng_persist :
        simulator.shutdown_engine()
    
    print(f"GA search completed. Results saved to: {folder_path}")
    return folder_path


def load_ga_results(results_path):
    """
    Load and analyze GA results from a results directory.
    
    Parameters:
        results_path (str or Path): Path to results directory
        
    Returns:
        dict: Loaded results data
    """
    import pathlib
    results_path = pathlib.Path(results_path)
    
    # Load summary data
    summary_path = results_path / "all_results.pkl" 
    with open(summary_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Load dataframe
    df_path = results_path / "dataframe.pkl"
    df = pd.read_pickle(df_path)
    
    return {
        'all_results': all_results,
        'dataframe': df,
        'results_directory': results_path
    }
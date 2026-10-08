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
                         flag_eng_persist=True,
                         seeding_method='coordination',  # 'id' for seeding.py, 'coordination' for coordination_seeding.py, or 'none' for random
                         functional_trajectory_dir=None, #passed functional path from main gets passed
                         baseline_results_path=None):  # path to existing results folder for first agent size; skips re-running it
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
        seeding_method (str): Seeding strategy - 'id' for ID-based (default), 'coordination' for spatial coordination, 'none' for no seeding
        functional_trajectory_dir (str or Path): Path to functional baseline results directory containing 'trajectory_data/' subdirectory.
                                    Required for coordination seeding. The trajectory_data/ folder should have iter001/, iter002/, etc.
                                    with state_var.mat files from functional runs (no failures).

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
    
    vect_agents = list(vect_agents) #converting to python list for easier indexing

    for idx, num_agents in enumerate(vect_agents): # loops through the indices and has the num_agents variable for current swarm size
        for trial_num in vect_trials:
            print(f"Running GA search: {num_agents} agents, trial {trial_num}")

    # for num_agents in vect_agents:
    #     for trial_num in vect_trials:
    #         print(f"Running GA search: {num_agents} agents, trial {trial_num}")
            
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

            # Determine sampling strategy
            from seeding.seeding import build_seeded_sampler
            from pathlib import Path

            # --- Baseline loading: skip GA for first agent size if pre-existing results provided ---
            if idx == 0 and baseline_results_path is not None:
                import shutil
                baseline_path = Path(baseline_results_path)
                src_pickle = baseline_path / f"seeds_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"
                dst_pickle = folder_path / f"seeds_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"
                src_result  = baseline_path / f"ga_result_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"

                if src_pickle.exists() and src_result.exists():
                    shutil.copy2(src_pickle, dst_pickle)
                    with open(src_result, 'rb') as f:
                        baseline_result_data = pickle.load(f)
                    all_results.append(baseline_result_data)
                    print(f"[BASELINE] Loaded trial {trial_num} ({num_agents} agents) from {baseline_results_path}")
                    continue  # skip GA for this trial
                else:
                    print(f"[BASELINE] WARNING: Baseline files not found for trial {trial_num} "
                          f"({num_agents} agents) — running GA with random sampling instead")
                    print(f"[BASELINE]   Expected seed pickle : {src_pickle}")
                    print(f"[BASELINE]   Expected result file : {src_result}")

            # Seeding logic:
            # - First agent count (idx==0): random sampling (or skipped above via baseline)
            # - Subsequent agent counts: use seeding from previous agent count based on seeding_method

            if idx == 0:
                # First agent count always uses random sampling (no previous data to seed from)
                sampling_strategy = FloatRandomSampling()
                print(f"[SEEDING] Using random sampling (first agent count in list: {num_agents})")
            elif seeding_method == 'none':
                # Force no seeding even for subsequent runs
                sampling_strategy = FloatRandomSampling()
                print(f"[SEEDING] Using random sampling (seeding_method='none')")
            elif seeding_method == 'constraint':
                # Constraint seeding: tighten time bounds based on obstacle clearance,
                # then use coordination seeding so seeds are filtered to the tightened window.
                from seeding.constraintseeding import apply_obstacle_time_constraint
                from seeding.coordination_seeding import build_coordination_seeded_sampler

                prev_agent_count = vect_agents[idx - 1]
                seed_path = folder_path / f"seeds_{prev_agent_count}agents_trial{trial_num}_failopt{failure_option}.pkl"

                if functional_trajectory_dir is not None:
                    functional_traj_dir = Path(functional_trajectory_dir)
                else:
                    functional_traj_dir = Path(functional_data_path).parent
                traj_path = functional_traj_dir / "trajectory_data" / f"{num_agents}agents" / f"iter{trial_num:03d}" / "state_var.mat"

                # Step 1: tighten problem.xu in-place (no return value)
                if not traj_path.exists():
                    print(f"[CONSTRAINT] WARNING: Functional trajectory not found: {traj_path}")
                    print(f"[CONSTRAINT] Skipping constraint, bounds remain unchanged")
                else:
                    apply_obstacle_time_constraint(problem, env_config_type, trial_num, traj_path)

                # Step 2: coordination seeding — seeds filtered against the (now tightened) bounds
                if idx == 1 and baseline_results_path is not None:
                    old_traj_source = Path(baseline_results_path)
                else:
                    old_traj_source = folder_path
                old_traj_dir = old_traj_source / "top10_trajectories" / f"{prev_agent_count}agents_trial{trial_num}"
                new_traj_path = traj_path

                print(f"[SEEDING] Using constraint seeding: {prev_agent_count} -> {num_agents} agents")

                if not new_traj_path.exists():
                    print(f"[ERROR] Functional trajectory not found: {new_traj_path}, falling back to random sampling")
                    sampling_strategy = FloatRandomSampling()
                else:
                    sampling_strategy = build_coordination_seeded_sampler(
                        seed_pickle_path=seed_path,
                        old_traj_dir=old_traj_dir,
                        new_traj_path=new_traj_path,
                        problem=problem,
                        failure_option=failure_option,
                        trial_idx=0,
                        n_seed=10,
                        n_pool=20
                    )
            else:
                # Use seeding from previous agent count
                prev_agent_count = vect_agents[idx - 1]
                seed_path = folder_path / f"seeds_{prev_agent_count}agents_trial{trial_num}_failopt{failure_option}.pkl"

                # Apply seeding based on selected method
                if seeding_method == 'coordination':
                    # Coordination seeding: uses spatial locations to match agents across swarm sizes
                    # Auto-constructs paths:
                    #   - old_traj_dir: folder_path/top10_trajectories/{prev_agents}agents_trial{N}/ (created after prev GA run)
                    #   - new_traj_path: derived from functional_data_path parent directory + trajectory_data/iter{N}/state_var.mat
                    from seeding.coordination_seeding import build_coordination_seeded_sampler

                    # Derive functional trajectory directory from functional_data_path
                    # functional_data_path points to dataframe.csv, so parent is the results folder
                    # which contains trajectory_data/ subdirectory
                    if functional_data_path is None:
                        print("[WARNING] Coordination seeding requested but functional_data_path not provided!")
                        print("[WARNING] Falling back to ID-based seeding")
                        sampling_strategy = build_seeded_sampler(seed_path, problem, n_seed=10)
                    else:
                        # Auto-construct paths
                        # Old trajectories: use baseline folder when idx==1 and baseline was provided,
                        # otherwise use current run's folder (populated by rerun_top10 after prev swarm size)
                        if idx == 1 and baseline_results_path is not None:
                            old_traj_source = Path(baseline_results_path)
                        else:
                            old_traj_source = folder_path
                        old_traj_dir = old_traj_source / "top10_trajectories" / f"{prev_agent_count}agents_trial{trial_num}"

                        # New trajectories: derive from functional_data_path (parent folder contains trajectory_data/)
                        # If functional_trajectory_dir is explicitly provided, use it; otherwise derive from functional_data_path
                        if functional_trajectory_dir is not None:
                            functional_traj_dir = Path(functional_trajectory_dir)
                        else:
                            functional_traj_dir = Path(functional_data_path).parent
                        # Structure: trajectory_data/{num_agents}agents/iter{trial}/state_var.mat
                        new_traj_path = functional_traj_dir / "trajectory_data" / f"{num_agents}agents" / f"iter{trial_num:03d}" / "state_var.mat" #correctly accessing the functional trajectory data that was store be

                        print(f"[SEEDING] Using coordination seeding: {prev_agent_count} -> {num_agents} agents")
                        print(f"[SEEDING]   Seed pickle (will exist after prev run): {seed_path}")
                        print(f"[SEEDING]   Old trajectory dir (will exist after prev run): {old_traj_dir}")
                        print(f"[SEEDING]   New trajectory (functional baseline): {new_traj_path}")

                        # Check if new trajectory exists (this should exist from functional baseline)
                        if not new_traj_path.exists():
                            print(f"[ERROR] Functional trajectory file not found: {new_traj_path}")
                            print(f"[ERROR] Make sure you ran functional baseline with trajectory saving enabled")
                            print(f"[ERROR] Falling back to ID-based seeding")
                            sampling_strategy = build_seeded_sampler(seed_path, problem, n_seed=10)
                        else:
                            # Let build_coordination_seeded_sampler handle missing old trajectories gracefully
                            # (it will fall back to random sampling if files don't exist yet)
                            sampling_strategy = build_coordination_seeded_sampler(
                                seed_pickle_path=seed_path,
                                old_traj_dir=old_traj_dir,
                                new_traj_path=new_traj_path,
                                problem=problem,
                                failure_option=failure_option,
                                trial_idx=0,
                                n_seed=10,
                                n_pool=20
                            )
                elif seeding_method == 'id':
                    # ID-based seeding: traditional approach (only works for same swarm size)
                    print(f"[SEEDING] Using ID-based seeding from {prev_agent_count} agents")
                    sampling_strategy = build_seeded_sampler(seed_path, problem, n_seed=10)
                else:
                    raise ValueError(f"Unknown seeding_method: '{seeding_method}'. Must be 'id', 'coordination', 'constraint', or 'none'")



            # #NEW CODE TO CALL SEEDING IF NECCESSARY
            # # Determine sampling strategy
            # from seeding import build_seeded_sampler
            # if num_agents == vect_agents[0]:
            #     sampling_strategy = FloatRandomSampling()
            # else:
            #     # Look for seed file from previous agent count
            #     prev_agent_count = vect_agents[vect_agents.tolist().index(num_agents) - 1]
            #     #vect_agents.tolist().index(num_agents) finds position of current swarmsize in the list. So -1 gets the index of previous size.
            #     # since it is in vect_agents[] we get the agent count of that index and can filter in the next line for the correct seeding file.
            #     seed_path = folder_path / f"seeds_{prev_agent_count}agents_trial{trial_num}_failopt{failure_option}.pkl"
            #     sampling_strategy = build_seeded_sampler(seed_path, problem, n_seed=10)
            
            # Configure GA algorithm
            algorithm = GA(
                pop_size=population_size,
                sampling=sampling_strategy,
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

            #NEWCODE TO SAVE SEEDS AFTER GA RUN
            seed_data = {
                'X': np.vstack(problem.all_solutions_X),
                'F': np.concatenate(problem.all_solutions_F)
            }
            #stores all evaluated solutions and their fitness values during the GA run.
            seed_pickle_path = folder_path / f"seeds_{num_agents}agents_trial{trial_num}_failopt{failure_option}.pkl"
            #creates new seeds file which only contains X and F arrays.
            with open(seed_pickle_path, 'wb') as f:
                pickle.dump(seed_data, f)

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

        # After completing all trials for this swarm size, re-run top-10 solutions with trajectory saving
        # This is needed for coordination seeding to work for the next swarm size.
        # Skip if this was the baseline swarm size — trajectories already exist in baseline_results_path.
        if idx < len(vect_agents) - 1 and seeding_method in ('coordination', 'constraint'):
            if idx == 0 and baseline_results_path is not None:
                print(f"\n{'='*60}")
                print(f"[BASELINE] Skipping top-10 re-run for {num_agents} agents.")
                print(f"[BASELINE] Trajectories will be read directly from baseline folder:")
                print(f"[BASELINE]   {Path(baseline_results_path) / 'top10_trajectories'}")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"Re-running top-10 solutions for {num_agents} agents with trajectory saving...")
                print(f"This generates trajectory data needed for coordination seeding")
                print(f"{'='*60}")

                from src.multifail.rerun_top10_with_trajectories import rerun_top10_solutions_with_trajectories

                try:
                    trajectory_paths = rerun_top10_solutions_with_trajectories(
                        ga_results_dir=folder_path,
                        num_agents_list=[num_agents],
                        trial_list=list(vect_trials),
                        failure_option=failure_option,
                        n_seed=20,
                        output_subdir='top10_trajectories'
                    )
                    print(f"\n{'='*60}")
                    print(f"✓ Generated {len(trajectory_paths)} trajectory sets for coordination seeding")
                    print(f"  Next swarm size ({vect_agents[idx + 1]} agents) will use these trajectories")
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"\n{'='*60}")
                    print(f"[WARNING] Failed to generate top-10 trajectories: {e}")
                    print(f"[WARNING] Coordination seeding for next swarm size may fall back to ID-based seeding")
                    print(f"{'='*60}\n")

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
    
    # Load dataframe
    df_path = results_path / "dataframe.pkl"
    df = pd.read_pickle(df_path)
    
    return {
        'dataframe': df,
        'results_directory': results_path
    }
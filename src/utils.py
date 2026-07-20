# Utility functions

import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from src.constants import PROJECT_ROOT, RESULTS_DIR


@contextmanager
def tic_toc():
    """A reusable context manager for timing code execution."""
    start = time.perf_counter()
    yield  # This is where the code inside the `with` block runs
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")


def make_dataframe(trial, seed_num, pop_size, num_gen,
                   x, coll_result, flight_time, flight_start, flight_end,
                   existing_df=None):
    # Iterate through dictionary and get data in specific fields
    coll_result_occur = []
    coll_result_type = []
    for i in coll_result:
        coll_result_occur.append(i['occur'])  # Should be bool
        coll_result_type.append(str(i['type']))  # Should be string

    # Process input data for dataframe
    x_0 = x[:, 0]
    x_1 = x[:, 1]
    x_2 = x[:, 2]
    x_3 = x[:, 3]
    x_4 = x[:, 4]
    x_5 = x[:, 5]
    x_6 = x[:, 6]

    # Put into dictionary
    data_dict = {
        "trial": trial,
        "seed_num": seed_num,
        "pop_size": pop_size,
        "num_gen": num_gen,
        "end_time": x_0,
        "start_x": x_1,
        "end_x": x_2,
        "num_agents": x_3.astype(int),
        "fail_agent_num": np.nan_to_num(x_4, nan=0).astype(int),
        "fail_time": x_5,
        "fail_in_swarm": np.nan_to_num(x_6, nan=False).astype(bool),
        "coll_result_occur": coll_result_occur,
        "coll_result_type": coll_result_type,
        "flight_time": flight_time,
        "flight_start": flight_start,
        "flight_end": flight_end
    }

    # Create new dataframe
    new_df = pd.DataFrame(data_dict)

    # Specify data types for each column
    new_df = new_df.astype({
        "trial": int,
        "seed_num": float,
        "pop_size": int,
        "num_gen": int,
        "end_time": float,
        "start_x": float,
        "end_x": float,
        "num_agents": int,
        "fail_agent_num": int,
        "fail_time": float,
        "fail_in_swarm": bool,
        "coll_result_occur": bool,
        "coll_result_type": str,
        "flight_time": float,
        "flight_start": float,
        "flight_end": float
    })

    # If there is an existing dataframe, then concatenate the new data
    if isinstance(existing_df, pd.DataFrame):
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    return new_df


def make_multifail_dataframe(trial, seed_num, pop_size, num_gen, x,
                             coll_result_occur, coll_result_type, flight_time, flight_start, flight_end,
                             existing_df=None):

    # Process sim input data for dataframe
    x_0 = x[:, 0]
    x_1 = x[:, 1]
    x_2 = x[:, 2]
    x_3 = x[:, 3]
    x_4 = x[:, 4]
    x_5 = x[:, 5]
    x_6 = x[:, 6]
    x_7 = x[:, 7]
    x_8 = x[:, 8]

    # Put into dictionary
    data_dict = {
        "trial": trial,
        "solution_num": x_8,
        "seed_num": seed_num,
        "pop_size": pop_size,
        "num_gen": num_gen,
        "end_time": x_0,
        "start_x": x_1,
        "end_x": x_2,
        "num_agents": x_3.astype(int),
        "fail_agent_num": np.nan_to_num(x_4, nan=0).astype(int),
        "fail_time": x_5,
        "fail_power": x_6,
        "fail_in_swarm": np.nan_to_num(x_7, nan=False).astype(bool),
        "coll_result_occur": coll_result_occur,
        "coll_result_type": coll_result_type,
        "flight_time": flight_time,
        "flight_start": flight_start,
        "flight_end": flight_end
    }

    # Create new dataframe
    new_df = pd.DataFrame(data_dict)

    # Specify data types for each column
    new_df = new_df.astype({
        "trial": int,
        "solution_num": int,
        "seed_num": float,
        "pop_size": int,
        "num_gen": int,
        "end_time": float,
        "start_x": float,
        "end_x": float,
        "num_agents": int,
        "fail_agent_num": int,
        "fail_time": float,
        "fail_power": float,
        "fail_in_swarm": bool,
        "coll_result_occur": bool,
        "coll_result_type": str,
        "flight_time": float,
        "flight_start": float,
        "flight_end": float
    })

    # If there is an existing dataframe, then concatenate the new data
    if isinstance(existing_df, pd.DataFrame):
        new_df = pd.concat([existing_df, new_df], ignore_index=True)

    return new_df


def create_results_dir(exp_type, results_dir=RESULTS_DIR, prefix="result"):
    """
    Create new directory for results and output the new folder's path as Path object.
    """

    # Convert results_dir to a Path object if it's not already
    results_dir = Path(results_dir)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the folder name and create the folder
    folder_name = f"{prefix}_{timestamp}_{exp_type}"
    folder_path = results_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=False)  # Create the folder

    return folder_path


def load_all_random_into_df(results_to_save_dir, results_dirs, num_agents, vect_trials, seed_range, time_step, result_vars):

    # Initialize variable for dataframe (not good practice to initialize empty dataframe)
    df = None

    # Iterate and load
    project_dir = Path(PROJECT_ROOT)
    for idx, results_dir in enumerate(results_dirs):  # Iterate through the directories

        temp_results_dir = project_dir / results_to_save_dir / results_dir

        for trial in vect_trials:  # Iterate through the trials (one file per trial)

            # Load results
            file_name = f"res_trial{trial}_seed{seed_range[trial - 1]}.pkl"
            with open(temp_results_dir / file_name, "rb") as file:
                res = pickle.load(file)

            # Extract variables from dictionary and add to dataframe
            temp_all_X = res["all_X"]

            # fail_agent_num
            temp_fail_agent_num = temp_all_X[:, 0].flatten()

            # fail_time
            temp_fail_time = temp_all_X[:, 1].flatten() * time_step

            # flight_time
            temp_all_flight_time = res["all_flight_time"]

            # coll_result_occur and coll_result_type
            temp_all_coll_result = res["all_coll_result"]
            temp_coll_result_occur = []
            temp_coll_result_type = []
            for i in temp_all_coll_result:
                temp_coll_result_occur.append(i['occur'])  # Should be bool
                temp_coll_result_type.append(str(i['type']))  # Should be string

            # num_agents
            temp_num_agents = np.full(len(temp_fail_agent_num), num_agents[idx])

            # Assemble into a dataframe
            temp_df = pd.DataFrame({
                result_vars[0]: temp_num_agents,
                result_vars[1]: trial,
                result_vars[2]: temp_fail_agent_num,
                result_vars[3]: temp_fail_time,
                result_vars[4]: temp_all_flight_time,
                result_vars[5]: temp_coll_result_occur,
                result_vars[6]: temp_coll_result_type,
            })

            # Concatenate to main dataframe
            if isinstance(df, pd.DataFrame):
                df = pd.concat([df, temp_df], ignore_index=True)
            else:
                df = temp_df

    # Add a column to specify search method
    df["search_method"] = "random_all"

    return df


def load_final_into_dataframe(results_to_save_dir, random_results_dirs, genetic_results_dirs):
    """
    Load csv files into two dataframes
    """

    # Load random search results
    random_dfs = []
    for dir in random_results_dirs:
        random_dfs.append(pd.read_csv(PROJECT_ROOT / results_to_save_dir / dir / "dataframe.csv"))
    random_df = pd.concat(random_dfs)

    # Load genetic algorithm results
    genetic_dfs = []
    for dir in genetic_results_dirs:
        genetic_dfs.append(pd.read_csv(PROJECT_ROOT / results_to_save_dir / dir / "dataframe.csv"))
    genetic_df = pd.concat(genetic_dfs)

    # Add a column to specify search method
    random_df["search_method"] = "random"
    genetic_df["search_method"] = "genetic"

    return random_df, genetic_df


def load_all_search_into_df(results_to_save_dir, results_dirs, num_agents, vect_trials, seed_range, time_step, result_vars):
    # Initialize variable for dataframe (not good practice to initialize empty dataframe)
    df = None

    # Iterate and load
    project_dir = Path(PROJECT_ROOT)
    for idx, results_dir in enumerate(results_dirs):  # Iterate through the directories

        temp_results_dir = project_dir / results_to_save_dir / results_dir

        for trial in vect_trials:  # Iterate through the trials (one file per trial)

            # Load results
            file_name = f"res_trial{trial}_seed{seed_range[trial - 1]}.pkl"
            with open(temp_results_dir / file_name, "rb") as file:
                res = pickle.load(file)

            # Extract variables from dictionary and add to dataframe
            # Iterate through each generation:
            for cur_gen in res.history:

                # Get all X values
                # cur_X = np.array([e.X for e in cur_gen.pop])  # This gets values of X for generation 1
                cur_X = cur_gen.pop.get("X")

                # fail_agent_num
                temp_fail_agent_num = cur_X[:, 0].flatten()

                # fail_time
                temp_fail_time = cur_X[:, 1].flatten() * time_step

                # Get all flight times
                temp_flight_time = cur_gen.pop.get("flight_time")

                # num_agents
                temp_num_agents = np.full(len(temp_fail_agent_num), num_agents[idx])

                # trial

                # collisions results
                temp_coll_result_occur = cur_gen.pop.get("coll_result_occur")
                temp_coll_result_occur= temp_coll_result_occur.astype(bool)  # convert to bool
                temp_coll_result_type = cur_gen.pop.get("coll_result_type")

                # Assemble into a dataframe
                temp_df = pd.DataFrame({
                    result_vars[0]: num_agents[idx],
                    result_vars[1]: trial,
                    "current_gen": cur_gen.n_gen,
                    result_vars[2]: temp_fail_agent_num,
                    result_vars[3]: temp_fail_time,
                    result_vars[4]: temp_flight_time,
                    result_vars[5]: temp_coll_result_occur,
                    result_vars[6]: temp_coll_result_type,
                })

                # Concatenate to main dataframe
                if isinstance(df, pd.DataFrame):
                    df = pd.concat([df, temp_df], ignore_index=True)
                else:
                    df = temp_df

    df["search_method"] = "genetic_all"

    return df


def load_all_multifail_random_into_df(results_to_save_dir, results_dirs, num_agents, vect_trials, seed_range, time_step, result_vars):

    # Initialize variable for dataframe (not good practice to initialize empty dataframe)
    df = None

    # Iterate and load
    project_dir = Path(PROJECT_ROOT)
    for idx, results_dir in enumerate(results_dirs):  # Iterate through the directories

        # # For status messages
        # print(f"Loading results from individual results directory: {results_to_save_dir} \ {results_dir}")
        # num_inf_in_experiment = 0

        temp_results_dir = project_dir / results_to_save_dir / results_dir

        for trial in vect_trials:  # Iterate through the trials (one file per trial)

            # Load results
            file_name = f"res_trial{trial}_seed{seed_range[trial - 1]}.pkl"
            with open(temp_results_dir / file_name, "rb") as file:
                res = pickle.load(file)

            # Extract variables from dictionary and add to dataframe
            temp_all_X = res["all_X"]

            # Determine number of individuals based on columns in temp_all_X
            # For each individual: one column for fail_agent_num, one for fail_time
            num_individuals = temp_all_X.shape[1] // 2

            # Extract data for each run
            temp_all_flight_time = res["all_flight_time"]  # flight time
            temp_all_coll_result = res["all_coll_result"]  # coll_result_occur and coll_result_type
            temp_coll_result_occur = []
            temp_coll_result_type = []
            for i in temp_all_coll_result:
                temp_coll_result_occur.append(i['occur'])  # Should be bool
                temp_coll_result_type.append(str(i['type']))  # Should be string

            # Process each run (row in temp_all_X)
            temp_df_list = []
            for run_idx in range(temp_all_X.shape[0]):
                # Create a signature for this run based on selected individuals and times
                # Extract all individual-time pairs for this run
                # Create a set of (individual_id, time) pairs for this run
                individual_time_pairs = []
                for ind_idx in range(num_individuals):
                    individual_id = temp_all_X[run_idx, ind_idx]
                    individual_time = temp_all_X[run_idx, ind_idx + num_individuals] * time_step
                    individual_time_pairs.append((individual_id, individual_time))

                # Convert to frozenset (an immutable set that can be hashed)
                individual_time_pairs = frozenset(individual_time_pairs)

                # Create a hash of the frozenset as the signature
                run_signature = hash(individual_time_pairs)

                # Iterate through each individual-time pair for this run and add rows to the dataframe
                for ind_idx in range(num_individuals):
                    # Get individual num and time
                    individual_num = temp_all_X[run_idx, ind_idx]
                    individual_time = temp_all_X[run_idx, ind_idx + num_individuals] * time_step

                    # Create dataframe row
                    temp_row = {
                        result_vars[0]: num_agents[idx],
                        result_vars[1]: trial,
                        'run_num': run_idx + 1,
                        'num_fail': num_individuals,
                        result_vars[2]: individual_num,
                        result_vars[3]: individual_time,
                        result_vars[4]: temp_all_flight_time[run_idx],
                        result_vars[5]: temp_coll_result_occur[run_idx],
                        result_vars[6]: temp_coll_result_type[run_idx],
                        'run_signature': run_signature,
                        'unique_run_id': f"{num_agents[idx]}_{trial}_{run_idx + 1}_{num_individuals}"
                    }
                    temp_df_list.append(temp_row)

            # Convert to DataFrame
            temp_df = pd.DataFrame(temp_df_list)

            # Concatenate to main dataframe
            if isinstance(df, pd.DataFrame):
                df = pd.concat([df, temp_df], ignore_index=True)
            else:
                df = temp_df

            # # For status message
            # num_inf_in_experiment += np.isinf(temp_df['flight_time']).sum()

        # # Status message: number with infinite flight time
        # # For status messages
        # print(f"\t{num_inf_in_experiment} runs have infinite flight time for this experiment.")

    # Add a column to specify search method
    df["search_method"] = "random_all"

    return df
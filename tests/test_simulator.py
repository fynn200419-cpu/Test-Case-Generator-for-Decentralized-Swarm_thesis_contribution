# Test simulator by running some example iterations in a consecutive loop
# and in a parallel loop, then comparing outputs.

import pytest
import numpy as np
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.utils import tic_toc
from src.constants import SIM_DIR


def run_functional_test(matlab_dir, flags):
    """
    Run a functional test of the swarm simulator.

    This function initializes the simulator and runs a series of simulation tests
    with various input configurations to evaluate performance and functionality.

    Parameters:
        matlab_dir (str): The directory path to the MATLAB-based simulation environment.
        flags (dict): A dictionary of configuration options for the simulation.
                      Possible keys include:
                      - 'run_parallel' (bool): Whether to run the simulation in parallel.
                      - 'viz' (bool): Whether to enable visualization during the simulation.
                      - 'save' (bool): Whether to save the simulation results.

    Returns:
        tuple: Contains the following outputs from the simulation:
            - coll_result: Results of collision checks.
            - coll_obs: Collisions with obstacles.
            - coll_drones: Collisions between drones.
            - coll_walls: Collisions with walls.
            - flight_time: Total flight time for agents.
            - flight_start: Flight start time.
            - flight_end: Flight end time.
    """

    # Initialize the simulator object
    # matlab_dir = r"C:\Users\dmarson\Documents\Projects\Marson_PhD\swarmlab-augmented\projects\robustness-targeted-failures"
    sim = SimulatorSwarmlab(matlab_dir)

    # Flags (dictionary -> struct)
    # flags = {'run_parallel': False, 'viz': False, 'save': False}
    # flags = {'run_parallel': True, 'viz': False, 'save': False}

    # End time
    end_time = np.full((7,), 30)

    # Start and end planes
    start_x = np.full((7,), 0)
    end_x = np.full((7,), 100)

    # Number of agents
    # num_agents = np.array([10, 10, 10, 50, 50, 50, 100])
    num_agents = np.array([10, 10, 10, 20, 20, 20, 30])  # shorter test

    # Failure conditions
    fail_agent_num = np.array([np.nan, 1, 1, np.nan, 1, 1, np.nan])
    fail_time = np.array([np.nan, 15, 15, np.nan, 15, 15, np.nan])
    fail_in_swarm = np.array([np.nan, True, False, np.nan, True, False, np.nan])

    # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
    x = np.column_stack((end_time, start_x, end_x, num_agents, fail_agent_num, fail_time, fail_in_swarm))

    # Run the simulation
    coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end = sim.simulate(x, flags)

    return coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end


def run_multifail_test_single_agent(matlab_dir, flags):
    # Initialize the simulator object
    sim = SimulatorSwarmlab(matlab_dir)

    # End time
    end_time = np.full((7,), 30)

    # Start and end planes
    start_x = np.full((7,), 0)
    end_x = np.full((7,), 100)

    # Number of agents
    # num_agents = np.array([10, 10, 10, 50, 50, 50, 100])
    num_agents = np.array([10, 10, 10, 20, 20, 20, 30])  # shorter test

    # Failure conditions
    fail_agent_num = np.array([np.nan, 1, 1, np.nan, 1, 1, np.nan])
    fail_time = np.array([np.nan, 15, 15, np.nan, 15, 15, np.nan])
    fail_in_swarm = np.array([np.nan, True, False, np.nan, True, False, np.nan])

    # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
    x = np.column_stack((end_time, start_x, end_x, num_agents, fail_in_swarm, fail_agent_num, fail_time))

    # Run the simulation
    coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end = (
        sim.multifail_simulate(x, flags))

    return coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end

def run_multifail_test_multi_agents(matlab_dir, flags):
    # Initialize the simulator object
    sim = SimulatorSwarmlab(matlab_dir)

    # End time
    end_time = np.full((3,), 30)

    # Start and end planes
    start_x = np.full((3,), 0)
    end_x = np.full((3,), 100)

    # Number of agents
    # num_agents = np.array([10, 10, 10, 50, 50, 50, 100])
    num_agents = np.array([10, 20, 30])  # shorter test

    # Failure conditions
    fail_in_swarm = np.array([True, False, True])
    fail_agent_num = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
    fail_time = np.array([[10.01, 11.02, 12.1],
                          [13.10, 14.31, 15.02],
                          [16.10, 17.30, 18.01]])

    # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
    x = np.column_stack((end_time, start_x, end_x, num_agents, fail_in_swarm, fail_agent_num, fail_time))

    # Run the simulation
    coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end = (
        sim.multifail_simulate(x, flags))

    return coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end


def test_simulator_system():
    """
    Test the swarm simulator system for consistency and performance.

    This function performs end-to-end testing of the simulator by running it
    under different configurations (sequential and parallel executions).
    It compares outputs to verify consistency across configurations.

    Procedure:
    1. Runs the simulation in a sequential mode.
    2. Runs the simulation in a parallel mode.
    3. Compares outputs (e.g., flight times, collisions) from both modes to ensure equivalence.

    Notes:
    - This function is intended to be executed using `pytest`. For example:
      `poetry run pytest -s` (to see print output) or `poetry run pytest -v` (for verbose output).

    Assertions:
    - Verifies flight time consistency across configurations.
    - Checks collision data consistency between drones and walls.

    Raises:
        AssertionError: If any of the output comparisons fail, indicating a discrepancy
        between sequential and parallel simulation runs.

    Returns:
        None
    """

    # Get simulator directory from constants
    matlab_dir = str(SIM_DIR)

    # Run simulation with consecutive iterations
    flags_1 = {'run_parallel': False, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # first try consecutive runs
    with tic_toc():
        (coll_result_1, coll_obs_1, coll_drones_1, coll_walls_1,
            flight_time_1, flight_start_1, flight_end_1) = run_functional_test(matlab_dir, flags_1)

    # Run simulation in parallel
    flags_2 = {'run_parallel': True, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # now try parallel runs
    with tic_toc():
        (coll_result_2, coll_obs_2, coll_drones_2, coll_walls_2,
            flight_time_2, flight_start_2, flight_end_2) = run_functional_test(matlab_dir, flags_2)

    # Compare outputs to each other
    assert flight_time_1 == pytest.approx(flight_time_2, abs=0.001), "Flight time outputs differ"
    assert flight_start_1 == pytest.approx(flight_start_2, abs=0.001), "Flight start outputs differ"
    assert flight_end_1 == pytest.approx(flight_end_2, abs=0.001), "Flight end outputs differ"

    # For this test, compare collision checks between drones and between drones and walls.
    # Other outputs are not checked; todo: account for NaNs in the comparison
    assert all([a == b for a, b in zip(coll_drones_1, coll_drones_2)])
    assert all([a == b for a, b in zip(coll_walls_1, coll_walls_2)])

def test_simulator_multifail_single_agent():
    # Get simulator directory from constants
    matlab_dir = str(SIM_DIR)

    # Run simulation with consecutive iterations
    flags_1 = {'run_parallel': False, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # first try consecutive runs
    with tic_toc():
        (coll_result_1, coll_obs_1, coll_drones_1, coll_walls_1,
         flight_time_1, flight_start_1, flight_end_1) = run_multifail_test_single_agent(matlab_dir, flags_1)

    # Run simulation in parallel
    flags_2 = {'run_parallel': True, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # now try parallel runs
    with tic_toc():
        (coll_result_2, coll_obs_2, coll_drones_2, coll_walls_2,
         flight_time_2, flight_start_2, flight_end_2) = run_multifail_test_single_agent(matlab_dir, flags_2)

    # Compare outputs to each other
    assert flight_time_1 == pytest.approx(flight_time_2, abs=0.001), "Flight time outputs differ"
    assert flight_start_1 == pytest.approx(flight_start_2, abs=0.001), "Flight start outputs differ"
    assert flight_end_1 == pytest.approx(flight_end_2, abs=0.001), "Flight end outputs differ"

    # For this test, compare collision checks between drones and between drones and walls.
    # Other outputs are not checked; todo: account for NaNs in the comparison
    assert all([a == b for a, b in zip(coll_drones_1, coll_drones_2)])
    assert all([a == b for a, b in zip(coll_walls_1, coll_walls_2)])

def test_simulator_multifail_multi_agents():
    # Get simulator directory from constants
    matlab_dir = str(SIM_DIR)

    # Run simulation with consecutive iterations
    flags_1 = {'run_parallel': False, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # first try consecutive runs
    with tic_toc():
        (coll_result_1, coll_obs_1, coll_drones_1, coll_walls_1,
         flight_time_1, flight_start_1, flight_end_1) = run_multifail_test_multi_agents(matlab_dir, flags_1)

    # Run simulation in parallel
    flags_2 = {'run_parallel': True, 'viz': False, 'save': False, 'dynamic_end': True, 'num_neig': 10,
             'map_option': 0}  # now try parallel runs
    with tic_toc():
        (coll_result_2, coll_obs_2, coll_drones_2, coll_walls_2,
         flight_time_2, flight_start_2, flight_end_2) = run_multifail_test_multi_agents(matlab_dir, flags_2)

    # Compare outputs to each other
    assert flight_time_1 == pytest.approx(flight_time_2, abs=0.001), "Flight time outputs differ"
    assert flight_start_1 == pytest.approx(flight_start_2, abs=0.001), "Flight start outputs differ"
    assert flight_end_1 == pytest.approx(flight_end_2, abs=0.001), "Flight end outputs differ"

    # For this test, compare collision checks between drones and between drones and walls.
    # Other outputs are not checked; todo: account for NaNs in the comparison
    assert all([a == b for a, b in zip(coll_drones_1, coll_drones_2)])
    assert all([a == b for a, b in zip(coll_walls_1, coll_walls_2)])
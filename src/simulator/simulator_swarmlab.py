import matlab.engine
import numpy as np
import time

# Use the matlab engine to run the simulation; designed to execute a set of runs;
# otherwise, the engine startup time vs. simulation time may not be worth it


class SimulatorSwarmlab:
    """
    A wrapper for interacting with the MATLAB-based swarm simulation engine.

    This class initializes a MATLAB engine, facilitates data exchange between Python
    and MATLAB, and provides methods to run simulations and process results.

    Attributes:
        eng (matlab.engine.MatlabEngine): The MATLAB engine instance used to run simulations.
    """

    # Initialize the MATLAB engine in the constructor
    def __init__(self, matlab_dir):
        """
        Initialize the SimulatorSwarmlab object.

        Starts a MATLAB engine and sets up the necessary paths to the simulation environment.

        Parameters:
            matlab_dir (str): Path to the MATLAB directory containing the simulation code
                              and wrappers. If not provided, the engine is not started.
        """

        if matlab_dir:
            self.start_engine(matlab_dir)
        else:
            self.eng = None

    # In case something tries to deepcopy this object, return an empty object (don't want to deepcopy matlab eng obj)
    def __deepcopy__(self, memo):
        """
        Create a shallow copy of the SimulatorSwarmlab object without duplicating the MATLAB engine.

        This method prevents issues that have happened before when attempting to deepcopy the MATLAB engine.

        Parameters:
            memo (dict): A dictionary to track objects already copied to avoid recursion.

        Returns:
            SimulatorSwarmlab: A new SimulatorSwarmlab object with no MATLAB engine initialized.
        """

        # Avoid recursion
        if id(self) in memo:
            return memo[id(self)]

        # Create a new object with no MATLAB engine
        new_obj = SimulatorSwarmlab('')
        memo[id(self)] = new_obj

        return new_obj

    # Start engine
    def start_engine(self, matlab_dir, max_retries=3, delay_sec=5):
        """
        Start the MATLAB engine with a specified directory.

        This method initializes the MATLAB engine and adds the specified directory to the MATLAB path.

        Parameters:
            matlab_dir (str): Path to the MATLAB directory containing the simulation code
                              and wrappers.
            max_retries (int): Maximum number of attempts to start the MATLAB engine.
                               Default is 3.
            delay_sec (int): Delay in seconds between retry attempts. Default is 5 seconds.
        """

        for attempt in range(max_retries):
            try:
                self.eng = matlab.engine.start_matlab()
                s = self.eng.genpath(matlab_dir)
                self.eng.addpath(s, nargout=0)
                print(f"MATLAB engine started successfully with directory: {matlab_dir}")
                return  # exit the loop and method if successfully started
            except Exception as e:
                print(f"Attempt {attempt+1} of {max_retries} failed to start MATLAB engine: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying after a short pause...")
                    time.sleep(delay_sec)
                else:
                    raise RuntimeError(f"Failed to start MATLAB engine.") from e

    # Shutdown engine
    def shutdown_engine(self):
        """
        Shutdown the MATLAB engine.

        This method stops the MATLAB engine if it is currently running, freeing up resources.
        """

        if self.eng:
            self.delete_parallel_pool()
            self.eng.quit()
            self.eng = None
            print(f"MATLAB engine shut down successfully.")

    # Restart engine
    def restart_engine(self, matlab_dir):
        """
        Restart the MATLAB engine with a new directory.

        This method stops the current MATLAB engine and starts a new one with the specified directory.

        Parameters:
            matlab_dir (str): Path to the MATLAB directory containing the simulation code
                              and wrappers.
        """

        self.shutdown_engine()
        self.start_engine(matlab_dir)

    # Start parallel pool
    def start_parallel_pool(self, max_retries=3, delay_sec=5):
        """
        Start a parallel pool in MATLAB.

        This method initializes a parallel pool in MATLAB to enable parallel execution of simulations. Does nothing if
        a parallel pool is already running.

        Parameters:
            matlab_dir (str): Path to the MATLAB directory containing the simulation code
                              and wrappers.
            max_retries (int): Maximum number of attempts to start the MATLAB engine.
                               Default is 3.
            delay_sec (int): Delay in seconds between retry attempts. Default is 5 seconds.
        """

        for attempt in range(max_retries):
            try:
                self.eng.eval("if isempty(gcp('nocreate')), parpool('Processes'); end", nargout=0)
                return  # exit the loop and method if successfully started
            except Exception as e:
                print(f"Attempt {attempt+1} of {max_retries} MATLAB engine failed to start parallel pool: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying after a short pause...")
                    time.sleep(delay_sec)
                else:
                    raise RuntimeError(f"Failed to start parallel pool.") from e

    # Shut down parallel pool
    def delete_parallel_pool(self):
        """
        Delete the parallel pool in MATLAB.

        This method stops the current parallel pool in MATLAB, freeing up resources. If no parallel pool is running,
        it does nothing.

        Returns:
            None
        """

        self.eng.eval("if ~isempty(gcp('nocreate')), delete(gcp('nocreate')); end", nargout=0)


    # Run the simulation with the given parameters
    def simulate(self, x, flags):
        """
        Run a swarm simulation using the MATLAB engine.

        This method processes input parameters, transfers them to MATLAB, executes the
        simulation, and retrieves the results.

        Parameters:
            x (numpy.ndarray): A 2D array where each row corresponds to a single simulation's
                               parameters, with the following columns:
                               - End time
                               - Start plane x-coordinate
                               - End plane x-coordinate
                               - Number of agents
                               - Failing agent number (or NaN if none)
                               - Failure start time (or NaN if none)
                               - Failure occurs in swarm (boolean or NaN if not applicable)
            flags (dict): Configuration options for the simulation, passed directly to MATLAB.
                          For example:
                          - 'run_parallel' (bool): Whether to run the simulation in parallel.
                          - 'viz' (bool): Whether to enable visualization.
                          - 'save' (bool): Whether to save the results.

        Returns:
            tuple: Outputs from the simulation, including:
                - coll_result: Collision check results.
                - coll_obs: Collisions with obstacles.
                - coll_drones: Collisions between drones.
                - coll_walls: Collisions with walls.
                - flight_time: Total flight times, rounded to the given precision.
                - flight_start: Start times of flights, rounded to the given precision.
                - flight_end: End times of flights, rounded to the given precision.
        """

        # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
        # https://www.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html

        # End time
        end_time = x[:, 0]

        # Start and end planes
        start_x = x[:, 1]
        end_x = x[:, 2]

        # Number of agents
        num_agents = x[:, 3]

        # Failure conditions
        fail_agent_num = x[:, 4]
        fail_time = x[:, 5]
        fail_in_swarm = x[:, 6]

        # Convert data type to hand over to MATLAB
        end_time = matlab.double(end_time.tolist())
        start_x = matlab.double(start_x.tolist())
        end_x = matlab.double(end_x.tolist())
        num_agents = matlab.double(num_agents.tolist())
        fail_agent_num = matlab.double(fail_agent_num.tolist())
        fail_time = matlab.double(fail_time.tolist())
        fail_in_swarm = matlab.double(fail_in_swarm.tolist())

        # Run the simulation
        (coll_result, coll_obs, coll_drones, coll_walls,
            flight_time, flight_start, flight_end) = self.eng.wrapper_sim(
            end_time, start_x, end_x, num_agents, fail_agent_num, fail_time, fail_in_swarm, flags,
            nargout=7)

        # Convert outputs that are doubles into ndarrays; Precision equal to 0.01 time steps
        const_time_step = 0.01
        flight_time = self.double_to_array(flight_time, 2)
        flight_start = self.double_to_array(flight_start, 2)
        flight_end = self.double_to_array(flight_end, 2)

        return coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end

    # Run the simulation with the given parameters
    def multifail_simulate(self, x, x_agents, x_time, x_power, flags):
        """
        Run a swarm simulation using the MATLAB engine. Same as "simulate" but with the ability to specify multiple
        agents for failure at their own fail times.
        """

        # Organize inputs to feed into the simulation (rows = indiv, cols = variables)
        # https://www.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html

        # End time
        end_time = x[:, 0]

        # Start and end planes
        start_x = x[:, 1]
        end_x = x[:, 2]

        # Number of agents
        num_agents = x[:, 3]

        # Failure conditions (fail_in_swarm moved to index 4 from 6 compared to original 'simulate' method)
        fail_in_swarm = x[:, 4]

        # Number of failing agents
        num_fail_agents = x_agents.shape[1]
        num_search_indiv = x_agents.shape[0]

        # Iterate through remaining pairs of columns to get failure agent number and time and put into two lists:
        # One list for agent numbers and one list for times
        fail_agent_num = []
        fail_time = []
        fail_power = []
        for i in range(num_search_indiv):
            fail_agent_num.append(x_agents[i, :])
            fail_time.append(x_time[i, :])
            fail_power.append(x_power[i, :])


        # Convert data type to hand over to MATLAB
        end_time = matlab.double(end_time.tolist())
        start_x = matlab.double(start_x.tolist())
        end_x = matlab.double(end_x.tolist())
        num_agents = matlab.double(num_agents.tolist())
        fail_in_swarm = matlab.double(fail_in_swarm.tolist())
        # fail_agent_num = matlab.double(fail_agent_num.tolist())
        # fail_time = matlab.double(fail_time.tolist())


        # Run the simulation
        (coll_result, coll_obs, coll_drones, coll_walls,
         flight_time, flight_start, flight_end) = self.eng.multifail_wrapper_sim(
            end_time, start_x, end_x, num_agents, fail_agent_num, fail_time, fail_power, fail_in_swarm, flags,
            nargout=7)

        # Convert outputs that are doubles into ndarrays; Precision equal to 0.01 time steps
        flight_time = self.double_to_array(flight_time, 2)
        flight_start = self.double_to_array(flight_start, 2)
        flight_end = self.double_to_array(flight_end, 2)

        return coll_result, coll_obs, coll_drones, coll_walls, flight_time, flight_start, flight_end

    def failure_modes_simulate(self, sim_config=None, swarm_config=None, sim_fail_config=None, results_dir=None, flags=None):
        """
        Run failure modes experiment using the MATLAB wrapper_run_generation_exp_failure_modes function.
        
        Parameters:
            sim_config (dict, optional): Simulation configuration
            swarm_config (dict, optional): Swarm configuration  
            sim_fail_config (dict, optional): Failure configuration
            results_dir (str, optional): Results directory path
            flags (dict, optional): Simulation flags
                
        Returns:
            MATLAB struct: Direct output from wrapper_run_generation_exp_failure_modes
        """
        
        # MATLAB engine automatically converts Python dicts to structs and handles basic types
        result = self.eng.wrapper_run_generation_exp_failure_modes(
            sim_config or {},
            swarm_config or {},
            sim_fail_config or {},
            results_dir or "",
            flags or {},
            nargout=1
        )
        
        return result

    # Convert 1D doubles to 1D numpy arrays with precision
    @staticmethod
    def double_to_array(double, precision=2):
        """
        Convert a MATLAB double array to a numpy array with specified precision.

        This static method takes a MATLAB double object (1D) and converts it into a
        numpy array. The results are rounded to a given precision.

        Parameters:
            double (matlab.double): The MATLAB double array to be converted.
            precision (int): The number of decimal places to round the results to. Default is 2.

        Returns:
            numpy.ndarray: A flattened numpy array with the specified precision.
        """

        output = np.array(double)
        output = np.round(output, precision)

        return output.flatten()

    # Close the MATLAB engine when object is deleted
    def __del__(self):
        """
        Close the MATLAB engine upon object deletion.

        This destructor ensures that the MATLAB engine instance is properly shut down
        to free resources when the SimulatorSwarmlab object is no longer in use.
        """

        if self.eng:
            self.shutdown_engine()

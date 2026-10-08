# Streamlined problem definition for failure modes using failure_modes_simulate interface
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR, TIME_STEP


class FailureModesProblem(Problem):
    """
    Streamlined GA problem for searching failure scenarios in UAV swarms.
    
    Uses the new failure_modes_simulate() interface and supports both power reduction
    and velocity change failures. Multiple failures can occur at different times during
    a single simulation, but all failures must be of the same type.
    """

    def __init__(self,
                 failure_option,      # 1=power reduction, 1.1=power cutoff, 2=velocity multi-agent, 2.1=velocity single-agent multi-event
                 num_failures,        # Number of failures per simulation
                 num_agents,          # Total number of agents in swarm
                 trial_num,           # Single trial number to test
                 functional_data_path, # Path to functional test data for timing bounds
                 end_time_mult=2.0,   # Multiplier for simulation end time based on functional flight time
                 v_ref=6.0,           # Reference velocity for velocity failure bounds
                 f_min_points=0.5,    # Points awarded at simulation end time
                 sim_config=None,     # Simulation configuration dict
                 swarm_config=None,   # Swarm configuration dict  
                 flags=None,          # Simulation flags
                 simulator=None       # Simulator object to execute MATLAB-based simulations
                ):
        
        self.failure_option = failure_option
        self.num_failures = num_failures
        self.num_agents = num_agents
        self.trial_num = trial_num
        self.end_time_mult = end_time_mult
        self.v_ref = v_ref
        self.f_min_points = f_min_points
        self.sim_config = sim_config or {}
        self.swarm_config = swarm_config or {}
        self.flags = flags or {}
        self.simulator = simulator
        self.generation = 0
        
        #NEW CODE TO INTILIAZE LISTS FOR SEEDING
        self.all_solutions_X = []
        self.all_solutions_F = []

        # Load functional test data for timing bounds
        self.functional_df = pd.read_csv(functional_data_path)
        
        # Cache functional flight data for this trial
        self.flight_start, self.flight_end = self._get_trial_time_bounds()
        self.functional_flight_time = self.flight_end - self.flight_start
        self.sim_end_time = self.flight_start + self.end_time_mult * self.functional_flight_time
        
        # Calculate variable bounds and dimensions
        if failure_option == 1:  # Power reduction failures
            # Variables: [agent_ids (num_failures), fail_times (num_failures), fail_powers (num_failures)]
            n_var = num_failures * 3
            
            # Bounds: agent_ids [1, num_agents], fail_times [trial-specific], fail_powers [0, 1]
            xl = np.concatenate([
                np.ones(num_failures),                    # agent_ids >= 1
                np.zeros(num_failures),                   # fail_times >= min_time (set per trial)
                np.zeros(num_failures)                    # fail_powers >= 0
            ])
            
            xu = np.concatenate([
                np.full(num_failures, num_agents),       # agent_ids <= num_agents
                np.full(num_failures, 1.0),            # fail_times <= max_time (set per trial)
                np.ones(num_failures)                     # fail_powers <= 1
            ])

        elif failure_option == 1.1:  # Power cutoff failure (fail_powers fixed at 0.0)
            # Variables: [agent_ids (num_failures), fail_times (num_failures)]
            n_var = num_failures * 2

            # Bounds: agent_ids [1, num_agents], fail_times [trial-specific]
            xl = np.concatenate([
                np.ones(num_failures),  # agent_ids >= 1
                np.zeros(num_failures)  # fail_times >= min_time (set per trial)
            ])

            xu = np.concatenate([
                np.full(num_failures, num_agents),  # agent_ids <= num_agents
                np.full(num_failures, 1.0)  # fail_times <= max_time (set per trial)
            ])
            
        elif failure_option == 2:  # Velocity change failures (multi-agent)
            # Variables: [agent_ids (num_failures), fail_times (num_failures), 
            #           fail_v_n (num_failures), fail_v_e (num_failures), fail_v_d (num_failures)]
            n_var = num_failures * 5
            
            # Bounds: agent_ids [1, num_agents], fail_times [trial-specific], velocities [-v_ref, v_ref]
            xl = np.concatenate([
                np.ones(num_failures),                    # agent_ids >= 1
                np.zeros(num_failures),                   # fail_times >= min_time (set per trial)
                np.full(num_failures, -v_ref),           # fail_v_n >= -v_ref
                np.full(num_failures, -v_ref),           # fail_v_e >= -v_ref
                np.full(num_failures, -v_ref)            # fail_v_d >= -v_ref
            ])
            
            xu = np.concatenate([
                np.full(num_failures, num_agents),       # agent_ids <= num_agents
                np.full(num_failures, 1.0),            # fail_times <= max_time (set per trial)
                np.full(num_failures, v_ref),            # fail_v_n <= v_ref
                np.full(num_failures, v_ref),            # fail_v_e <= v_ref
                np.full(num_failures, v_ref)             # fail_v_d <= v_ref
            ])
            
        elif failure_option == 2.1:  # Multi-event single-agent velocity failures
            # Variables: [agent_id (1), fail_times (num_failures), 
            #           fail_v_n (num_failures), fail_v_e (num_failures), fail_v_d (num_failures)]
            n_var = 1 + num_failures * 4
            
            # Bounds: single agent_id [1, num_agents], multiple events with velocities [-v_ref, v_ref]
            xl = np.concatenate([
                [1],                                      # single agent_id >= 1
                np.zeros(num_failures),                   # fail_times >= min_time (set per trial)
                np.full(num_failures, -v_ref),           # fail_v_n >= -v_ref
                np.full(num_failures, -v_ref),           # fail_v_e >= -v_ref
                np.full(num_failures, -v_ref)            # fail_v_d >= -v_ref
            ])
            
            xu = np.concatenate([
                [num_agents],                             # single agent_id <= num_agents
                np.full(num_failures, 1.0),            # fail_times <= max_time (set per trial)
                np.full(num_failures, v_ref),            # fail_v_n <= v_ref
                np.full(num_failures, v_ref),            # fail_v_e <= v_ref
                np.full(num_failures, v_ref)             # fail_v_d <= v_ref
            ])
            
        else:
            raise ValueError(f"Invalid failure_option: {failure_option}. Must be 1 (power reduction), 1.1 (power cutoff), 2 (velocity multi-agent), or 2.1 (velocity single-agent multi-event)")
            
        # Initialize Problem superclass
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=xl,
            xu=xu,
            elementwise=False,  # Vectorized evaluation for efficiency
            vtype=float  # Use float for continuous variables
        )
    
    def _get_trial_time_bounds(self):
        """Get flight start and end times for this problem's trial from functional data."""
        trial_data = self.functional_df[(self.functional_df['trial'] == self.trial_num)
                                        & (self.functional_df['num_agents'] == self.num_agents)]
        if len(trial_data) == 0:
            # Throw error if data not found
            raise ValueError(f"No functional data found for trial {self.trial_num}")
        elif len(trial_data) > 1:
            # Throw error if more than one row found
            raise ValueError(f"More than one functional test result found for this trial and swarm size.")
        
        flight_start = trial_data['flight_start'].iloc[0]
        flight_end = trial_data['flight_end'].iloc[0]
        return flight_start, flight_end
    
    
    def decode_variables(self, x):
        """Decode GA variables into failure parameters."""
        n_ind = x.shape[0]
        
        # Get time bounds for scaling failure times from [0,1] to [flight_start, flight_end]
        min_time, max_time = self._get_trial_time_bounds()
        time_range = max_time - min_time
        
        if self.failure_option == 1:  # Power failures
            # Extract agent IDs (round to integers and ensure uniqueness)
            agent_ids = np.round(x[:, :self.num_failures]).astype(int)

            # Extract and scale failure times from [0,1] to trial bounds, then round to TIME_STEP
            fail_times_raw = x[:, self.num_failures:2*self.num_failures]
            fail_times = np.round((min_time + fail_times_raw * time_range) / TIME_STEP) * TIME_STEP

            # Extract failure powers
            fail_powers = x[:, 2*self.num_failures:3*self.num_failures]

            return agent_ids, fail_times, fail_powers, None, None, None

        elif self.failure_option == 1.1:  # Power failures - complete cutoff
            # Extract agent IDs (round to integers and ensure uniqueness)
            agent_ids = np.round(x[:, :self.num_failures]).astype(int)

            # Extract and scale failure times from [0,1] to trial bounds, then round to TIME_STEP
            fail_times_raw = x[:, self.num_failures:2*self.num_failures]
            fail_times = np.round((min_time + fail_times_raw * time_range) / TIME_STEP) * TIME_STEP

            # Failure powers for this failure option will always be zero
            fail_powers = np.zeros((n_ind, self.num_failures))

            return agent_ids, fail_times, fail_powers, None, None, None
            
        elif self.failure_option == 2:  # Velocity failures (multi-agent)
            # Extract agent IDs (round to integers and ensure uniqueness)
            agent_ids = np.round(x[:, :self.num_failures]).astype(int)

            # Extract and scale failure times from [0,1] to trial bounds, then round to TIME_STEP
            fail_times_raw = x[:, self.num_failures:2*self.num_failures]
            fail_times = np.round((min_time + fail_times_raw * time_range) / TIME_STEP) * TIME_STEP

            # Extract velocity components
            fail_v_n = x[:, 2*self.num_failures:3*self.num_failures]
            fail_v_e = x[:, 3*self.num_failures:4*self.num_failures]
            fail_v_d = x[:, 4*self.num_failures:5*self.num_failures]

            return agent_ids, fail_times, None, fail_v_n, fail_v_e, fail_v_d
            
        elif self.failure_option == 2.1:  # Multi-event single-agent velocity failures
            # Extract single agent ID (replicated for all events for each individual)
            single_agent_ids = np.round(x[:, 0]).astype(int)  # One agent ID per individual
            agent_ids = np.repeat(single_agent_ids[:, np.newaxis], self.num_failures, axis=1)  # Replicate for all events

            # Extract and scale failure times from [0,1] to trial bounds, then round to TIME_STEP
            fail_times_raw = x[:, 1:1+self.num_failures]
            fail_times = np.round((min_time + fail_times_raw * time_range) / TIME_STEP) * TIME_STEP

            # Extract velocity components
            fail_v_n = x[:, 1+self.num_failures:1+2*self.num_failures]
            fail_v_e = x[:, 1+2*self.num_failures:1+3*self.num_failures]
            fail_v_d = x[:, 1+3*self.num_failures:1+4*self.num_failures]

            return agent_ids, fail_times, None, fail_v_n, fail_v_e, fail_v_d

        else:
            # Throw error if invalid option
            raise ValueError(f"Invalid failure_option: {self.failure_option}. Must be 1 (power reduction), 1.1 (power cutoff), 2 (velocity multi-agent), or 2.1 (velocity single-agent multi-event)")
    
    def _calculate_fitness(self, times_end):
        """
        Calculate fitness based on agent completion times using point system.
        
        Parameters:
            times_end_result (array-like): Flight end times for all agents from MATLAB result
            
        Returns:
            float: Fitness value (sum of points across all agents, negated for minimization)
        """
        
        # Check for NaN values and throw error
        if np.any(np.isnan(times_end)):
            raise ValueError(f"NaN values found in flight end times: {times_end}")
        
        total_points = 0.0
        functional_end_time = self.flight_end
        
        for agent_end_time in times_end:
            if agent_end_time > self.sim_end_time:
                # Agent did not complete flight within simulation time (equivalent to "inf")
                points = 0.0
            elif agent_end_time <= functional_end_time:
                # Agent completed within original functional flight time
                points = 1.0
            else:
                # Linear interpolation between functional flight time and sim end time
                # At functional flight time: 1.0 points
                # At sim end time: f_min_points
                points = 1.0 + (self.f_min_points - 1.0) * (agent_end_time - functional_end_time) / (self.sim_end_time - functional_end_time)
            
            total_points += points
        
        # GA minimizes and max points is good for the swarm, so minimize to challenge the swarm.
        return total_points
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population using failure_modes_simulate."""
        
        self.generation += 1
        print(f"Starting generation {self.generation}")
        
        n_ind = x.shape[0]
        penalty = 1000.0
        
        # Decode variables
        agent_ids, fail_times, fail_powers, fail_v_n, fail_v_e, fail_v_d = self.decode_variables(x)
        
        # Check for duplicate agent IDs in each individual (invalid solutions)
        # For failure_option == 2.1, duplicates are expected (same agent, multiple events)
        if self.failure_option == 2.1:
            # All individuals are valid for single-agent multi-event mode
            is_valid = np.ones(n_ind, dtype=bool)
        else:
            # Check for duplicate agent IDs (invalid for multi-agent modes)
            is_valid = np.array([len(set(row)) == len(row) for row in agent_ids])
        n_valid = np.sum(is_valid)
        
        # Initialize outputs
        fitness_all = np.full(n_ind, penalty)
        flight_time_all = np.full(n_ind, np.nan)
        collision_occur_all = np.full(n_ind, False)
        
        if n_valid > 0:
            # Failure times are already scaled and rounded in _decode_variables
            valid_fail_times = fail_times[is_valid]
            
            # Use pre-calculated simulation end time
            sim_end_time = self.sim_end_time
            
            # Prepare simulation configuration (one entry per valid individual)
            # Convert all scalar values to lists of length n_valid
            sim_config = {}
            for key, value in self.sim_config.items():
                if key == 'end_time':
                    sim_config[key] = [sim_end_time] * n_valid  # Use calculated end time
                else:
                    sim_config[key] = [value] * n_valid
            
            # Prepare failure configuration (one entry per valid individual)
            if self.failure_option == 1 or self.failure_option == 1.1:  # Power failures
                sim_fail_config = {
                    'failure_option': [1] * n_valid,  # For this failure type, use 1 for simulator
                    'fail_agent_num': [agent_ids[is_valid][i] for i in range(n_valid)],
                    'fail_time': [valid_fail_times[i] for i in range(n_valid)],
                    'fail_power': [fail_powers[is_valid][i] for i in range(n_valid)],
                    'fail_in_swarm': [True] * n_valid
                }
            elif self.failure_option == 2 or self.failure_option == 2.1:  # Velocity failures (multi-agent or single-agent multi-event)
                sim_fail_config = {
                    'failure_option': [2] * n_valid,  # For this failure type, use 2 for simulator
                    'fail_agent_num': [agent_ids[is_valid][i] for i in range(n_valid)],
                    'fail_time': [valid_fail_times[i] for i in range(n_valid)],
                    'fail_v_n': [fail_v_n[is_valid][i] for i in range(n_valid)],
                    'fail_v_e': [fail_v_e[is_valid][i] for i in range(n_valid)],
                    'fail_v_d': [fail_v_d[is_valid][i] for i in range(n_valid)],
                    'fail_in_swarm': [True] * n_valid
                }
            else:
                raise ValueError(f"Invalid failure_option value: {self.failure_option}.")
            
            # Run simulation
            result = self.simulator.failure_modes_simulate(
                sim_config=sim_config,
                swarm_config=self.swarm_config,
                sim_fail_config=sim_fail_config,
                results_dir="",
                flags=self.flags
            )

            # Process results - iterate through each population individual
            for i, valid_idx in enumerate(np.where(is_valid)[0]):
                # Flight times
                times_start = np.array(result['times_start'][valid_idx]).flatten()
                times_end = np.array(result['times_end'][valid_idx]).flatten()

                if len(times_start) > 0 and len(times_end) > 0:
                    flight_start = np.min(times_start)
                    flight_end = np.max(times_end)
                    flight_time = flight_end - flight_start

                    flight_time_all[valid_idx] = round(flight_time, 2)  # Round to hundredth of a second

                else:
                    # Throw an error
                    raise ValueError(f"Simulation returned empty times for individual {valid_idx} in generation {self.generation}")

                # Check for collisions occurring within the end time
                collision_data = np.array(result['collision'][valid_idx]).flatten()
                if collision_data.size > 0 and np.any(collision_data):
                    collision_occur_all[valid_idx] = True

                # Fitness: calculate using point system based on agent completion times
                fitness_all[valid_idx] = self._calculate_fitness(times_end)
        
        # Assign outputs
        out["F"] = fitness_all
        out["flight_time"] = flight_time_all
        out["collision_occur"] = collision_occur_all

        #NEW CODE TO ADD FUNCTION FOR UPDATING SEEDS
        # Store all solutions and fitness for seeding
        self.all_solutions_X.append(x.copy())
        self.all_solutions_F.append(fitness_all.copy())
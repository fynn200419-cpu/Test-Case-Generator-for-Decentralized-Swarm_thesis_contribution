import pickle
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling


class MixedSeedSampling(Sampling):
    """
    Places best seed_solutions first (sorted by fitness), 
    then uses FloatRandomSampling to fill the rest.
    
    seed_solutions: ndarray (n_seed, n_var)
    seed_fitness: ndarray (n_seed,) — fitness values corresponding to each seed
    """

# ndarray is np and is used to store data in array format. Array has fixed size and mathermatical operations can be performed on it ithout needing to iterate through them.
# pymoo expects ndarray as input for population and fitness values.
     
     #following call chain: run_ga_failure_modes -> build_seeded_sampler -> MixedSeedSampling.__init__() -> GA starts -> pymoo calls MixedSeedSampling._do()
    def __init__(self, seed_solutions, seed_fitness=None):
        # seed_solutions: This is expected to be a NumPy ndarray where each row is one of the best solutions from the previous GA run. This input is provided by build_seeded_sampler.
        # seed_fitness: An optional ndarray containing the fitness value for each corresponding solution in seed_solutions. If seed-Fitness is passed, it wont be None, if it does not pass a seed_fitness, it will be None.
        super().__init__()
        self.seed_solutions = np.asarray(seed_solutions, dtype=float)
        #ensures its a NumPy array with float data type.
        self.seed_fitness = None
        #safe default in case no fitness data is provided.
        
        if seed_fitness is not None:
            self.seed_fitness = np.asarray(seed_fitness, dtype=float).flatten()
            #if seed_fitness was actualy provided, it converts them into NumPy array of floats and flattens it to 1D because that simplifies sorting and indexing later.
            # Ensure fitness array matches number of seeds
            if self.seed_fitness.shape[0] != self.seed_solutions.shape[0]:
                #compare number of rows of solutions with length of fitness array and therefore number of fitness values.
                print(f"Warning: seed_fitness size {self.seed_fitness.shape[0]} != seed_solutions rows {self.seed_solutions.shape[0]}")
                self.seed_fitness = None
        
        self.random_sampler = FloatRandomSampling()
        #this is not used currently because in do I use an own method of generating random samples because the random Sampler did not respect the correct shape and therefore crashed the whole GA.
        self.n_var_seed = self.seed_solutions.shape[1]
        # It creates an instance of pymoo's default random sampler. This object will be used inside the _do method to generate the non-seeded part of the population.
        # It inspects the shape of the seed_solutions array (.shape[1] gets the number of columns) and stores the number of variables. This is used in the _do method as a final validation step to ensure the problem dimensions haven't changed unexpectedly.

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate initial population:
        - First seeds_to_use rows: best seeds (sorted by fitness ascending)
        - Remaining rows: random samples from FloatRandomSampling
        
        Ensures 66% random + 34% seeds constraint.
        """
        #n_samples is the total population size requested by the GA.
        n_seed = self.seed_solutions.shape[0]
        #by now the seed_solutions only contains the solutions with the n best ids so n_seed is the total number of available seeds.

        # Validate problem dimensions match seed shape
        if problem.n_var != self.n_var_seed:
            print(f"ERROR: problem.n_var={problem.n_var} but seed has {self.n_var_seed} columns")
            print(f"  problem.xl shape: {np.shape(problem.xl)}")
            print(f"  problem.xu shape: {np.shape(problem.xu)}")
            raise ValueError(
                f"Problem dimension mismatch: problem.n_var={problem.n_var} vs seed columns={self.n_var_seed}\n"
                f"Check that FailureModesProblem was initialized with correct failure_option/num_failures."
            )
        # This is a final, robust check to ensure that the problem object (which defines the current GA's variable space) is compatible with the seed_solutions that were loaded.
        # Action: If problem.n_var (number of variables in the current problem) does not match self.n_var_seed (number of variables in the loaded seeds), it raises a ValueError. This is a critical error because the seed solutions wouldn't fit the problem's definition.

        # Cap seeds to max 34% of requested population (ensure >=66% random)
        max_seed_pct = 0.34
        max_seeds = int(np.floor(max_seed_pct * n_samples))
        seeds_to_use = min(n_seed, max_seeds)
        # The actual number of seeds used will be the smaller of (a) the total number of available seeds (n_seed) and (b) the maximum allowed seeds based on the 34% cap.
         # First, generate a full population of random samples.
        # This ensures the base population always has the correct shape.
        X = self.random_sampler._do(problem, n_samples, **kwargs)
        if X.ndim == 1:
            X = np.atleast_2d(X)

        # If no seeds are to be used, we are done.
        if seeds_to_use <= 0:
            return X

        # --- Select BEST seeds and overwrite the start of the population ---
        # If we have fitness data, sort by it; otherwise take the first seeds_to_use.
        if self.seed_fitness is not None:
            # Sort by fitness ascending (lower = better, assumes minimization).
            best_idx = np.argsort(self.seed_fitness)[:seeds_to_use]
            seed_rows = self.seed_solutions[best_idx]
            best_fitness = self.seed_fitness[best_idx]
            print(f"[SEEDING] Replacing first {seeds_to_use} random samples with best seeds (fitness: {best_fitness})")
        else:
            # No fitness info: take the first seeds_to_use rows as they are.
            seed_rows = self.seed_solutions[:seeds_to_use]
            print(f"[SEEDING] Replacing first {seeds_to_use} random samples with seeds (no fitness ranking available)")

        # Overwrite the first part of the randomly generated population with the seed solutions.
        X[:seeds_to_use] = seed_rows

        return X


def build_seeded_sampler(seed_pickle_path, problem, n_seed=10):
    """
    Factory to create MixedSeedSampling from a saved seed pickle.
    
    Loads 'X' (population) and 'F' (fitness) from pickle, validates compatibility
    with the current problem, and returns a MixedSeedSampling that selects
    the best seeds by fitness.

    Parameters
    ----------
    seed_pickle_path : str or Path or None
        Path to pickle file with {'X': ndarray, 'F': ndarray}.
        If None, returns FloatRandomSampling (no seeding).
    problem : pymoo Problem
        Current problem instance (used to validate n_var and bounds).
    n_seed : int
        Max number of seeds to use from the loaded population.

    Returns
    -------
    pymoo.core.sampling.Sampling
        MixedSeedSampling if seeds loaded and compatible, else FloatRandomSampling.
    """
    #seed_pickle_path is path to the pickle file containing previous run's seeds. Only X and F arrays are there and they are saved in run_ga_failure_modes.py after the GA run. and failure_modes_problem.py defines the problem.


    # No seeding requested
    if seed_pickle_path is None:
        return FloatRandomSampling()

    # Load pickle
    try:
        with open(seed_pickle_path, 'rb') as f:
            seed_data = pickle.load(f)
        #attempts to load data from that pickle file. If it fails (file not found, corrupted, etc.), it catches the exception and returns FloatRandomSampling as a fallback.
    except Exception as e:
        print(f"Warning: Could not load seed data from {seed_pickle_path}: {e}")
        return FloatRandomSampling()

    # Extract X and F
    X_prev = seed_data.get('X')
    F_prev = seed_data.get('F')
    
    if X_prev is None or F_prev is None:
        print("Warning: seed_data missing 'X' or 'F'. Using pure random sampling.")
        return FloatRandomSampling()
        #fallback to ensure no crash happens.
    X_prev = np.asarray(X_prev, dtype=float)
    F_prev = np.asarray(F_prev, dtype=float)
    
    # Ensure 2D
    if X_prev.ndim == 1:
        X_prev = np.atleast_2d(X_prev)
        #if it is 1D, convert to 2D with one row. Pymoo expects a 2D array where each row is a solution. Thats why this is necessary.
    if F_prev.ndim == 1:
        F_prev = F_prev.reshape(-1)
        #if F_prev is a 2D array with a single column, it flattens it to a 1D array. This is the expected format from pymoo for fitness values.

    # Validate dimensions match current problem
    if X_prev.shape[1] != problem.n_var:
        #if the number of columns in X_prev (i.e., the number of variables per solution) does not match problem.n_var, it indicates a mismatch between the previous run's problem definition and the current one.
        print(f"ERROR: Loaded seed X has {X_prev.shape[1]} columns "
              f"but problem.n_var={problem.n_var}. Incompatible!")
        print(f"  This likely means failure_option or num_failures changed between runs.")
        return FloatRandomSampling()

    # Ensure fitness and X rows match
    if F_prev.shape[0] != X_prev.shape[0]:
        print(f"Warning: F has {F_prev.shape[0]} elements but X has {X_prev.shape[0]} rows. "
              f"Trimming to match.")
        min_rows = min(F_prev.shape[0], X_prev.shape[0])
        X_prev = X_prev[:min_rows]
        F_prev = F_prev[:min_rows]

    # Select top n_seed by best (lowest) fitness
    best_idx = np.argsort(F_prev)[:n_seed]
    seed_solutions = X_prev[best_idx]
    seed_fitness = F_prev[best_idx]

    print(f"Loaded {seed_solutions.shape[0]} seed solutions from {seed_pickle_path}")
    print(f"  Best fitness values: {seed_fitness}")
    
    return MixedSeedSampling(seed_solutions=seed_solutions, seed_fitness=seed_fitness)
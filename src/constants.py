# Store constants here; these should NOT be changed because they affect experiment configurations AND postprocessing.

from pathlib import Path
import numpy as np


# Project home directory and results directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'

# Directory where simulator script is located
SIM_DIR = PROJECT_ROOT / '..' / 'swarmlab-experimental' / 'projects' / 'robustness-failures-obstacles'

# Number of trials
MAX_NUM_TRIALS = 100  # Assuming no more than 100 unique trials used.

# Vector of seeds to match trial
SEED_RANGE = np.arange(10, (MAX_NUM_TRIALS + 1) * 10, 10)  # Range starting at 10 with step size of 10

# Time step size
TIME_STEP = 0.01

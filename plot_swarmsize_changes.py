# ...existing code...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths to your result folders - update these paths
results_paths = {
    '10': 'results_to_save/exp_swarmsize/result_20251104_153417_functional_rInf/dataframe.csv',
    '15': 'results_to_save/exp_swarmsize/result_20251104_155444_functional_rInf/dataframe.csv',
    '20': 'results_to_save/exp_swarmsize/result_20251105_122351_functional_rInf/dataframe.csv'
}

# Load all dataframes
dataframes = {}
for size, path in results_paths.items():
    df = pd.read_csv(path)
    dataframes[size] = df


# use the parent "exp_swarmsize" directory of the first result path
_example_path = list(results_paths.values())[0]
OUTPUT_DIR = Path(_example_path).parents[1]  # results_to_save/exp_swarmsize
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ...existing code...

def plot_completion_comparison(spacing_factor: float = 1.8):
    """Plot which maps failed (coll_result_occur == True) by each swarm size.
    spacing_factor increases spacing between trial ticks for better readability.
    """
    plt.figure(figsize=(14, 6))
    
    # use trial indices from the first dataframe as x-axis reference
    ref_df = dataframes[list(dataframes.keys())[0]]
    trials = ref_df['trial'].values
    n_trials = len(trials)
    # base positions spaced out
    base_positions = np.arange(n_trials) * spacing_factor
    n_sizes = len(dataframes)
    bar_width = (spacing_factor * 0.8) / n_sizes  # fit bars within one spaced tick
    
    for i, (size, df) in enumerate(dataframes.items()):
        # Use coll_result_occur: True means a collision occurred -> failed
        failed = df['coll_result_occur'].fillna(False).astype(bool).values
        positions = base_positions + i * bar_width - ( (n_sizes-1) * bar_width / 2 )
        plt.bar(positions, failed.astype(int), width=bar_width, label=f'{size} agents (failed)', alpha=0.8)
        # plot marker above failed bars
        failed_pos = positions[failed]
        if failed_pos.size:
            plt.scatter(failed_pos, np.ones_like(failed_pos)*1.05, marker='x', color='red', s=60, zorder=10)
    
    # set xticks at center of grouped bars
    center_positions = base_positions
    plt.xticks(center_positions, trials)
    plt.xlim(center_positions[0] - spacing_factor*0.6, center_positions[-1] + spacing_factor*0.6)
    plt.xlabel('Map Index (trial)')
    plt.ylabel('Failed (1=Yes, 0=No)')
    plt.title('Map Failure by Swarm Size (based on coll_result_occur)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'completion_comparison.png'))
    plt.close()

def plot_flight_time_comparison(spacing_factor: float = 1.8):
    """Grouped bar chart: flight times for each map across swarm sizes, with wider spacing."""
    plt.figure(figsize=(16, 6))
    
    ref_df = dataframes[list(dataframes.keys())[0]]
    trials = ref_df['trial'].values
    n_trials = len(trials)
    base_positions = np.arange(n_trials) * spacing_factor
    n_sizes = len(dataframes)
    bar_width = (spacing_factor * 0.8) / n_sizes
    
    for i, (size, df) in enumerate(dataframes.items()):
        ft_series = pd.Series(df['flight_time'].values, index=df['trial'].values)
        times = ft_series.reindex(trials).replace([np.inf], np.nan).values
        positions = base_positions + i * bar_width - ( (n_sizes-1) * bar_width / 2 )
        plt.bar(positions, times, width=bar_width, label=f'{size} agents', alpha=0.9)
    
    plt.xticks(base_positions, trials)
    plt.xlim(base_positions[0] - spacing_factor*0.6, base_positions[-1] + spacing_factor*0.6)
    plt.xlabel('Map Index (trial)')
    plt.ylabel('Flight Time')
    plt.title('Flight Time per Map by Swarm Size (bars)')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'flight_time_comparison.png'))
    plt.close()

def print_statistics():
    """Print completion/failure rates and average flight times using coll_result_occur"""
    print("\nStatistics:")
    print("-" * 40)
    
    for size, df in dataframes.items():
        failed = df['coll_result_occur'].fillna(False).astype(bool).sum()
        total = len(df)
        avg_time = df[~df['flight_time'].replace([np.inf], np.nan).isna()]['flight_time'].mean()
        
        print(f"\nSwarm size: {size} agents")
        print(f"Maps failed (collisions): {failed}/{total} ({failed/total*100:.1f}%)")
        print(f"Average flight time (completed maps): {avg_time:.2f}")

if __name__ == "__main__":
    plot_completion_comparison(spacing_factor=1.8)
    plot_flight_time_comparison(spacing_factor=1.8)
    print_statistics()
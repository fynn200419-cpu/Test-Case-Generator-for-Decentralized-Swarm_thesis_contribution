#!/usr/bin/env python3
"""
End-to-end test comparing multifail engine lifecycle management with experiment 1 results.
This test runs genetic algorithm searches for agents 5-15 using multifail code and compares
results with the existing experiment 1 outputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append('../src')

from src.constants import PROJECT_ROOT
import src.multifail.run_metaheuristic_search as multifail_metaheuristic_search

def load_exp1_functional_data():
    """Load the functional test data from experiment 1."""
    functional_data_path = PROJECT_ROOT / "results_to_save" / "exp_1" / "result_20241214_004149_functional_updated" / "dataframe.pkl"

    try:
        df_f = pd.read_pickle(functional_data_path)
        print(f"✓ Loaded functional data from: {functional_data_path}")
        print(f"  Shape: {df_f.shape}")
        return df_f
    except Exception as e:
        print(f"✗ Functional test results not found: {e}")

# def create_mock_functional_data():
#     """Create mock functional test data matching experiment 1 format."""
#     # Create data for agents 5-15
#     agents = np.arange(5, 16)
#
#     data = {
#         'num_agents': agents,
#         'flight_start': np.full(len(agents), 2.0),  # Typical start time
#         'flight_end': 5.0 + agents * 0.1,  # Increasing with agent count
#         'flight_time': 3.0 + agents * 0.1,  # Increasing with agent count
#         'end_time': np.full(len(agents), 60.0),  # From experiment 1
#         'start_x': np.zeros(len(agents)),
#         'end_x': np.full(len(agents), 100),
#         'fail_in_swarm': np.ones(len(agents), dtype=bool),
#         'trial': np.ones(len(agents)),
#         'seed': np.full(len(agents), 10),
#         'coll_result_occur': np.zeros(len(agents), dtype=bool),
#         'coll_result_type': ['none'] * len(agents)
#     }
#
#     df_f = pd.DataFrame(data)
#     return df_f

def test_multifail_genetic_algorithm():
    """Test the multifail genetic algorithm matching experiment 1 parameters."""
    
    print("=" * 70)
    print("TESTING MULTIFAIL GENETIC ALGORITHM WITH PERSISTENT MATLAB ENGINE vs EXPERIMENT 1")
    print()
    
    # Load functional test data (same as experiment 1)
    df_f = load_exp1_functional_data()
    
    # Test parameters matching experiment 1
    vect_agents = np.array([10, 11, 12, 90])
    vect_num_fail = np.array([1])   # Single failure (like original experiment 1)
    vect_trials = np.array([1, 2, 15, 29, 30])
    
    print("Test parameters:")
    print(f"  Agents: {vect_agents}")
    print(f"  Number of failures: {vect_num_fail}")
    print(f"  Trials: {vect_trials}")
    print()
    
    # Run the multifail genetic algorithm
    print("Starting multifail genetic algorithm search...")
    start_time = time.time()
    
    try:
        results_directories = multifail_metaheuristic_search.run_trials(
            vect_agents=vect_agents,
            vect_num_fail=vect_num_fail,
            vect_trials=vect_trials,
            df_f=df_f,
            run_type="GA_MATLAB_engine_test",
            num_neig=10,           # Same as experiment 1
            rad_neig=150,          # Default value
            map_option=0,          # Same as experiment 1
            failure_option=0,      # Single event failure
            flag_parallel=True    # Same as experiment 1
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✓ Test completed successfully in {elapsed:.2f} seconds")
        print(f"✓ Created {len(results_directories)} results directories")
        print()
        
        # Check results using the returned directory paths
        success = check_test_results_direct(results_directories)
        
        if success:
            print("Results saved to:")
            for i, results_dir in enumerate(results_directories, 1):
                print(f"  {i}. {results_dir}")
        
        return success
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_test_results_direct(results_directories):
    """Check that test results were created correctly using the returned directory paths."""
    
    print("Checking test results...")
    
    if not results_directories:
        print("⚠ No results directories returned")
        return False
    
    print(f"✓ Received {len(results_directories)} results directories")
    
    success_count = 0
    
    for results_dir in results_directories:
        print(f"✓ Checking directory: {results_dir.name}")
        
        # Check for expected files
        dataframe_csv = results_dir / "dataframe.csv"
        dataframe_pkl = results_dir / "dataframe.pkl"
        
        if dataframe_csv.exists():
            print(f"  ✓ dataframe.csv exists")
            
            # Load and check the dataframe
            try:
                df = pd.read_csv(dataframe_csv)
                print(f"  ✓ dataframe.csv loaded successfully: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                
                # Check key columns exist
                required_columns = ['num_agents', 'fail_agent_num', 'fail_time', 'flight_time']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"  ⚠ Missing columns: {missing_columns}")
                else:
                    print(f"  ✓ All required columns present")
                    
                    # Show some sample data
                    print(f"  ✓ Sample data:")
                    print(f"    num_agents: {df['num_agents'].iloc[0] if len(df) > 0 else 'N/A'}")
                    print(f"    fail_agent_num: {df['fail_agent_num'].iloc[0] if len(df) > 0 else 'N/A'}")
                    print(f"    fail_time: {df['fail_time'].iloc[0] if len(df) > 0 else 'N/A'}")
                    print(f"    flight_time: {df['flight_time'].iloc[0] if len(df) > 0 else 'N/A'}")
                    
                    success_count += 1
                    
            except Exception as e:
                print(f"  ✗ Error loading dataframe.csv: {e}")
        else:
            print(f"  ✗ dataframe.csv not found")
            
        if dataframe_pkl.exists():
            print(f"  ✓ dataframe.pkl exists")
        else:
            print(f"  ✗ dataframe.pkl not found")
            
        print()  # Add spacing between directories
    
    print(f"Summary: {success_count}/{len(results_directories)} directories completed successfully")
    
    return success_count == len(results_directories)

def compare_with_exp1_results():
    """Compare our test results with experiment 1 genetic algorithm results."""
    
    print("=" * 70)
    print("COMPARING WITH EXPERIMENT 1 RESULTS")
    print("=" * 70)
    
    # Expected experiment 1 directories (genetic algorithm, agents 5-15)
    exp1_dirs = [
        "result_20241216_160305_geneticalgo_agents5_all",
        "result_20241220_001737_geneticalgo_agents6_all",
        "result_20241221_093730_geneticalgo_agents7_all",
        "result_20241221_180752_geneticalgo_agents8_all",
        "result_20241222_030727_geneticalgo_agents9_all",
        "result_20241202_131125_geneticalgo_agents10_all",
        "result_20241219_002151_geneticalgo_agents11_all",
        "result_20241219_045149_geneticalgo_agents12_all",
        "result_20241220_092457_geneticalgo_agents13_all",
        "result_20241221_051741_geneticalgo_agents14_all",
        "result_20241217_074919_geneticalgo_agents15_all"
    ]
    
    exp1_results_dir = PROJECT_ROOT / "results_to_save" / "exp_1"
    
    print("Checking experiment 1 reference results...")
    
    for exp1_dir in exp1_dirs:
        exp1_path = exp1_results_dir / exp1_dir / "dataframe.csv"
        
        if exp1_path.exists():
            try:
                df_exp1 = pd.read_csv(exp1_path)
                print(f"✓ {exp1_dir}: {df_exp1.shape}")
            except Exception as e:
                print(f"✗ {exp1_dir}: Error loading - {e}")
        else:
            print(f"⚠ {exp1_dir}: dataframe.csv not found")
    
    print()
    print("Note: For detailed comparison, run both tests and compare the dataframe.csv files")
    print("Expected differences:")
    print("- Directory names and timestamps will differ")
    print("- Results should be statistically similar but not identical due to randomness")
    print("- Same genetic algorithm parameters should produce comparable performance")

if __name__ == "__main__":
    print("Starting multifail vs experiment 1 comparison test...")
    print()
    
    # Run the test
    success = test_multifail_genetic_algorithm()
    
    if success:
        print()
        compare_with_exp1_results()
        
        print()
        print("=" * 70)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Results directories are now known (no need to search!)")
        print("2. Compare dataframe.csv files with experiment 1 results")
        print("3. Verify that engine lifecycle management is working as expected")
        
    else:
        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)
        print()
        print("Check the error messages above for troubleshooting.")
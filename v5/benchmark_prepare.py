import argparse
import json
import os
import subprocess
import itertools
import time
import csv
import tempfile
import re

# This dictionary will be used for the grid search.
# The user can fill in the parameters and their possible values.
# Example:
# GRID_SEARCH_PARAMS = {
#     "max_hops": [2, 3, 4],
#     "decay_factor": [0.1, 0.2],
# }
GRID_SEARCH_PARAMS = {
    # Please fill in the parameters for the grid search here
    "max_hops": [2, 3, 4, 5],
    "decay_factor": [0.1, 0.2, 0.4, 0.5, 0.7, 0.8],
    "max_fanout": [50, 100, 150, 200]
}

def run_command(command):
    """Runs a command and returns its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise

def parse_time_output(stderr):
    """Parses the output of /usr/bin/time -l to get time and memory."""
    real_time_match = re.search(r"(\d+\.\d+)\s+real", stderr)
    max_memory_match = re.search(r"(\d+)\s+maximum resident set size", stderr)

    real_time = float(real_time_match.group(1)) if real_time_match else None
    # Convert from bytes to MB
    max_memory_mb = (int(max_memory_match.group(1)) / (1024 * 1024)) if max_memory_match else None

    return real_time, max_memory_mb

def main():
    parser = argparse.ArgumentParser(description="Benchmark the prepare stage.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("--num_threads", type=int, required=True, help="Number of threads.")
    parser.add_argument("--config", required=True, help="Path to the base configuration file.")
    parser.add_argument("--output_csv", default="benchmark_results.csv", help="Path to save the benchmark results CSV file.")
    args = parser.parse_args()

    # Load the base configuration
    with open(args.config, 'r') as f:
        base_config = json.load(f)

    # Generate all combinations of parameters for the grid search
    param_names = list(GRID_SEARCH_PARAMS.keys())
    param_values = list(GRID_SEARCH_PARAMS.values())
    param_combinations = list(itertools.product(*param_values))

    results = []

    # Run the data stage once before starting the benchmark loop
    print("--- Running initial 'data' stage ---")
    initial_config = base_config.copy()
    initial_config["dataset"] = args.dataset
    initial_config["num_threads"] = args.num_threads
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", dir=".") as temp_f:
        temp_config_path = temp_f.name
        json.dump(initial_config, temp_f, indent=2)

    try:
        run_command(f"./run.sh data {temp_config_path}")
        print("--- 'data' stage completed successfully ---\n")
    finally:
        os.remove(temp_config_path)


    for i, combo in enumerate(param_combinations):
        print(f"--- Running benchmark for combination {i+1}/{len(param_combinations)} ---")
        
        # Create a new config for this combination
        current_config = base_config.copy()
        current_config["dataset"] = args.dataset
        current_config["num_threads"] = args.num_threads

        combo_dict = {}
        for name, value in zip(param_names, combo):
            current_config[name] = value
            combo_dict[name] = value
            print(f"  {name}: {value}")

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", dir=".") as temp_f:
            temp_config_path = temp_f.name
            json.dump(current_config, temp_f, indent=2)
        
        try:
            # Benchmark the 'prepare' stage
            print(f"  Benchmarking 'prepare' stage with config: {temp_config_path}")
            
            # Using /usr/bin/time -l to measure time and memory on macOS
            benchmark_command = f"/usr/bin/time -l ./run.sh prepare {temp_config_path}"
            
            _, stderr = run_command(benchmark_command)
            
            # Parse the benchmark results
            total_time, max_memory = parse_time_output(stderr)
            
            print(f"  Time: {total_time:.2f}s, Memory: {max_memory:.2f} MB")

            # Store results
            result_row = combo_dict.copy()
            result_row["time_seconds"] = total_time
            result_row["memory_mb"] = max_memory
            results.append(result_row)

        finally:
            # Clean up the temporary config file
            os.remove(temp_config_path)
        
        print("-" * (len(str(i+1)) + len(str(len(param_combinations))) + 35) + "\n")


    # Write results to CSV
    if results:
        header = list(param_names) + ["time_seconds", "memory_mb"]
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"Benchmark results saved to {args.output_csv}")
    else:
        print("No benchmarks were run. Is GRID_SEARCH_PARAMS empty?")

if __name__ == "__main__":
    main()

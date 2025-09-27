import argparse
import json
import os
import subprocess
import itertools
import time
import csv
import tempfile
import re
import shutil
import platform

import psutil

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
    """Parses the timing output from either BSD/macOS (/usr/bin/time -l) or GNU time/time builtin."""

    real_time = None
    max_memory_mb = None

    # Handle `time -p` (real <seconds>)
    match_seconds = re.search(r"real\s+(\d+\.\d+)", stderr)
    if match_seconds:
        real_time = float(match_seconds.group(1))

    # Handle default POSIX format (real 0m0.123s)
    if real_time is None:
        match_min_sec = re.search(r"real\s+(\d+)m(\d+\.\d+)s", stderr)
        if match_min_sec:
            minutes = int(match_min_sec.group(1))
            seconds = float(match_min_sec.group(2))
            real_time = minutes * 60 + seconds

    # Handle macOS /usr/bin/time -l format (<seconds> real)
    if real_time is None:
        match_real_prefix = re.search(r"(\d+\.\d+)\s+real", stderr)
        if match_real_prefix:
            real_time = float(match_real_prefix.group(1))

    # Memory parsing for both BSD (-l) and GNU (-v)
    # macOS: "123456  maximum resident set size" (bytes)
    match_bsd_mem = re.search(r"(\d+)\s+maximum resident set size", stderr, re.IGNORECASE)
    if match_bsd_mem:
        max_memory_mb = int(match_bsd_mem.group(1)) / (1024 * 1024)

    # GNU time -v: "Maximum resident set size (kbytes): 1234"
    match_gnu_mem = re.search(r"maximum resident set size \(kbytes\):\s*(\d+)", stderr, re.IGNORECASE)
    if match_gnu_mem:
        max_memory_mb = int(match_gnu_mem.group(1)) / 1024

    return real_time, max_memory_mb


def run_command_with_monitor(command, poll_interval=0.1):
    """Runs a command and tracks elapsed time and memory usage via psutil."""

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    max_rss_bytes = 0
    start_time = time.perf_counter()

    try:
        ps_proc = psutil.Process(process.pid)
    except psutil.Error:
        ps_proc = None

    try:
        while True:
            if ps_proc is not None:
                try:
                    rss = ps_proc.memory_info().rss
                    for child in ps_proc.children(recursive=True):
                        rss += child.memory_info().rss
                    if rss > max_rss_bytes:
                        max_rss_bytes = rss
                except psutil.Error:
                    ps_proc = None

            retcode = process.poll()
            if retcode is not None:
                break

            time.sleep(poll_interval)

        stdout, stderr = process.communicate()
    except Exception:
        process.kill()
        stdout, stderr = process.communicate()
        raise

    elapsed = time.perf_counter() - start_time

    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
        raise subprocess.CalledProcessError(process.returncode, command)

    max_memory_mb = max_rss_bytes / (1024 * 1024) if max_rss_bytes else None

    return stdout, stderr, elapsed, max_memory_mb

def main():
    parser = argparse.ArgumentParser(description="Benchmark the prepare stage.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("--num_threads", type=int, required=True, help="Number of threads.")
    parser.add_argument("--config", required=True, help="Path to the base configuration file.")
    parser.add_argument("--output_dir", default="benchmark_results.csv", help="Prefix or path for the benchmark results CSV file.")
    args = parser.parse_args()

    # Load the base configuration
    with open(args.config, 'r') as f:
        base_config = json.load(f)

    output_prefix = args.output_dir
    output_dir = os.path.dirname(output_prefix)
    output_base = os.path.basename(output_prefix)
    if not output_base:
        output_base = "benchmark_results"
    base_root, base_ext = os.path.splitext(output_base)
    if base_ext.lower() != ".csv":
        base_root = output_base
        base_ext = ".csv"

    final_output_filename = f"{base_root}_{args.dataset}{base_ext}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_dir_path = os.path.join(output_dir, final_output_filename)
    else:
        output_dir_path = final_output_filename

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

            time_binary = shutil.which("/usr/bin/time")
            if time_binary:
                if platform.system() == "Darwin":
                    benchmark_command = f"{time_binary} -l ./run.sh prepare {temp_config_path}"
                else:
                    benchmark_command = f"{time_binary} -v ./run.sh prepare {temp_config_path}"

                _, stderr = run_command(benchmark_command)
                total_time, max_memory = parse_time_output(stderr)
            else:
                # Fallback: measure time and memory within Python when external time binary is unavailable
                _, stderr, total_time, max_memory = run_command_with_monitor(
                    f"./run.sh prepare {temp_config_path}"
                )

            if total_time is not None:
                time_msg = f"{total_time:.2f}s"
            else:
                time_msg = "N/A"

            if max_memory is not None:
                memory_msg = f"{max_memory:.2f} MB"
            else:
                memory_msg = "N/A"

            coverage_ratio = None
            coverage_summary_path = os.path.join(
                current_config.get("storage_dir", "."),
                f"{current_config['dataset']}_prepare_summary.json"
            )
            summary_paths_found = None
            summary_edges_evaluated = None
            if os.path.exists(coverage_summary_path):
                try:
                    with open(coverage_summary_path, "r") as summary_file:
                        summary_payload = json.load(summary_file)
                        coverage_ratio = summary_payload.get("coverage_ratio")
                        summary_paths_found = summary_payload.get("paths_found")
                        summary_edges_evaluated = summary_payload.get("edges_evaluated")
                except (json.JSONDecodeError, OSError) as exc:
                    print(
                        f"  Warning: Failed to parse coverage summary at {coverage_summary_path}: {exc}"
                    )
            else:
                print(
                    f"  Warning: Coverage summary file not found at {coverage_summary_path}"
                )

            if coverage_ratio is not None:
                coverage_msg = f"{coverage_ratio * 100:.2f}%"
            else:
                coverage_msg = "N/A"

            print(
                f"  Time: {time_msg}, Memory: {memory_msg}, Coverage: {coverage_msg}"
            )

            if (
                coverage_ratio is not None
                and summary_paths_found is not None
                and summary_edges_evaluated is not None
            ):
                print(
                    f"    Coverage detail: {summary_paths_found}/{summary_edges_evaluated} edges"
                )

            # Store results
            result_row = combo_dict.copy()
            result_row["time_seconds"] = total_time
            result_row["memory_mb"] = max_memory
            result_row["coverage_ratio"] = coverage_ratio
            results.append(result_row)

        finally:
            # Clean up the temporary config file
            os.remove(temp_config_path)
        
        print("-" * (len(str(i+1)) + len(str(len(param_combinations))) + 35) + "\n")


    # Write results to CSV
    if results:
        header = list(param_names) + ["time_seconds", "memory_mb", "coverage_ratio"]
        with open(output_dir_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"Benchmark results saved to {output_dir_path}")
    else:
        print("No benchmarks were run. Is GRID_SEARCH_PARAMS empty?")

if __name__ == "__main__":
    main()

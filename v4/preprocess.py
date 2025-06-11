import argparse
import json
import os
import subprocess
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import sys

def load_configuration(config_path: str) -> dict:
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[Error] Invalid JSON in configuration file: {config_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to load configuration from {config_path}: {e}", file=sys.stderr)
        sys.exit(1)

def run_preprocess_thread(args_tuple: tuple) -> dict:
    """
    Worker function to run a single preprocess.cpp instance.
    Captures its stdout (JSON) and returns it as a Python dictionary.
    """
    (binary_path, dataset_name, partition, model_name, 
     beam_width, storage_dir, total_num_threads_for_cpp, current_thread_id_for_cpp) = args_tuple

    cmd = [
        binary_path,
        dataset_name,
        partition,
        model_name,
        str(beam_width),
        storage_dir,
        str(total_num_threads_for_cpp),
        str(current_thread_id_for_cpp)
    ]
    
    process_results = {}
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout_data, stderr_data = proc.communicate()

        # Log C++ stderr content if any
        if stderr_data:
            # Prepend thread ID to each line of stderr for clarity
            stderr_lines = stderr_data.strip().split('\n')
            for line in stderr_lines:
                print(f"[C++ stderr T-{current_thread_id_for_cpp}] {line}", file=sys.stderr)
        
        if proc.returncode != 0:
            print(f"[Error] Python orchestrator: Thread {current_thread_id_for_cpp} C++ process exited with code {proc.returncode}.", file=sys.stderr)
            return {} 

        if stdout_data:
            process_results = json.loads(stdout_data)
        else:
            # This might be normal if a thread has no queries assigned or finds no paths.
            # The C++ process should still exit with 0.
            print(f"[Info] Python orchestrator: Thread {current_thread_id_for_cpp} C++ process produced no stdout data (JSON expected). This might be okay if no results for this shard.", file=sys.stderr)
            return {}

    except FileNotFoundError:
        print(f"[Error] Python orchestrator: Thread {current_thread_id_for_cpp} C++ binary not found at {binary_path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"[Error] Python orchestrator: Thread {current_thread_id_for_cpp} Failed to decode JSON from C++ process stdout. Error: {e}", file=sys.stderr)
        # print(f"Problematic stdout from C++ T-{current_thread_id_for_cpp}:\n{stdout_data[:500]}...", file=sys.stderr) # Print snippet
        return {}
    except Exception as e:
        print(f"[Error] Python orchestrator: Thread {current_thread_id_for_cpp} An unexpected error occurred: {e}", file=sys.stderr)
        return {}
        
    return process_results

def main():
    parser = argparse.ArgumentParser(description="Python orchestrator for parallel C++ preprocess execution.")
    parser.add_argument('--config', required=True, help="Path to the main configuration JSON file.")
    parser.add_argument('--binary', required=True, help="Path to the compiled C++ preprocess binary.")
    parser.add_argument('--partition', required=True, choices=['train', 'val', 'test'], help="Data partition to process (e.g., train, val, test).")
    parser.add_argument('--sampling', type=int, default=None, help="Number of C++ threads (data shards) to process. If set, processes shards 0 to sampling-1. Defaults to num_threads from config.")
    
    cli_args = parser.parse_args()

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        print("[Warning] Python orchestrator: Could not set multiprocessing start method to 'spawn'. This might lead to issues if CUDA is involved in C++ or related libraries.", file=sys.stderr)

    config = load_configuration(cli_args.config)

    dataset_name = config.get('dataset')
    storage_dir = config.get('storage_dir', '.')
    # num_threads_config is the total sharding factor the C++ code should be aware of.
    num_threads_config = config.get('num_threads', 1) 
    model_name = config.get('model_name') 
    # beam_width for C++ preprocess, can be num_neg from legacy configs or a specific beam_width field.
    beam_width = config.get('beam_width', config.get('num_neg', 20)) 

    if not dataset_name:
        print("[Error] Python orchestrator: 'dataset' not found in config.", file=sys.stderr)
        sys.exit(1)
    if not model_name:
        print("[Error] Python orchestrator: 'model_name' not found in config. This is required for the C++ preprocess binary.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(cli_args.binary):
        print(f"[Error] Python orchestrator: C++ binary not found at {cli_args.binary}", file=sys.stderr)
        sys.exit(1)


    # Determine the number of C++ instances (tasks) to run.
    # If sampling is set, it limits the number of shards processed.
    # Otherwise, all num_threads_config shards are processed.
    tasks_to_run_count = cli_args.sampling if cli_args.sampling is not None else num_threads_config
    tasks_to_run_count = min(tasks_to_run_count, num_threads_config) # Cannot run more tasks than total shards

    # The pool size determines how many C++ processes run concurrently.
    # It should ideally be <= tasks_to_run_count and also consider system resources.
    # For simplicity, let's use tasks_to_run_count as pool_size, or a system-sensible default like os.cpu_count().
    # However, to match `prepare.py`'s behavior where `run_threads` (sampling) controls pool size:
    pool_size = tasks_to_run_count 
    # If tasks_to_run_count is very large, user might want a smaller concurrent pool.
    # For now, let pool_size be the number of tasks we intend to run.
    # A more robust approach might cap pool_size at os.cpu_count().
    # pool_size = min(tasks_to_run_count, os.cpu_count() or 1) # Example of capping

    print(f"[Info] Python orchestrator: Dataset: {dataset_name}, Partition: {cli_args.partition}, Model: {model_name}", file=sys.stderr)
    print(f"[Info] Python orchestrator: Total conceptual data shards (num_threads for C++): {num_threads_config}", file=sys.stderr)
    print(f"[Info] Python orchestrator: Number of shards to process (C++ instances to run): {tasks_to_run_count}", file=sys.stderr)
    print(f"[Info] Python orchestrator: Concurrent C++ processes (Python Pool size): {pool_size}", file=sys.stderr)

    tasks_args_list = []
    for i in range(tasks_to_run_count):
        current_task_thread_id = i # C++ thread_id will be 0, 1, ..., tasks_to_run_count-1
        # Each C++ instance is told the total sharding factor is num_threads_config
        task_args = (
            cli_args.binary,
            dataset_name,
            cli_args.partition,
            model_name,
            beam_width,
            storage_dir,
            num_threads_config,      # Total sharding factor C++ is aware of
            current_task_thread_id   # Specific shard this C++ instance handles
        )
        tasks_args_list.append(task_args)

    final_aggregated_results = {}
    
    if not tasks_args_list:
        print("[Warning] Python orchestrator: No tasks to run based on configuration and sampling.", file=sys.stderr)
    else:
        print(f"[Info] Python orchestrator: Starting parallel execution of {len(tasks_args_list)} C++ preprocess tasks using a pool of {pool_size} workers...", file=sys.stderr)
        with Pool(processes=pool_size) as pool:
            for thread_result_dict in tqdm(pool.imap_unordered(run_preprocess_thread, tasks_args_list), total=len(tasks_args_list), desc="Running C++ preprocess"):
                if thread_result_dict: 
                    final_aggregated_results.update(thread_result_dict)
    
    print(f"[Info] Python orchestrator: Aggregation complete. Total unique edge IDs in final results: {len(final_aggregated_results)}", file=sys.stderr)

    output_filename = f"{model_name}_{dataset_name}_{cli_args.partition}_neg.json"
    output_filepath = os.path.join(storage_dir, output_filename)

    print(f"[Info] Python orchestrator: Writing aggregated results to: {output_filepath}", file=sys.stderr)
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_aggregated_results, f, indent=2)
        print(f"[Info] Python orchestrator: Successfully saved results to {output_filepath}", file=sys.stderr)
    except IOError as e:
        print(f"[Error] Python orchestrator: Failed to write results to {output_filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeError as e: 
        print(f"[Error] Python orchestrator: Failed to serialize results to JSON: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
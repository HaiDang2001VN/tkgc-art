import os
import subprocess
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from utils import load_configuration

def run_thread(args):
    """
    Invoke the C++ worker and parse its tab-delimited output.
    Returns a dict mapping edge_id to the path info.
    """
    binary, csv, max_hops, tid, nthreads = args
    cmd = [binary, csv, str(max_hops), str(tid), str(nthreads)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    results = {}
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        
        # parts: [edge_id, hops, nodes, node_types, edge_types]
        eid = parts[0]
        hops = int(parts[1])
        # parse nodes as list of ids separated by comma
        nodes = []
        if parts[2]:
            nodes = [int(x) for x in parts[2].split(",")]
        # parse node_types as list of ints
        node_types = [int(x) for x in parts[3].split(",")] if parts[3] else []
        # parse edge_types as list of strings
        edge_types = parts[4].split(",") if parts[4] else []

        results[eid] = {
            "hops": hops,
            "nodes": nodes,
            "node_types": node_types,
            "edge_types": edge_types
        }
    proc.wait()
    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel pathfinder manager")
    parser.add_argument('--config',        required=True,
                        help="Path to config.json file")
    parser.add_argument('--binary',     required=True,
                        help="Compiled C++ worker binary")
    parser.add_argument('--sampling', type=int, default=4,
                        help="Number of actual threads to run (sampling)")
    args = parser.parse_args()
    
    config = load_configuration(args.config)
    data_dir = config.get('storage_dir', '.')
    num_threads = config.get('num_threads', 1)
    csv_path = os.path.join(data_dir, f"{config['dataset']}_edges.csv")
    max_hops = config.get('max_hops', 4)

    # Prepare per-thread invocation args
    run_threads = min(num_threads, args.sampling)
    tasks = [
        (args.binary, csv_path, max_hops, tid, num_threads)
        for tid in range(run_threads)
    ]

    # Run in parallel
    with Pool(run_threads) as pool:
        thread_maps = pool.map(run_thread, tasks)

    # Collate all results
    final_map = {}
    for tm in thread_maps:
        final_map.update(tm)

    # Write to paths.json
    # os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(data_dir, f"{config['dataset']}_paths.json")
    with open(out_file, 'w') as f:
        import json
        json.dump(final_map, f, indent=2)

    print(f"Saved combined paths to {out_file}")


if __name__ == '__main__':
    main()

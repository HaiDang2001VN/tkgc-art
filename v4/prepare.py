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


def write_paths_text_file(final_map, output_path):
    """
    Write shortest paths to text file in the specified format:
    - First line: number of shortest paths
    - For each path: eid, hops, neighbors line, node types line, meta-pattern line
    """
    with open(output_path, 'w') as f:
        # Write number of shortest paths
        f.write(f"{len(final_map)}\n")

        # Write each shortest path block
        for eid, path_info in final_map.items():
            # Write edge ID
            f.write(f"{eid}\n")

            # Write number of hops
            f.write(f"{path_info['hops']}\n")

            # Write neighbors (nodes) on single line, space-separated
            nodes_line = " ".join(map(str, path_info['nodes']))
            f.write(f"{nodes_line}\n")

            # Write node types on single line, space-separated
            node_types_line = " ".join(map(str, path_info['node_types']))
            f.write(f"{node_types_line}\n")

            # Write meta-pattern (edge types) on single line, space-separated
            edge_types_line = " ".join(map(str, path_info['edge_types']))
            f.write(f"{edge_types_line}\n")


def main():
    parser = argparse.ArgumentParser(description="Parallel pathfinder manager")
    parser.add_argument('--config', required=True,
                        help="Path to config.json file")
    parser.add_argument('--binary', required=True,
                        help="Compiled C++ worker binary")
    parser.add_argument('--sampling', type=int, default=None,
                        help="Number of actual threads to run (sampling)")
    args = parser.parse_args()

    config = load_configuration(args.config)
    data_dir = config.get('storage_dir', '.')
    num_threads = config.get('num_threads', 1)
    csv_path = os.path.join(data_dir, f"{config['dataset']}_edges.csv")
    max_hops = config.get('max_hops', 4)

    # Prepare per-thread invocation args
    run_threads = min(
        num_threads, args.sampling) if args.sampling is not None else num_threads
    tasks = [
        (args.binary, csv_path, max_hops, tid, num_threads)
        for tid in range(run_threads)
    ]

    print(
        f"Starting parallel shortest path computation with {run_threads} threads...")
    print(f"Processing CSV: {csv_path}")
    print(f"Max hops: {max_hops}")

    # Run in parallel
    with Pool(run_threads) as pool:
        thread_maps = list(tqdm(
            pool.imap(run_thread, tasks),
            total=len(tasks),
            desc="Processing threads"
        ))

    # Collate all results
    final_map = {}
    total_paths = 0
    for tm in thread_maps:
        final_map.update(tm)
        total_paths += len(tm)

    print(f"Collected {total_paths} shortest paths from all threads")

    # Write to paths.txt instead of JSON
    out_file = os.path.join(data_dir, f"{config['dataset']}_paths.txt")

    print(f"Writing shortest paths to text file: {out_file}")
    write_paths_text_file(final_map, out_file)

    print(f"Successfully saved {len(final_map)} shortest paths to {out_file}")

    # Print sample of the output format for verification
    if final_map:
        print("\nSample output format:")
        print(f"Total paths: {len(final_map)}")
        sample_eid = next(iter(final_map))
        sample_path = final_map[sample_eid]
        print(f"Sample path for edge {sample_eid}:")
        print(f"  Hops: {sample_path['hops']}")
        print(f"  Nodes: {' '.join(map(str, sample_path['nodes']))}")
        print(f"  Edge types: {' '.join(map(str, sample_path['edge_types']))}")


if __name__ == '__main__':
    main()

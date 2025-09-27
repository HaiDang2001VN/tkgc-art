import os
import subprocess
import argparse
import json
from tqdm import tqdm
from multiprocessing import Pool
from utils import load_configuration


def run_thread(args):
    """
    Invoke the C++ worker and parse its tab-delimited output.
    Returns a dict mapping edge_id to the path info.
    """
    binary, csv, max_hops, tid, nthreads, decay_factor, max_fanout = args
    cmd = [binary, csv, str(max_hops), str(tid), str(nthreads), str(decay_factor), str(max_fanout)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    results = {}
    summary_paths_found = 0
    summary_edges_total = 0

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        if line.rfind("SUMMARY|", 0) == 0:
            parts = line.split("|")
            if len(parts) >= 4:
                try:
                    summary_paths_found = int(parts[2])
                    summary_edges_total = int(parts[3])
                except ValueError:
                    pass
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

        # parse edge_timestamps as list of strings
        edge_timestamps = parts[5].split(",") if parts[5] else []

        results[eid] = {
            "hops": hops,
            "nodes": nodes,
            "node_types": node_types,
            "edge_types": edge_types,
            "edge_timestamps": edge_timestamps
        }

    proc.wait()

    if summary_paths_found == 0 and results:
        summary_paths_found = len(results)
    if summary_edges_total == 0:
        summary_edges_total = summary_paths_found

    return tid, results, summary_paths_found, summary_edges_total


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

            # Write meta-pattern (edge timestamps) on single line, space-separated
            edge_timestamps_line = " ".join(map(str, path_info['edge_timestamps']))
            f.write(f"{edge_timestamps_line}\n")


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
    decay_factor = config.get('decay_factor', 1.0)
    max_fanout = config.get('max_fanout', 50)

    # Prepare per-thread invocation args
    run_threads = min(
        num_threads, args.sampling) if args.sampling is not None else num_threads
    tasks = [
        (args.binary, csv_path, max_hops, tid, num_threads, decay_factor, max_fanout)
        for tid in range(run_threads)
    ]

    print(
        f"Starting parallel shortest path computation with {run_threads} threads...")
    print(f"Processing CSV: {csv_path}")
    print(f"Max hops: {max_hops}")
    print(f"Decay Factor: {decay_factor}")
    print(f"Max Fanout: {max_fanout}")

    # Run in parallel
    with Pool(run_threads) as pool:
        thread_results = list(tqdm(
            pool.imap(run_thread, tasks),
            total=len(tasks),
            desc="Processing threads"
        ))

    # Collate all results
    final_map = {}
    total_paths = 0
    total_evaluated_edges = 0
    per_thread_logs = []

    for idx, result in enumerate(thread_results):
        tid, thread_map, paths_found, edges_total = result

        final_map.update(thread_map)
        total_paths += len(thread_map)

        if paths_found == 0 and thread_map:
            paths_found = len(thread_map)
        if edges_total == 0:
            edges_total = max(paths_found, len(thread_map))

        total_evaluated_edges += edges_total

        per_thread_logs.append({
            "thread_id": tid,
            "paths_found": paths_found,
            "edges_evaluated": edges_total,
            "coverage_ratio": (paths_found / edges_total) if edges_total else 0.0,
        })

    print(f"Collected {total_paths} shortest paths from all threads")

    # Write to paths.txt instead of JSON
    out_file = os.path.join(data_dir, f"{config['dataset']}_paths.txt")

    print(f"Writing shortest paths to text file: {out_file}")
    write_paths_text_file(final_map, out_file)

    print(f"Successfully saved {len(final_map)} shortest paths to {out_file}")

    paths_found_total = sum(entry["paths_found"] for entry in per_thread_logs)
    total_edges_evaluated = total_evaluated_edges
    coverage_ratio = (paths_found_total / total_edges_evaluated) if total_edges_evaluated else 0.0

    summary_payload = {
        "dataset": config["dataset"],
        "paths_found": paths_found_total,
        "edges_evaluated": total_edges_evaluated,
        "coverage_ratio": coverage_ratio,
        "per_thread": per_thread_logs,
    }

    summary_json_path = os.path.join(data_dir, f"{config['dataset']}_prepare_summary.json")
    with open(summary_json_path, "w") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    summary_log_path = os.path.join(data_dir, f"{config['dataset']}_prepare_summary.log")
    with open(summary_log_path, "w") as summary_log:
        summary_log.write("ThreadID\tPathsFound\tEdgesEvaluated\tCoverageRatio\n")
        for entry in per_thread_logs:
            summary_log.write(
                f"{entry['thread_id']}\t{entry['paths_found']}\t{entry['edges_evaluated']}\t{entry['coverage_ratio']:.4f}\n"
            )
        summary_log.write(
            f"TOTAL\t{paths_found_total}\t{total_edges_evaluated}\t{coverage_ratio:.4f}\n"
        )

    print(
        f"Coverage summary: {paths_found_total}/{total_edges_evaluated} "
        f"({coverage_ratio * 100:.2f}%); saved to {summary_json_path}"
    )

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

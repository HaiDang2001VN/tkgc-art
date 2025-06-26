#!/usr/bin/env python3
"""data.py
~~~~~~~~~~
Standalone loader & exporter for *temporal knowledge‑graph* benchmarks that
follow the **four‑file layout**:

    train.txt   subject  relation  object  timestamp  \t‑separated integers
    valid.txt   "
    test.txt    "
    stat.txt    single line:  <#entities> <#relations> <#timestamps>

Supported datasets include ICEWS14/18/05‑15, GDELT (pre‑processed), and the
YAGO15K temporal slice.  The script builds the edge list expected by our
pipeline (same schema as the OGB/TGB path) and persists mapping dictionaries
for nodes & relations.

Usage
-----
    python data.py --config path/to/config.json

The *config.json* must minimally contain::

    {
      "dataset": "icews14",            # folder name under storage_dir
      "storage_dir": "./data",         # optional, default: '.'
      "train_ratio": 0.1,              # fraction of earliest train edges
                                       #     to mark as the 'pre' split
      "neg_per_pos": 1                 # (optional) negatives per positive
    }

Outputs (all under ``storage_dir``):
    <dataset>_edges.csv         – edge list with split/label columns
    <dataset>_node_map.pt       – torch‑saved dict raw‑node‑id → contiguous id
    <dataset>_edge_type_map.pt  – relation‑id → contiguous id

Node features are not available in these benchmarks; an empty dict is saved
for API compatibility.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple, List
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# project helper -------------------------------------------------------------
from utils import load_configuration

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse four‑file temporal‑KG dataset and export CSV/pt artefacts",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON configuration file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _save_mapping(path: Path, mapping: Dict):
    torch.save(mapping, path)
    print(f"Saved {path.name} (size={len(mapping)})")


def save_edges(configuration: Dict, node_map: Dict, edge_type_map: Dict):
    """Persist mapping dictionaries next to the dataset directory."""
    output_dir = Path(configuration.get("storage_dir", "."))
    dataset = configuration["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_mapping(output_dir / f"{dataset}_node_map.pt",      node_map)
    _save_mapping(output_dir / f"{dataset}_edge_type_map.pt", edge_type_map)


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def map_position_to_node_id(position: int, sorted_neighbors: list, n_entities: int) -> int:
    """
    Map a position in the non-neighbor space to an actual node ID.
    
    Args:
        position: Integer position in [0, n_entities - len(neighbors) - 1]
        sorted_neighbors: Sorted list of neighbor node IDs to skip
        n_entities: Total number of entities
        
    Returns:
        The node ID at the given position in the non-neighbor space
    """
    current_pos = 0
    node_id = 0
    
    for neighbor in sorted_neighbors:
        # Calculate gap size before this neighbor
        gap_size = neighbor - node_id
        
        if current_pos + gap_size > position:
            # The target position is in this gap
            return node_id + (position - current_pos)
        
        # Skip over this gap and the neighbor
        current_pos += gap_size
        node_id = neighbor + 1
    
    # If we reach here, the position is in the final gap after all neighbors
    return node_id + (position - current_pos)


def generate_negatives_for_edge(args):
    """Helper function for multiprocessing negative sampling"""
    head, tail, rel, ts, split_name, head_neighbors, all_entities, neg_per_pos, seed = args
    
    # Create a local random generator with a unique seed derived from the edge and global seed
    local_seed = int(head * 10000 + tail * 100 + seed)
    rng = np.random.default_rng(seed=local_seed)
    
    # Get neighbors for this head node (including self to avoid self-loops)
    head_neighbors_with_self = head_neighbors | {head}
    
    # Calculate number of non-neighbors
    num_non_neighbors = len(all_entities) - len(head_neighbors_with_self)
    
    if num_non_neighbors == 0:
        return []  # Skip if no non-neighbors available
    
    # Sample positions in the non-neighbor space
    sample_size = min(num_non_neighbors, neg_per_pos)
    positions = rng.choice(num_non_neighbors, size=sample_size, replace=False)
    
    # Convert neighbor set to sorted list for efficient gap traversal
    sorted_neighbors = sorted(head_neighbors_with_self)
    
    # Map positions to actual node IDs and generate negatives
    negatives = []
    for pos in positions:
        v_neg = map_position_to_node_id(pos, sorted_neighbors, len(all_entities))
        
        # Safety check (should not be needed with correct implementation)
        if v_neg not in head_neighbors_with_self:
            negatives.append((head, int(v_neg), rel, ts, split_name))
    
    return negatives


def process_quad_dataset(configuration: Dict) -> pd.DataFrame:
    """Read train/valid/test quadruple files and return a DataFrame.

    The resulting columns match the edge schema used elsewhere in the codebase:
        feat_pos_u | feat_pos_v | u | v | u_type | v_type | ts | split | label | edge_type
    """

    ds_name = configuration["dataset"]
    root_dir = Path(configuration.get("storage_dir", ".")) / ds_name.upper()

    # validate presence of required files ------------------------------------------------
    required = [root_dir /
                f for f in ("train.txt", "valid.txt", "test.txt", "stat.txt")]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is incomplete; missing: " + ", ".join(missing))

    # read entity universe size (for negative sampling) ----------------------------------
    n_ent, n_rel, _ = map(int, (root_dir / "stat.txt").read_text().split())
    all_entities = np.arange(n_ent, dtype=np.int64)

    # configuration knobs ----------------------------------------------------------------
    train_ratio = float(configuration.get(
        "train_ratio", 0.0))  # 0.0 → all 'train'
    neg_per_pos = int(configuration.get("neg_per_pos", 1))
    split_code = {"pre": 0, "train": 1, "valid": 2, "test": 3}

    # Determine the number of workers for multiprocessing
    num_workers = configuration.get("num_threads", None)
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(num_workers, mp.cpu_count())

    # mapping dicts ----------------------------------------------------------------------
    node_map: Dict[int, int] = {}  # unified mapping for both u and v
    edge_type_map: Dict[int, int] = {}
    neighbor_dict: Dict[int, set] = {}  # node -> set of all neighbors across all time
    
    records = []

    def _id_assign(mapping: Dict[int, int], raw_id: int) -> int:
        if raw_id not in mapping:
            mapping[raw_id] = len(mapping)
        return mapping[raw_id]

    # helper to append an edge record ----------------------------------------------------
    def _append(u_raw: int, v_raw: int, r_raw: int, ts_val: int, split: str, label: int):
        u = _id_assign(node_map, u_raw)
        v = _id_assign(node_map, v_raw)
        e = _id_assign(edge_type_map, r_raw)

        records.append({
            "feat_pos_u": u_raw,
            "feat_pos_v": v_raw,
            "u": u,
            "v": v,
            "u_type": 0,          # mono‑typed graphs
            "v_type": 0,
            "ts": ts_val,
            "split": split_code.get(split, split),  # Convert to int code if possible
            "label": label,
            "edge_type": e,
        })
        
        # Update neighbor dictionary for positive edges only
        if label == 1:
            if u_raw not in neighbor_dict:
                neighbor_dict[u_raw] = set()
            if v_raw not in neighbor_dict:
                neighbor_dict[v_raw] = set()
            neighbor_dict[u_raw].add(v_raw)
            neighbor_dict[v_raw].add(u_raw)  # undirected graph assumption

    # ----------------------------------------------------------------------
    # iterate over splits - first pass: collect all positive edges
    # ----------------------------------------------------------------------
    positive_edges_by_split = {}  # split -> list of positive edge tuples
    
    for split in ("train", "valid", "test"):
        fpath = root_dir / f"{split}.txt"
        df = pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            names=["head", "rel", "tail", "ts", "label"],
            dtype={"head": np.int64, "rel": np.int64,
                   "tail": np.int64, "ts": np.int64, "label": np.int64},
        )

        # determine chronological cutoff for 'pre' split (only for train) ---------------
        if split == "train" and train_ratio > 0:
            threshold = np.quantile(df["ts"], train_ratio)
        else:
            threshold = None

        # Store positive edges for this split
        positive_edges_by_split[split] = []
        
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{split} pos"):
            split_label = (
                "pre" if split == "train" and threshold is not None and row.ts < threshold else split
            )
            _append(row.head, row.tail, row.rel, int(row.ts), split_label, 1)
            
            # Store for later negative sampling
            if split in ("valid", "test"):
                positive_edges_by_split[split].append((row.head, row.tail, row.rel, int(row.ts), split))

    print(f"Collected neighbor information from {len(neighbor_dict)} nodes")
    
    # ----------------------------------------------------------------------
    # second pass: generate negatives using global neighbor information
    # ----------------------------------------------------------------------
    if neg_per_pos > 0:
        seed = 42  # Global random seed
        
        for split in ("valid", "test"):
            print(f"Generating negatives for {split} split using {num_workers} workers...")
            pos_edges = positive_edges_by_split[split]
            
            # Prepare arguments for multiprocessing
            mp_args = [(head, tail, rel, ts, split_name, neighbor_dict.get(head, set()),
                        all_entities, neg_per_pos, seed) for head, tail, rel, ts, split_name in pos_edges]
            
            # Use multiprocessing to generate negatives in parallel
            with mp.Pool(processes=num_workers) as pool:
                neg_results = list(tqdm(
                    pool.imap(generate_negatives_for_edge, mp_args, chunksize=50),
                    total=len(pos_edges),
                    desc=f"{split} neg"
                ))
            
            # Process negative results
            for neg_edges in neg_results:
                for head, v_neg, rel, ts, split_name in neg_edges:
                    _append(head, v_neg, rel, ts, split_name, 0)

    print(f"Loaded {len(records):,} total (pos+neg) edges")
    save_edges(configuration, node_map, edge_type_map)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_edges_csv(df: pd.DataFrame, configuration: Dict):
    df.sort_values(["ts", "label"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_dir = Path(configuration.get("storage_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{configuration['dataset']}_edges.csv"
    df.to_csv(csv_path, index=True, index_label="edge_id")
    print(f"Edge CSV written to {csv_path} (rows={len(df)})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    configuration = load_configuration(args.config)

    edges_df = process_quad_dataset(configuration)
    save_edges_csv(edges_df, configuration)

    # save empty feature map for downstream compatibility ------------------
    torch.save({}, Path(configuration.get("storage_dir", ".")) /
               f"{configuration['dataset']}_features.pt")


if __name__ == "__main__":
    main()

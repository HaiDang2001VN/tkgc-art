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
      "train_ratio": 0.1,               # fraction of earliest train edges
                                         #     to mark as the 'pre' split
      "neg_per_pos": 1                  # (optional) negatives per positive
    }

Outputs (all under ``storage_dir``):
    <dataset>_edges.csv         – edge list with split/label columns
    <dataset>_u_map.pt         – torch‑saved dict raw‑node‑id → contiguous id
    <dataset>_v_map.pt         – "
    <dataset>_edge_type_map.pt – relation‑id → contiguous id

Node features are not available in these benchmarks; an empty dict is saved
for API compatibility.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

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


def save_edges(configuration: Dict, u_map: Dict, v_map: Dict, edge_type_map: Dict):
    """Persist mapping dictionaries next to the dataset directory."""
    output_dir = Path(configuration.get("storage_dir", "."))
    dataset = configuration["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_mapping(output_dir / f"{dataset}_u_map.pt",         u_map)
    _save_mapping(output_dir / f"{dataset}_v_map.pt",         v_map)
    _save_mapping(output_dir / f"{dataset}_edge_type_map.pt", edge_type_map)


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def process_quad_dataset(configuration: Dict) -> pd.DataFrame:
    """Read train/valid/test quadruple files and return a DataFrame.

    The resulting columns match the edge schema used elsewhere in the codebase:
        feat_pos_u | feat_pos_v | u | v | u_type | v_type | ts | split | label | edge_type
    """

    ds_name = configuration["dataset"]
    root_dir = Path(configuration.get("storage_dir", ".")) / ds_name

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

    # mapping dicts ----------------------------------------------------------------------
    u_map: Dict[int, int] = {}
    v_map: Dict[int, int] = {}
    edge_type_map: Dict[int, int] = {}
    u_next = v_next = edge_next = 0

    records = []

    def _id_assign(mapping: Dict[int, int], raw_id: int, next_id: int) -> Tuple[int, int]:
        if raw_id not in mapping:
            mapping[raw_id] = next_id
            next_id += 1
        return mapping[raw_id], next_id

    # helper to append an edge record ----------------------------------------------------
    def _append(u_raw: int, v_raw: int, r_raw: int, ts_val: int, split: str, label: int):
        nonlocal u_next, v_next, edge_next
        u, u_next = _id_assign(u_map, u_raw, u_next)
        v, v_next = _id_assign(v_map, v_raw, v_next)
        e, edge_next = _id_assign(edge_type_map, r_raw, edge_next)

        records.append({
            "feat_pos_u": u_raw,
            "feat_pos_v": v_raw,
            "u": u,
            "v": v,
            "u_type": 0,          # mono‑typed graphs
            "v_type": 0,
            "ts": ts_val,
            "split": split,
            "label": label,
            "edge_type": e,
        })

    # ----------------------------------------------------------------------
    # iterate over splits
    # ----------------------------------------------------------------------
    for split in ("train", "valid", "test"):
        fpath = root_dir / f"{split}.txt"
        df = pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            names=["head", "rel", "tail", "ts"],
            dtype={"head": np.int64, "rel": np.int64,
                   "tail": np.int64, "ts": np.int64},
        )

        # determine chronological cutoff for 'pre' split (only for train) ---------------
        if split == "train" and train_ratio > 0:
            threshold = np.quantile(df["ts"], train_ratio)
        else:
            threshold = None

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{split} pos"):
            split_label = (
                "pre" if split == "train" and threshold is not None and row.ts < threshold else split
            )
            _append(row.head, row.tail, row.rel, int(row.ts), split_label, 1)

        # simple tail‑corruption negatives for valid/test --------------------------------
        if split in ("valid", "test") and neg_per_pos > 0:
            rng = np.random.default_rng(seed=42)
            for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{split} neg"):
                corrupt_candidates = rng.choice(
                    all_entities, size=neg_per_pos, replace=False)
                for v_neg in corrupt_candidates:
                    if v_neg == row.tail:
                        continue  # avoid accidental positives
                    _append(row.head, int(v_neg), row.rel,
                            int(row.ts), split, 0)

    print(f"Loaded {len(records):,} total (pos+neg) edges")
    save_edges(configuration, u_map, v_map, edge_type_map)

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

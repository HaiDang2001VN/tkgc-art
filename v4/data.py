#!/usr/bin/env python3
import argparse
import os
import random

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Unified dataset imports
from ogb.linkproppred import LinkPropPredDataset as OGBDataset
from tgb.linkproppred.dataset import LinkPropPredDataset as TGBDataset

from utils import load_configuration


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate edges CSV and feature mappings from OGB or TGB datasets based on config.json"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the config JSON file"
    )
    return parser.parse_args()

def process_ogb_dataset(configuration):
    ogb_dataset = OGBDataset(
        name=configuration['dataset'],
        root=configuration.get('storage_dir', '.')
    )
    ogb_data = ogb_dataset[0]
    # Handle missing node features
    if ogb_dataset.is_hetero:
        feature_map = ogb_data.node_feat_dict if hasattr(
            ogb_data, 'node_feat_dict') else {}
    else:
        feature_map = {0: ogb_data['node_feat']
                       } if 'node_feat' in ogb_data else {}

    splits = ogb_dataset.get_edge_split()
    temporal_field = (
        'year'
        if any('year' in splits.get(sp, {}) for sp in ['train', 'valid', 'test'])
        else None
    )
    train_ratio = float(configuration.get('train_ratio', 0.0))

    if temporal_field:
        train_ts = splits['train'][temporal_field].tolist()
        sorted_ts = sorted(train_ts)
        threshold = sorted_ts[int(
            train_ratio * len(sorted_ts))] if sorted_ts else 0
    else:
        train_info = splits['train']
        num_edges = (
            len(train_info.get('head', []))
            if ogb_dataset.is_hetero
            else train_info['edge'].shape[0]
        )
        pre_indices = set(
            random.sample(range(num_edges), int(train_ratio * num_edges))
        )

    split_code = {'pre': 0, 'train': 1, 'valid': 2, 'test': 3}
    records = []

    for split in ['train', 'valid', 'test']:
        info = splits.get(split)
        if info is None:
            continue

        if ogb_dataset.is_hetero:
            u_types = info['head_type'].tolist()
            v_types = info['tail_type'].tolist()
            u_nodes = info['head'].tolist()
            v_nodes = info['tail'].tolist()
            edge_types = info['relation'].tolist()
            neg_pairs = list(zip(
                info.get('head_neg', []), info.get('tail_neg', [])
            ))
            timestamps = (
                info[temporal_field].tolist()
                if temporal_field else [None] * len(u_nodes)
            )

            for idx, ((u_type, v_type), u, v, edge_type_val, ts_val) in enumerate(
                tqdm(
                    zip(zip(u_types, v_types), u_nodes,
                        v_nodes, edge_types, timestamps),
                    desc=f"Processing {split} edges"
                )
            ):
                if temporal_field:
                    split_label = (
                        'pre' if split == 'train' and ts_val < threshold
                        else split
                    )
                else:
                    split_label = (
                        'pre' if split == 'train' and idx in pre_indices
                        else split
                    )
                    ts_val = split_code[split_label]

                records.append({
                    'feat_pos_u': u, 'feat_pos_v': v,
                    'u': f"{u_type}_{u}", 'v': f"{v_type}_{v}",
                    'u_type': u_type, 'v_type': v_type,
                    'ts': ts_val, 'split': split_label,
                    'label': 1, 'edge_type': edge_type_val
                })

                if idx < len(neg_pairs):
                    u_neg, v_neg = neg_pairs[idx]
                    neg_ts = ts_val if temporal_field else split_code[split]
                    records.append({
                        'feat_pos_u': u_neg, 'feat_pos_v': v_neg,
                        'u': f"{u_type}_{u_neg}", 'v': f"{v_type}_{v_neg}",
                        'u_type': u_type, 'v_type': v_type,
                        'ts': neg_ts, 'split': split,
                        'label': 0, 'edge_type': edge_type_val
                    })
        else:
            edges = info['edge'].tolist()
            neg_edges = info.get('edge_neg', [])

            for idx, (u, v) in enumerate(
                tqdm(edges, desc=f"{split} positive edges")
            ):
                if temporal_field:
                    split_label = (
                        'pre' if split == 'train' and info[temporal_field][idx] < threshold
                        else split
                    )
                    ts_val = info[temporal_field][idx]
                else:
                    split_label = (
                        'pre' if split == 'train' and idx in pre_indices
                        else split
                    )
                    ts_val = split_code[split_label]

                records.append({
                    'feat_pos_u': u, 'feat_pos_v': v,
                    'u': u, 'v': v,
                    'u_type': 0, 'v_type': 0,
                    'ts': ts_val, 'split': split_label,
                    'label': 1, 'edge_type': 0
                })

            if split in ['valid', 'test']:
                temporal_values = np.unique(
                    info[temporal_field]) if temporal_field else split_code[split]
                
                for idx, (u_neg, v_neg) in enumerate(
                    tqdm(neg_edges, desc=f"{split} negative edges")
                ):
                    ts_val = (
                        np.random.choice(temporal_values) if temporal_field # assign a random timestamp for negative edge
                        else split_code[split]
                    )
                    records.append({
                        'feat_pos_u': u_neg, 'feat_pos_v': v_neg,
                        'u': u_neg, 'v': v_neg,
                        'u_type': 0, 'v_type': 0,
                        'ts': ts_val, 'split': split,
                        'label': 0, 'edge_type': 0
                    })

    return pd.DataFrame(records), feature_map


def process_tgb_dataset(configuration):
    tgb_dataset = TGBDataset(
        name=configuration['dataset'], root=configuration.get('storage_dir', '.'), preprocess=True
    )
    data = tgb_dataset.full_data
    node_types = (
        tgb_dataset.node_type
        if tgb_dataset.node_type is not None
        else np.zeros(tgb_dataset.num_nodes, dtype=int)
    )

    train_ratio = float(configuration.get('train_ratio', 0.0))
    all_ts = data['timestamps'][tgb_dataset.train_mask]
    threshold_ts = np.sort(all_ts)[int(
        train_ratio * len(all_ts))] if len(all_ts) else 0

    records = []
    for split, mask in [('train', tgb_dataset.train_mask), ('valid', tgb_dataset.val_mask), ('test', tgb_dataset.test_mask)]:
        idxs = np.where(mask)[0]
        u_nodes, v_nodes, ts_vals, edge_types = (
            data['sources'][idxs], data['destinations'][idxs],
            data['timestamps'][idxs], data.get(
                'edge_type', np.zeros_like(data['timestamps']))[idxs]
        )

        for u, v, ts_val, edge_type_val in zip(u_nodes, v_nodes, ts_vals, edge_types):
            split_label = 'pre' if split == 'train' and ts_val < threshold_ts else split
            records.append({
                'feat_pos_u': int(u), 'feat_pos_v': int(v),
                'u': int(u), 'v': int(v),
                'u_type': int(node_types[int(u)]), 'v_type': int(node_types[int(v)]),
                'ts': int(ts_val), 'split': split_label,
                'label': 1, 'edge_type': int(edge_type_val)
            })

        if split in ('valid', 'test'):
            neg_lists = tgb_dataset.negative_sampler.query_batch(
                u_nodes, v_nodes, ts_vals, edge_type=edge_types, split_mode=split
            )
            for u, ts_val, edge_type_val, neg_vs in zip(u_nodes, ts_vals, edge_types, neg_lists):
                for nv in neg_vs:
                    records.append({
                        'feat_pos_u': int(u), 'feat_pos_v': int(nv),
                        'u': int(u), 'v': int(nv),
                        'u_type': int(node_types[int(u)]), 'v_type': int(node_types[int(nv)]),
                        'ts': int(ts_val), 'split': split,
                        'label': 0, 'edge_type': int(edge_type_val)
                    })

    return pd.DataFrame(records)


def save_edges_csv(dataframe, configuration):
    dataframe.sort_values(['ts', 'label'], ascending=[
                          True, False], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    output_dir = configuration.get('storage_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{configuration['dataset']}_edges.csv")
    dataframe.to_csv(path, index=True, index_label="edge_id")
    print(f"Saved edges CSV to: {path}")


def save_feature_map(feature_map, configuration):
    output_dir = configuration.get('storage_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{configuration['dataset']}_features.pt")
    torch.save(feature_map, path)
    print(f"Saved node features to: {path}")


def main():
    args = parse_arguments()
    configuration = load_configuration(args.config)
    dataset_key = configuration.get('dataset', '').lower()

    if 'ogb' in dataset_key:
        edges_df, feature_map = process_ogb_dataset(configuration)
        save_edges_csv(edges_df, configuration)
        save_feature_map(feature_map, configuration)
    elif 'tgb' in dataset_key:
        edges_df = process_tgb_dataset(configuration)
        save_edges_csv(edges_df, configuration)
    else:
        raise ValueError(
            "Unsupported dataset type; must contain 'ogb' or 'tgb'.")


if __name__ == '__main__':
    main()

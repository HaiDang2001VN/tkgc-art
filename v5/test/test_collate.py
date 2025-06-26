#!/usr/bin/env python3
"""
Test script to demonstrate the new prefix-based collate function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader import collate_by_prefix_length
import torch

def test_collate_function():
    # Create sample data that simulates what __getitem__ would return
    sample_batch = [
        {
            'label': torch.tensor(1),
            'u': torch.tensor(100),
            'v': torch.tensor(200),
            'ts': torch.tensor(1000),
            'negs_by_prefix_length': {
                1: [[101], [102]],  # Two negatives for prefix length 1
                2: [[103, 104]]     # One negative for prefix length 2
            },
            'neg_edge_types_by_prefix_length': {
                1: [[], []],
                2: [[10]]
            },
            'neg_timestamps_by_prefix_length': {
                1: [[1001], [1002]],
                2: [[1003, 1004]]
            },
            'paths': [[100, 200], [101], [102], [103, 104]],  # pos + negs
            'path_timestamps': [[1000, 1001], [1001], [1002], [1003, 1004]],
            'shallow_emb': [
                torch.randn(2, 128),  # positive path embedding
                torch.randn(1, 128),  # negative path 1 embedding (prefix 1)
                torch.randn(1, 128),  # negative path 2 embedding (prefix 1)
                torch.randn(2, 128)   # negative path 3 embedding (prefix 2)
            ],
            'edge_emb': [
                torch.randn(1, 128),  # positive path edge embedding
                torch.randn(0, 128),  # negative path 1 edge embedding (empty)
                torch.randn(0, 128),  # negative path 2 edge embedding (empty)
                torch.randn(1, 128)   # negative path 3 edge embedding
            ]
        },
        {
            'label': torch.tensor(0),
            'u': torch.tensor(300),
            'v': torch.tensor(400),
            'ts': torch.tensor(2000),
            'negs_by_prefix_length': {
                1: [[301]],         # One negative for prefix length 1
                3: [[302, 303, 304]]  # One negative for prefix length 3
            },
            'neg_edge_types_by_prefix_length': {
                1: [[]],
                3: [[20, 21]]
            },
            'neg_timestamps_by_prefix_length': {
                1: [[2001]],
                3: [[2002, 2003, 2004]]
            },
            'paths': [[300, 400], [301], [302, 303, 304]],  # pos + negs
            'path_timestamps': [[2000, 2001], [2001], [2002, 2003, 2004]],
            'shallow_emb': [
                torch.randn(2, 128),  # positive path embedding
                torch.randn(1, 128),  # negative path 1 embedding (prefix 1)
                torch.randn(3, 128)   # negative path 2 embedding (prefix 3)
            ],
            'edge_emb': [
                torch.randn(1, 128),  # positive path edge embedding
                torch.randn(0, 128),  # negative path 1 edge embedding (empty)
                torch.randn(2, 128)   # negative path 2 edge embedding
            ]
        }
    ]
    
    print("Testing collate_by_prefix_length function...")
    print("Input batch contains 2 samples with different prefix lengths")
    
    # Test the collate function
    result = collate_by_prefix_length(sample_batch)
    
    print(f"\nResult keys (prefix lengths): {list(result.keys())}")
    
    for prefix_len, samples in result.items():
        print(f"\nPrefix length {prefix_len}:")
        print(f"  Number of samples: {len(samples)}")
        
        for i, sample in enumerate(samples):
            print(f"  Sample {i}:")
            print(f"    label: {sample['label']}")
            print(f"    u: {sample['u']}, v: {sample['v']}")
            print(f"    pos_path: {sample.get('pos_path', 'N/A')}")
            print(f"    neg_paths: {sample['neg_paths']}")
            print(f"    neg_timestamps: {sample['neg_timestamps']}")
            print(f"    prefix_length: {sample['prefix_length']}")
            
            if 'pos_shallow_emb' in sample:
                print(f"    pos_shallow_emb shape: {sample['pos_shallow_emb'].shape}")
            if 'neg_shallow_embs' in sample and sample['neg_shallow_embs']:
                print(f"    neg_shallow_embs shapes: {[emb.shape for emb in sample['neg_shallow_embs']]}")

if __name__ == "__main__":
    test_collate_function()

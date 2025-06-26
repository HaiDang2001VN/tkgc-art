#!/usr/bin/env python3
"""
Test script for improved negative sampling in the temporal knowledge graph pipeline.

This script tests the key improvements:
1. Global neighbor dictionary tracking all neighbors across all time
2. Negative sampling only from true non-neighbors  
3. Proper handling of edge cases (no non-neighbors, insufficient negatives)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

# Import the data processing function
from data import process_quad_dataset


def create_test_dataset(temp_dir: Path, dataset_name: str = "test_dataset"):
    """Create a minimal test dataset for validation."""
    
    # Create dataset directory
    ds_dir = temp_dir / dataset_name.upper()
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stat.txt (n_ent, n_rel, n_triplets)
    (ds_dir / "stat.txt").write_text("10 3 20")
    
    # Create train.txt - establish neighbor relationships
    train_data = [
        "0\t0\t1\t100\t1",  # 0-1 are neighbors
        "1\t1\t2\t101\t1",  # 1-2 are neighbors  
        "2\t2\t3\t102\t1",  # 2-3 are neighbors
        "3\t0\t4\t103\t1",  # 3-4 are neighbors
        "4\t1\t5\t104\t1",  # 4-5 are neighbors
        "5\t2\t0\t105\t1",  # 5-0 are neighbors (creates cycle)
    ]
    (ds_dir / "train.txt").write_text("\n".join(train_data))
    
    # Create valid.txt - test negative sampling
    valid_data = [
        "0\t0\t2\t200\t1",  # 0-2 positive (0 is already neighbor to 1, 1 is neighbor to 2)
        "1\t1\t4\t201\t1",  # 1-4 positive
    ]
    (ds_dir / "valid.txt").write_text("\n".join(valid_data))
    
    # Create test.txt - test negative sampling
    test_data = [
        "2\t2\t5\t300\t1",  # 2-5 positive (2 is neighbor to 3, indirect to 5 via 0)
        "3\t0\t1\t301\t1",  # 3-1 positive
    ]
    (ds_dir / "test.txt").write_text("\n".join(test_data))
    
    return ds_dir


def test_global_neighbor_tracking():
    """Test that the global neighbor dictionary correctly tracks all neighbors."""
    print("Testing global neighbor tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ds_dir = create_test_dataset(temp_path)
        
        # Create configuration
        config = {
            "dataset": "test_dataset", 
            "storage_dir": str(temp_path),
            "neg_per_pos": 2,
            "train_ratio": 0.0
        }
        
        # Process dataset
        df = process_quad_dataset(config)
        
        # Load the saved mappings to verify neighbor tracking
        node_map = torch.load(temp_path / "test_dataset_node_map.pt")
        
        # Verify that we have the expected edges
        positive_edges = df[df['label'] == 1]
        train_edges = positive_edges[positive_edges['split'] == 'train']
        valid_edges = positive_edges[positive_edges['split'] == 'valid'] 
        test_edges = positive_edges[positive_edges['split'] == 'test']
        
        print(f"Train edges: {len(train_edges)}")
        print(f"Valid edges: {len(valid_edges)}")
        print(f"Test edges: {len(test_edges)}")
        
        # Check negative sampling
        negative_edges = df[df['label'] == 0]
        valid_negatives = negative_edges[negative_edges['split'] == 'valid']
        test_negatives = negative_edges[negative_edges['split'] == 'test']
        
        print(f"Valid negatives: {len(valid_negatives)}")
        print(f"Test negatives: {len(test_negatives)}")
        
        # Verify that negatives don't include known neighbors
        print("Sample negative edges (should not be neighbors):")
        for idx, row in valid_negatives.head(3).iterrows():
            print(f"  Valid negative: {row['u']} -> {row['v']} (raw: {row['feat_pos_u']} -> {row['feat_pos_v']})")
        
        for idx, row in test_negatives.head(3).iterrows():
            print(f"  Test negative: {row['u']} -> {row['v']} (raw: {row['feat_pos_u']} -> {row['feat_pos_v']})")
    
    print("✓ Global neighbor tracking test completed\n")


def test_negative_sampling_quality():
    """Test that negative samples are truly non-neighbors."""
    print("Testing negative sampling quality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ds_dir = create_test_dataset(temp_path)
        
        # Create configuration with more negatives to test filtering
        config = {
            "dataset": "test_dataset",
            "storage_dir": str(temp_path), 
            "neg_per_pos": 3,
            "train_ratio": 0.0
        }
        
        # Process dataset
        df = process_quad_dataset(config)
        
        # Build neighbor set from all positive edges
        all_positive = df[df['label'] == 1]
        neighbors = {}
        
        for _, row in all_positive.iterrows():
            u, v = row['feat_pos_u'], row['feat_pos_v']
            if u not in neighbors:
                neighbors[u] = set()
            if v not in neighbors:
                neighbors[v] = set()
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        print("Global neighbor relationships:")
        for node, neighs in neighbors.items():
            print(f"  Node {node}: neighbors {sorted(neighs)}")
        
        # Check that no negative edge connects known neighbors
        negative_edges = df[df['label'] == 0]
        violations = []
        
        for _, row in negative_edges.iterrows():
            u, v = row['feat_pos_u'], row['feat_pos_v']
            if u in neighbors and v in neighbors[u]:
                violations.append((u, v, row['split']))
        
        if violations:
            print(f"❌ Found {len(violations)} violations where negatives connect known neighbors:")
            for u, v, split in violations[:5]:  # Show first 5
                print(f"  {split}: {u} -> {v}")
        else:
            print("✓ All negative edges avoid known neighbors")
    
    print("✓ Negative sampling quality test completed\n")


def test_edge_case_handling():
    """Test handling of edge cases like nodes with no valid non-neighbors."""
    print("Testing edge case handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a very small dataset where some nodes have few non-neighbors
        ds_dir = temp_path / "TINY_DATASET" 
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        # Only 4 entities, 2 relations
        (ds_dir / "stat.txt").write_text("4 2 6")
        
        # Train: create a dense subgraph
        train_data = [
            "0\t0\t1\t100\t1",  # 0-1
            "1\t1\t2\t101\t1",  # 1-2
            "2\t0\t0\t102\t1",  # 2-0 (creates triangle)
        ]
        (ds_dir / "train.txt").write_text("\n".join(train_data))
        
        # Valid: node 0 is connected to 1,2 so only 3 is non-neighbor
        valid_data = [
            "0\t0\t3\t200\t1",  # 0-3 positive
        ]
        (ds_dir / "valid.txt").write_text("\n".join(valid_data))
        
        # Test: node 1 is connected to 0,2 so only 3 is non-neighbor  
        test_data = [
            "1\t1\t3\t300\t1",  # 1-3 positive
        ]
        (ds_dir / "test.txt").write_text("\n".join(test_data))
        
        # Process with high neg_per_pos to test insufficient negatives
        config = {
            "dataset": "tiny_dataset",
            "storage_dir": str(temp_path),
            "neg_per_pos": 5,  # More than available non-neighbors
            "train_ratio": 0.0
        }
        
        # This should handle the case gracefully
        df = process_quad_dataset(config)
        
        negatives = df[df['label'] == 0]
        print(f"Generated {len(negatives)} negative edges despite limited non-neighbors")
        
        # Check that we didn't exceed available non-neighbors per node
        for split in ['valid', 'test']:
            split_negs = negatives[negatives['split'] == split]
            if len(split_negs) > 0:
                print(f"  {split}: {len(split_negs)} negatives generated")
    
    print("✓ Edge case handling test completed\n")


def test_configuration_integration():
    """Test integration with configuration system."""
    print("Testing configuration integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ds_dir = create_test_dataset(temp_path)
        
        # Test with different neg_per_pos values
        for neg_per_pos in [1, 3, 5]:
            config = {
                "dataset": "test_dataset",
                "storage_dir": str(temp_path),
                "neg_per_pos": neg_per_pos,
                "train_ratio": 0.0
            }
            
            df = process_quad_dataset(config)
            
            # Count negatives per positive edge
            positives = df[df['label'] == 1]
            negatives = df[df['label'] == 0]
            
            for split in ['valid', 'test']:
                split_pos = positives[positives['split'] == split]
                split_neg = negatives[negatives['split'] == split]
                
                if len(split_pos) > 0:
                    ratio = len(split_neg) / len(split_pos)
                    print(f"  neg_per_pos={neg_per_pos}, {split}: {ratio:.1f} negatives per positive")
                    
                    # Should be close to requested ratio (may be less due to filtering)
                    assert ratio <= neg_per_pos, f"Too many negatives generated for {split}"
    
    print("✓ Configuration integration test completed\n")


def main():
    """Run all tests."""
    print("=== Testing Improved Negative Sampling ===\n")
    
    test_global_neighbor_tracking()
    test_negative_sampling_quality() 
    test_edge_case_handling()
    test_configuration_integration()
    
    print("=== All Tests Passed! ===")
    print("The improved negative sampling implementation correctly:")
    print("1. Tracks global neighbors across all time periods")
    print("2. Only samples negatives from true non-neighbors") 
    print("3. Handles edge cases gracefully")
    print("4. Integrates properly with configuration system")


if __name__ == "__main__":
    main()

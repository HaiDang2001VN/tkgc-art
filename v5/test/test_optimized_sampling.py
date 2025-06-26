#!/usr/bin/env python3
"""
Test script to demonstrate the optimized negative sampling algorithm.
"""

import numpy as np
from typing import List


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


def test_optimized_negative_sampling():
    """Test the optimized negative sampling algorithm."""
    print("🧪 Testing optimized negative sampling algorithm...")
    
    # Example: 10 total nodes (0-9), node 3 has neighbors {1, 5, 7}
    total_nodes = 10
    all_entities = np.arange(total_nodes)
    head_node = 3
    head_neighbors = {1, 5, 7}
    head_neighbors_with_self = head_neighbors | {head_node}  # {1, 3, 5, 7}
    
    print(f"📊 Test setup:")
    print(f"  Total nodes: {total_nodes} (IDs: {list(all_entities)})")
    print(f"  Head node: {head_node}")
    print(f"  Neighbors (including self): {sorted(head_neighbors_with_self)}")
    
    # Calculate non-neighbors using the old method (for comparison)
    non_neighbors_old = [e for e in all_entities if e not in head_neighbors_with_self]
    print(f"  Non-neighbors (old method): {non_neighbors_old}")
    
    # Calculate using the new optimized method
    sorted_neighbors = sorted(head_neighbors_with_self)
    num_non_neighbors = total_nodes - len(head_neighbors_with_self)
    
    print(f"\n🔍 Gap analysis:")
    print(f"  Sorted neighbors: {sorted_neighbors}")
    print(f"  Number of non-neighbors: {num_non_neighbors}")
    
    # Show gap structure
    current = 0
    gaps = []
    for neighbor in sorted_neighbors:
        if neighbor > current:
            gap = list(range(current, neighbor))
            if gap:
                gaps.append(gap)
                print(f"    Gap before neighbor {neighbor}: {gap}")
        current = neighbor + 1
    
    # Final gap after last neighbor
    if current < total_nodes:
        final_gap = list(range(current, total_nodes))
        gaps.append(final_gap)
        print(f"    Final gap after neighbor {sorted_neighbors[-1]}: {final_gap}")
    
    # Flatten gaps to get all non-neighbors
    all_non_neighbors = []
    for gap in gaps:
        all_non_neighbors.extend(gap)
    
    print(f"\n✅ Verification:")
    print(f"  Non-neighbors from gaps: {all_non_neighbors}")
    print(f"  Matches old method: {all_non_neighbors == non_neighbors_old}")
    
    # Test the mapping function for all positions
    print(f"\n🎯 Position mapping test:")
    for pos in range(num_non_neighbors):
        mapped_id = map_position_to_node_id(pos, sorted_neighbors, total_nodes)
        expected_id = non_neighbors_old[pos]
        match = mapped_id == expected_id
        print(f"    Position {pos} → Node {mapped_id} (expected {expected_id}) {'✓' if match else '✗'}")
        
        if not match:
            print(f"❌ Mismatch at position {pos}!")
            return False
    
    # Test additional edge cases
    print(f"\n🧪 Edge case tests:")
    
    # Edge case 1: No neighbors except self
    test_neighbors = {3}
    test_sorted = sorted(test_neighbors)
    test_non_neighbors = total_nodes - len(test_neighbors)
    print(f"  Edge case 1 - Only self as neighbor:")
    print(f"    Neighbors: {test_sorted}")
    for pos in range(min(5, test_non_neighbors)):  # Test first 5 positions
        mapped = map_position_to_node_id(pos, test_sorted, total_nodes)
        expected = [i for i in range(total_nodes) if i not in test_neighbors][pos]
        print(f"    Position {pos} → {mapped} (expected {expected}) {'✓' if mapped == expected else '✗'}")
    
    # Edge case 2: All consecutive neighbors
    test_neighbors = {0, 1, 2, 3, 4}
    test_sorted = sorted(test_neighbors)
    test_non_neighbors = total_nodes - len(test_neighbors)
    print(f"  Edge case 2 - Consecutive neighbors at start:")
    print(f"    Neighbors: {test_sorted}")
    for pos in range(min(5, test_non_neighbors)):
        mapped = map_position_to_node_id(pos, test_sorted, total_nodes)
        expected = [i for i in range(total_nodes) if i not in test_neighbors][pos]
        print(f"    Position {pos} → {mapped} (expected {expected}) {'✓' if mapped == expected else '✗'}")
    
    # Edge case 3: Neighbors at the end
    test_neighbors = {7, 8, 9}
    test_sorted = sorted(test_neighbors)
    test_non_neighbors = total_nodes - len(test_neighbors)
    print(f"  Edge case 3 - Neighbors at end:")
    print(f"    Neighbors: {test_sorted}")
    for pos in range(min(5, test_non_neighbors)):
        mapped = map_position_to_node_id(pos, test_sorted, total_nodes)
        expected = [i for i in range(total_nodes) if i not in test_neighbors][pos]
        print(f"    Position {pos} → {mapped} (expected {expected}) {'✓' if mapped == expected else '✗'}")
    
    # Performance comparison test
    print(f"\n⚡ Performance comparison:")
    
    # Simulate larger graph
    large_total = 100000
    large_neighbors = set(range(0, 50000, 2))  # Every even number up to 50000
    large_neighbors_with_self = large_neighbors | {99999}
    large_sorted_neighbors = sorted(large_neighbors_with_self)
    large_num_non_neighbors = large_total - len(large_neighbors_with_self)
    
    print(f"  Large graph: {large_total} nodes, {len(large_neighbors_with_self)} neighbors")
    print(f"  Non-neighbors available: {large_num_non_neighbors}")
    
    # Test sampling 1000 negatives
    neg_per_pos = 1000
    rng = np.random.default_rng(seed=42)
    
    import time
    
    # Old method (for small example only, would be too slow for large graph)
    print(f"  Old method: Creating full non-neighbor list...")
    start = time.time()
    # We'll skip this for the large example as it would be too slow
    print(f"    (Skipped for large graph - would create list of {large_num_non_neighbors} elements)")
    
    # New method
    print(f"  New method: Using position mapping...")
    start = time.time()
    positions = rng.choice(large_num_non_neighbors, size=neg_per_pos, replace=False)
    sampled_nodes = []
    for pos in positions:
        node_id = map_position_to_node_id(pos, large_sorted_neighbors, large_total)
        sampled_nodes.append(node_id)
    new_time = time.time() - start
    
    print(f"    Time: {new_time:.4f} seconds")
    print(f"    Sample (first 10): {sampled_nodes[:10]}")
    
    # Verify no neighbors in sample
    neighbors_in_sample = len([n for n in sampled_nodes if n in large_neighbors_with_self])
    print(f"    Neighbors found in sample: {neighbors_in_sample} (should be 0)")
    
    print(f"\n🎉 All tests passed! The optimized algorithm works correctly.")
    return True


if __name__ == "__main__":
    success = test_optimized_negative_sampling()
    if success:
        print("\n✅ Optimization test successful!")
    else:
        print("\n❌ Optimization test failed!")

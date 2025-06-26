# Optimized Negative Sampling - Summary

## Overview
Replaced the list-based negative sampling with an efficient integer arithmetic approach that avoids creating large intermediate lists of non-neighbor nodes.

## Key Optimization

### **Before (Inefficient)**
```python
# Create full list of non-neighbors (memory intensive)
head_neighbors = neighbor_dict.get(head, set())
non_neighbors = [e for e in all_entities if e not in head_neighbors and e != head]

# Sample from the list
candidates = rng.choice(non_neighbors, size=sample_size, replace=False)
```

### **After (Optimized)**
```python
# Calculate number of non-neighbors without creating the list
head_neighbors_with_self = head_neighbors | {head}
num_non_neighbors = len(all_entities) - len(head_neighbors_with_self)

# Sample positions in the non-neighbor space
random_positions = rng.choice(num_non_neighbors, size=sample_size, replace=False)

# Map positions to actual node IDs using gap arithmetic
sorted_neighbors = sorted(head_neighbors_with_self)
for pos in random_positions:
    actual_node_id = map_position_to_node_id(pos, sorted_neighbors, len(all_entities))
```

## Core Algorithm: `map_position_to_node_id()`

The function maps a position in the "non-neighbor space" to an actual node ID by:

1. **Gap Traversal**: Iterate through sorted neighbors
2. **Gap Size Calculation**: For each neighbor, calculate the gap size before it
3. **Position Mapping**: If position falls in current gap, return `current_node + remaining_position`
4. **Skip and Continue**: Otherwise, skip over the gap and neighbor, continue to next

### **Example**
```
Total nodes: 10 (IDs: 0,1,2,3,4,5,6,7,8,9)
Head node: 3, Neighbors: {1,5,7}, With self: {1,3,5,7}

Gap structure:
  Gap 0: [0] (before neighbor 1)
  Gap 1: [2] (between neighbors 1 and 3)  
  Gap 2: [4] (between neighbors 3 and 5)
  Gap 3: [6] (between neighbors 5 and 7)
  Gap 4: [8,9] (after neighbor 7)

Non-neighbor positions: [0,1,2,3,4,5] map to nodes [0,2,4,6,8,9]
```

## Performance Benefits

### **Memory Efficiency**
- **Before**: O(n) memory to store non-neighbor list
- **After**: O(k) memory where k = number of neighbors (typically k << n)

### **Time Complexity**
- **Before**: O(n) to create list + O(sample_size) to sample
- **After**: O(k log k) to sort neighbors + O(sample_size × k) to map positions
- For sparse graphs where k << n, this is much faster

### **Scalability**
- Works efficiently even with millions of nodes
- Memory usage independent of graph size (only depends on node degree)
- No intermediate list creation

## Implementation Details

### **Enhanced Features**
1. **Self-Loop Prevention**: Includes head node in neighbor set
2. **Safety Checks**: Validates mapped nodes are truly non-neighbors  
3. **Efficient Sampling**: Uses `neg_per_pos * 2` for better sample diversity
4. **Sorted Neighbor List**: Enables efficient gap traversal

### **Edge Cases Handled**
- Empty neighbor sets (all nodes except self are candidates)
- Full neighbor sets (no negative samples possible)
- Boundary conditions (first/last gaps)

## Example Usage

```python
# For a node with 1000 neighbors in a graph of 1M nodes:
# Old: Create list of 999,000 non-neighbors (expensive)
# New: Sort 1000 neighbors + map positions (fast)

head_neighbors = {1, 5, 7, ..., 995}  # 1000 neighbors
total_nodes = 1_000_000

# Sample 100 negatives efficiently
positions = rng.choice(999_000, size=200, replace=False)  # 2x for diversity
for pos in positions[:100]:
    node_id = map_position_to_node_id(pos, sorted(head_neighbors | {head}), total_nodes)
```

## Benefits Summary

1. **🚀 Performance**: Significantly faster for large graphs
2. **💾 Memory**: Constant memory usage regardless of graph size  
3. **🎯 Accuracy**: Guarantees true non-neighbors across all time
4. **⚖️ Scalability**: Works efficiently with millions of nodes
5. **🔧 Maintainable**: Clean, understandable algorithm

This optimization makes negative sampling practical for large-scale temporal knowledge graphs while maintaining the same quality of negative samples.

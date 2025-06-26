# Edge Embeddings Enhancement - Summary

## Changes Made

The `EdgeDataset` class in `loader.py` has been enhanced to return edge type embeddings alongside node embeddings from the Knowledge Graph Embedding (KGE) model.

### Key Modifications:

1. **Positive Path Edge Types Extraction**:
   - Added extraction of `edge_types` from positive path information
   - Modified line ~52: Added `pos_edge_types = pos_path_info.get('edge_types', [])`

2. **Negative Path Edge Types Extraction**:
   - Enhanced the negative path processing to extract edge types from interleaved paths
   - For paths like `[n0, r1, n1, r2, n2]`, nodes are at even indices `[n0, n1, n2]` and edge types are at odd indices `[r1, r2]`
   - Added `negs_edge_types_only` list to store edge types for each negative path

3. **Edge Type Embeddings Extraction**:
   - Added logic to extract edge type embeddings using `kge_proxy.model.rel_emb(edge_types_tensor)`
   - Embeddings are extracted only when:
     - Edge types exist for the path
     - KGE model has `rel_emb` attribute
     - `rel_emb` is not None
   - Empty tensors are added for paths without edge types to maintain structure consistency

4. **New Return Field**:
   - Added `item['edge_emb']` - a list of tensors containing edge type embeddings
   - Each tensor has shape `(num_edges_in_path, kge_dim)`
   - For a path with N nodes, there are N-1 edges, so N-1 edge embeddings

### Data Structure:

**Input Path Structure:**
- Positive paths: stored with separate `nodes` and `edge_types` lists
- Negative paths: stored as interleaved sequences `[node, edge_type, node, edge_type, ...]`

**Output Structure:**
```python
item = {
    'label': tensor,
    'paths': [[node_ids], ...],           # List of node ID lists
    'shallow_emb': [tensor, ...],         # List of node embedding tensors
    'edge_emb': [tensor, ...],            # NEW: List of edge embedding tensors
    # ... other fields
}
```

### Edge Cases Handled:
- Empty paths: Empty tensors with correct dimensions
- Missing edge types: Empty tensors
- No KGE model: Edge embeddings not added
- KGE model without relation embeddings: Empty tensors

### Benefits:
- Maintains backward compatibility
- Provides rich edge type information for downstream models
- Consistent tensor shapes for batch processing
- Proper device handling (GPU/CPU)

### Usage:
The enhanced dataset can now provide both node and edge embeddings, enabling models to leverage both node features and relationship types in their learning process. The number of edge embeddings equals the number of edges in each path (nodes - 1).

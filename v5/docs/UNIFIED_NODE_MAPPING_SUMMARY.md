# Unified Node Mapping - Summary of Changes

## Overview
Modified `data.py` to treat `u` (source) and `v` (target) nodes as the same entity type, using a unified mapping instead of separate mappings.

## Changes Made

### 1. **Unified Mapping Variables**
**Before:**
```python
u_map: Dict[int, int] = {}
v_map: Dict[int, int] = {}
edge_type_map: Dict[int, int] = {}
u_next = v_next = edge_next = 0
```

**After:**
```python
node_map: Dict[int, int] = {}  # unified mapping for both u and v
edge_type_map: Dict[int, int] = {}
node_next = edge_next = 0
```

### 2. **Updated ID Assignment Logic**
**Before:**
```python
def _append(u_raw: int, v_raw: int, r_raw: int, ts_val: int, split: str, label: int):
    nonlocal u_next, v_next, edge_next
    u, u_next = _id_assign(u_map, u_raw, u_next)
    v, v_next = _id_assign(v_map, v_raw, v_next)
    e, edge_next = _id_assign(edge_type_map, r_raw, edge_next)
```

**After:**
```python
def _append(u_raw: int, v_raw: int, r_raw: int, ts_val: int, split: str, label: int):
    nonlocal node_next, edge_next
    u, node_next = _id_assign(node_map, u_raw, node_next)
    v, node_next = _id_assign(node_map, v_raw, node_next)
    e, edge_next = _id_assign(edge_type_map, r_raw, edge_next)
```

### 3. **Updated Save Function**
**Before:**
```python
def save_edges(configuration: Dict, u_map: Dict, v_map: Dict, edge_type_map: Dict):
    _save_mapping(output_dir / f"{dataset}_u_map.pt",         u_map)
    _save_mapping(output_dir / f"{dataset}_v_map.pt",         v_map)
    _save_mapping(output_dir / f"{dataset}_edge_type_map.pt", edge_type_map)
```

**After:**
```python
def save_edges(configuration: Dict, node_map: Dict, edge_type_map: Dict):
    _save_mapping(output_dir / f"{dataset}_node_map.pt",      node_map)
    _save_mapping(output_dir / f"{dataset}_edge_type_map.pt", edge_type_map)
```

### 4. **Updated Function Call**
**Before:**
```python
save_edges(configuration, u_map, v_map, edge_type_map)
```

**After:**
```python
save_edges(configuration, node_map, edge_type_map)
```

### 5. **Updated Documentation**
**Before:**
```
<dataset>_u_map.pt         – torch‑saved dict raw‑node‑id → contiguous id
<dataset>_v_map.pt         – "
<dataset>_edge_type_map.pt – relation‑id → contiguous id
```

**After:**
```
<dataset>_node_map.pt       – torch‑saved dict raw‑node‑id → contiguous id
<dataset>_edge_type_map.pt  – relation‑id → contiguous id
```

## Benefits

1. **Simplified Architecture**: Single mapping for all nodes regardless of their role (source or target)
2. **Consistent ID Space**: All nodes share the same contiguous ID space
3. **Reduced Complexity**: Eliminates need to track separate mappings
4. **More Intuitive**: Reflects the reality that nodes can appear as both sources and targets
5. **Efficient**: Reduces the number of mapping files from 3 to 2

## Output Files
After the changes, the script now generates:
- `<dataset>_edges.csv` - Edge list with split/label columns
- `<dataset>_node_map.pt` - Unified node mapping (raw → contiguous IDs)
- `<dataset>_edge_type_map.pt` - Edge type mapping
- `<dataset>_features.pt` - Empty features dict (for compatibility)

The unified node mapping ensures that all entities (whether appearing as sources or targets) are mapped to a single, consistent ID space.

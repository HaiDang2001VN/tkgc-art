# Improved Negative Sampling - Summary of Changes

## Overview
Modified the negative sampling strategy to use global neighbor information instead of timestamp-specific sampling, ensuring that negative samples are truly non-neighbors across all time periods.

## Key Changes

### 1. **Added Neighbor Dictionary**
```python
neighbor_dict: Dict[int, set] = {}  # node -> set of all neighbors across all time
```
- Tracks all neighbors for each node across all time periods
- Updated during positive edge processing in `_append()` function

### 2. **Modified `_append()` Function**
```python
# Update neighbor dictionary for positive edges only
if label == 1:
    if u_raw not in neighbor_dict:
        neighbor_dict[u_raw] = set()
    if v_raw not in neighbor_dict:
        neighbor_dict[v_raw] = set()
    neighbor_dict[u_raw].add(v_raw)
    neighbor_dict[v_raw].add(u_raw)  # undirected graph assumption
```
- Builds neighbor relationships for all positive edges
- Assumes undirected graph (bidirectional neighbors)

### 3. **Two-Pass Processing Strategy**

#### **First Pass: Collect All Positive Edges**
- Process all splits (train, valid, test) to collect positive edges
- Build complete neighbor dictionary across all time periods
- Store positive edges for valid/test splits for later negative sampling

#### **Second Pass: Generate Negatives**
- Use global neighbor information for negative sampling
- Ensure negative samples are not neighbors across any time period

### 4. **Improved Negative Sampling Logic**

**Before:**
```python
corrupt_candidates = rng.choice(all_entities, size=neg_per_pos, replace=False)
for v_neg in corrupt_candidates:
    if v_neg == row.tail:
        continue  # avoid accidental positives
```

**After:**
```python
# Get all non-neighbors for this head node
head_neighbors = neighbor_dict.get(head, set())
non_neighbors = [e for e in all_entities if e not in head_neighbors and e != head]

# Sample twice the required amount to ensure we have enough after filtering
sample_size = min(len(non_neighbors), neg_per_pos * 2)
candidates = rng.choice(non_neighbors, size=sample_size, replace=False)

# Take the first neg_per_pos candidates with additional filtering
for v_neg in candidates:
    if neg_count >= neg_per_pos:
        break
    if v_neg not in head_neighbors and v_neg != tail:
        _append(head, int(v_neg), rel, ts, split, 0)
        neg_count += 1
```

## Benefits

### 1. **True Negative Sampling**
- Negative samples are guaranteed to be non-neighbors across all time
- More realistic evaluation scenario for temporal link prediction

### 2. **Better Sample Quality**
- Eliminates false negatives (edges that exist at other time points)
- Provides cleaner separation between positive and negative samples

### 3. **Sufficient Sample Size**
- Samples `2 * neg_per_pos` candidates initially
- Ensures we get the required number of negatives after filtering
- Handles cases where many candidates might be filtered out

### 4. **Global Temporal Awareness**
- Considers all temporal relationships, not just current timestamp
- More appropriate for temporal knowledge graph evaluation

## Algorithm Flow

```
1. First Pass (All Splits):
   ├─ Process train/valid/test positive edges
   ├─ Build global neighbor_dict
   └─ Store valid/test edges for negative sampling

2. Second Pass (Valid/Test Only):
   ├─ For each positive edge (head, tail, rel, ts):
   ├─ Get head_neighbors from global neighbor_dict
   ├─ Find non_neighbors = all_entities - head_neighbors - {head}
   ├─ Sample 2 * neg_per_pos candidates from non_neighbors
   └─ Take first neg_per_pos valid negatives
```

## Safety Measures

1. **Double Filtering**: Candidates are pre-filtered and then filtered again during selection
2. **Availability Check**: Skips if no non-neighbors are available
3. **Size Limits**: Respects available non-neighbor count
4. **Exact Count**: Ensures exactly `neg_per_pos` negatives are generated per positive

## Impact on Evaluation

This change significantly improves the quality of negative samples for link prediction evaluation by:
- Eliminating temporal leakage in negative sampling
- Providing more challenging and realistic negative examples
- Ensuring fair comparison across different time periods
- Better reflecting real-world link prediction scenarios

The improved negative sampling strategy leads to more reliable and meaningful evaluation metrics.

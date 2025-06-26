# Manual Evaluation Implementation - Summary

## Overview
Replaced the OGB/TGB evaluator dependency with a manual implementation that calculates MRR (Mean Reciprocal Rank) and Hits@K metrics for link prediction tasks.

## Key Changes

### 1. **Removed Dependencies**
- **Before**: Required OGB (`pip install ogb`) and TGB (`pip install py-tgb`) libraries
- **After**: Self-contained implementation using only PyTorch and NumPy

### 2. **Updated Input Format**
- **Before**: `evaluate(dataset_name, pos_scores, neg_scores)`
- **After**: `evaluate(pos_scores, neg_scores)` (removed dataset_name parameter)

### 3. **Specific Input Shapes**
- **pos_scores**: Shape `(num_edges,)` - One score per positive edge
- **neg_scores**: Shape `(num_edges, num_neg)` - Multiple negative scores per positive edge

### 4. **Core Implementation: `calculate_mrr_hits()`**

```python
def calculate_mrr_hits(pos_scores, neg_scores, k_values=[1, 3, 10]):
    # Calculate ranking: how many negatives score higher than positive
    ranks = (neg_scores > pos_scores_expanded).sum(dim=1) + 1
    
    # MRR = average of reciprocal ranks
    mrr = (1.0 / ranks.float()).mean().item()
    
    # Hits@K = fraction of cases where rank <= K
    hits_at_k = {f'hits@{k}': (ranks <= k).float().mean().item() for k in k_values}
```

### 5. **Output Format**
Returns a dictionary with:
```python
{
    'mrr': 0.7234,        # Mean Reciprocal Rank
    'hits@1': 0.4567,     # Hits@1
    'hits@3': 0.6789,     # Hits@3
    'hits@10': 0.8901     # Hits@10
}
```

## Metric Definitions

### **MRR (Mean Reciprocal Rank)**
- For each positive edge, calculate its rank among all candidates (positive + negatives)
- Rank = (number of negatives with higher score) + 1
- Reciprocal rank = 1 / rank
- MRR = average of all reciprocal ranks
- Range: [0, 1], higher is better

### **Hits@K**
- Fraction of positive edges that rank in the top K positions
- Range: [0, 1], higher is better
- Hits@1 ≤ Hits@3 ≤ Hits@10 (non-decreasing property)

## Example Usage

### **As a Module**
```python
from evaluation import evaluate
import torch

pos_scores = torch.tensor([0.8, 0.9, 0.7])  # Shape: (3,)
neg_scores = torch.rand(3, 100)              # Shape: (3, 100)

results = evaluate(pos_scores, neg_scores)
print(f"MRR: {results['mrr']:.4f}")
print(f"Hits@10: {results['hits@10']:.4f}")
```

### **Command Line**
```bash
python evaluation.py --pos-scores pos.npy --neg-scores neg.npy
```

## Advantages

1. **No External Dependencies**: Self-contained implementation
2. **Flexible**: Works with any number of negatives per positive
3. **Efficient**: Vectorized computation using PyTorch
4. **Standard Metrics**: Implements widely-used MRR and Hits@K
5. **Input Validation**: Clear error messages for shape mismatches
6. **Backward Compatible**: Supports both PyTorch tensors and NumPy arrays

## Validation Features

- Input shape validation
- Data type conversion (numpy → torch)
- Comprehensive error messages
- Test suite with edge cases
- Manual calculation verification

The implementation provides the same core functionality as OGB/TGB evaluators but with explicit control over the evaluation process and no external dependencies.

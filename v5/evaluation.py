#!/usr/bin/env python
# eval.py - Manual evaluation script for link prediction tasks with MRR and Hits@K

import torch
import numpy as np


def calculate_mrr_hits(pos_scores, neg_scores, k_values=[1, 3, 10]):
    """
    Calculate Mean Reciprocal Rank (MRR) and Hits@K for link prediction.
    
    Args:
        pos_scores (torch.Tensor): Positive edge scores of shape (num_edges,)
        neg_scores (torch.Tensor): Negative edge scores of shape (num_edges, num_neg)
        k_values (list): List of K values for Hits@K calculation
    
    Returns:
        dict: Dictionary containing MRR and Hits@K metrics
    """
    if pos_scores.dim() != 1:
        raise ValueError(f"pos_scores should be 1D tensor, got shape {pos_scores.shape}")
    if neg_scores.dim() != 2:
        raise ValueError(f"neg_scores should be 2D tensor, got shape {neg_scores.shape}")
    if pos_scores.size(0) != neg_scores.size(0):
        raise ValueError(f"Number of positive and negative samples must match. "
                        f"Got {pos_scores.size(0)} positive and {neg_scores.size(0)} negative samples")
    
    num_edges = pos_scores.size(0)
    num_neg = neg_scores.size(1)
    
    # Expand positive scores to match negative scores shape for comparison
    pos_scores_expanded = pos_scores.unsqueeze(1)  # Shape: (num_edges, 1)
    
    # Calculate how many negative scores are greater than positive scores
    # ranks[i] = number of negatives with score > pos_score[i] + 1 (for 1-indexed ranking)
    ranks = (neg_scores > pos_scores_expanded).sum(dim=1) + 1
    
    # Calculate MRR
    mrr = (1.0 / ranks.float()).mean().item()
    
    # Calculate Hits@K
    hits_at_k = {}
    for k in k_values:
        hits_at_k[f'hits@{k}'] = (ranks <= k).float().mean().item()
    
    return {
        'mrr': mrr,
        **hits_at_k
    }


def evaluate(pos_scores, neg_scores, verbose=True):
    """
    Evaluate link prediction performance using manual MRR and Hits@K calculation.
    
    Args:
        pos_scores: Positive edge scores (torch.Tensor or numpy.ndarray) of shape (num_edges,)
        neg_scores: Negative edge scores (torch.Tensor or numpy.ndarray) of shape (num_edges, num_neg)
        verbose (bool, optional): Whether to print information. Defaults to True.
    
    Returns:
        dict: Evaluation results containing MRR and Hits@K metrics
    """
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(pos_scores, np.ndarray):
        pos_scores = torch.tensor(pos_scores, dtype=torch.float32)
    if isinstance(neg_scores, np.ndarray):
        neg_scores = torch.tensor(neg_scores, dtype=torch.float32)

    # Ensure tensors are of type float
    if not pos_scores.dtype == torch.float32:
        pos_scores = pos_scores.float()
    if not neg_scores.dtype == torch.float32:
        neg_scores = neg_scores.float()

    if verbose:
        print(f"Positive scores shape: {pos_scores.shape}")
        print(f"Negative scores shape: {neg_scores.shape}")
        print(f"Number of edges: {pos_scores.size(0)}")
        print(f"Number of negatives per edge: {neg_scores.size(1)}")

    # Calculate metrics
    results = calculate_mrr_hits(pos_scores, neg_scores)
    
    if verbose:
        print(f"MRR: {results['mrr']:.4f}")
        print(f"Hits@1: {results['hits@1']:.4f}")
        print(f"Hits@3: {results['hits@3']:.4f}")
        print(f"Hits@10: {results['hits@10']:.4f}")

    return results


if __name__ == "__main__":
    # This section only runs when the script is executed directly
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate link prediction performance with manual MRR and Hits@K')
    parser.add_argument('--pos-scores', type=str, required=True,
                        help='Path to positive scores file (numpy array of shape (num_edges,))')
    parser.add_argument('--neg-scores', type=str, required=True,
                        help='Path to negative scores file (numpy array of shape (num_edges, num_neg))')
    args = parser.parse_args()

    # Load scores from files
    pos_scores = np.load(args.pos_scores)
    neg_scores = np.load(args.neg_scores)

    # Perform evaluation
    results = evaluate(pos_scores, neg_scores)
    print(f"Final evaluation results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

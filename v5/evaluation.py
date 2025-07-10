#!/usr/bin/env python
# eval.py - Manual evaluation script for link prediction tasks with MRR and Hits@K

import torch
import numpy as np


def calculate_mrr_hits(pos_score, neg_scores, k_values=[1, 3, 10]):
    """
    Calculate Reciprocal Rank (RR) and Hits@K for a single positive sample against a list of negative samples.
    Prioritizes by length first (shorter is better), then by score (higher is better).

    Args:
        pos_score (float or tuple): The score of the positive sample, either a single value or (score, length) tuple.
        neg_scores (list): A list of scores for negative samples, either single values or (score, length) tuples.
        k_values (list): A list of integers for K in Hits@K.

    Returns:
        dict: A dictionary containing the Reciprocal Rank ('rr') and Hits@K values for the given sample.
    """
    # Check if the scores are in tuple format (score, length)
    is_tuple_format = isinstance(pos_score, tuple) and (not neg_scores or isinstance(neg_scores[0], tuple))
    
    if is_tuple_format:
        pos_score_value, pos_length = pos_score
        
        # If there are no negative scores, the rank is 1 (the best possible)
        if not neg_scores:
            rank = 1
        else:
            # Count how many negative samples are "better" than the positive sample
            # "Better" means: 1) shorter length or 2) same length but higher score
            rank = 1 + sum(1 for s in neg_scores if s[1] < pos_length or (s[1] == pos_length and s[0] > pos_score_value))
    else:
        # Scalar case (pos_score is a single float)
        if not neg_scores:
            rank = 1
        else:
            rank = 1 + sum(1 for s in neg_scores if s > pos_score)

    # Calculate Reciprocal Rank
    rr = 1.0 / rank

    # Calculate Hits@K
    hits_at_k = {f'hits@{k}': 1.0 if rank <= k else 0.0 for k in k_values}

    return {'rr': rr, **hits_at_k}


def evaluate(edge_groups, verbose=True, k_values=[1, 3, 10]):
    """
    Evaluate link prediction performance from grouped scores.

    Args:
        edge_groups (dict): A dictionary where keys are edge identifiers and values are dicts
                            with 'pos_score' and 'neg_scores'.
        verbose (bool, optional): Whether to print information. Defaults to True.
        k_values (list): A list of K values for Hits@K calculation.

    Returns:
        dict: A dictionary with the final evaluation metrics (MRR, Hits@K).
              Groups without a positive score but with negative scores get zero metrics.
              Groups with a positive score but no negative scores get a perfect score.
              Groups with neither positive nor negative scores are ignored.
    """
    results = {
        'rr': 0.0,
        **{f'hits@{k}': 0.0 for k in k_values}
    }
    num_evaluated = 0

    for edge, scores in edge_groups.items():
        pos_score = scores.get('pos_score')
        neg_scores = scores.get('neg_scores', [])

        if pos_score is None:
            if not neg_scores:
                # Skip this edge group if there is no positive score and no negative scores
                if verbose:
                    print(f"Skipping edge {edge} - no positive score and no negative scores")
                continue
            else:
                # Count as failed prediction if there are negative scores but no positive score
                if verbose:
                    print(f"Edge {edge} has no positive score but has negative scores - counting as failed prediction")
                num_evaluated += 1
                # All metrics remain 0 for this edge
                continue

        if verbose:
            print(f"Evaluating edge {edge}: pos_score = {pos_score}, neg_scores = {neg_scores}")

        num_evaluated += 1

        # Calculate metrics for this edge group
        metrics = calculate_mrr_hits(pos_score, neg_scores, k_values=k_values)

        # Aggregate metrics
        results['rr'] += metrics['rr']
        for k in k_values:
            results[f'hits@{k}'] += metrics[f'hits@{k}']

    # Average the results over the number of evaluated edges
    if num_evaluated > 0:
        results['rr'] /= num_evaluated
        for k in k_values:
            results[f'hits@{k}'] /= num_evaluated
    else:
        if verbose:
            print("No edges were evaluated (all skipped).")

    return results


if __name__ == "__main__":
    # This section only runs when the script is executed directly
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate link prediction performance with manual MRR and Hits@K')
    parser.add_argument('--edge-groups', type=str, required=True,
                        help='Path to edge groups file (numpy array of shape (num_edges,))')
    args = parser.parse_args()

    # Load edge groups from file
    edge_groups = np.load(args.edge_groups, allow_pickle=True).item()

    # Perform evaluation
    results = evaluate(edge_groups)
    print(f"Final evaluation results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

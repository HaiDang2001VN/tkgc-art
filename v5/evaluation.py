#!/usr/bin/env python
# eval.py - Manual evaluation script for link prediction tasks with MRR and Hits@K

import torch
import numpy as np


def calculate_mrr_hits(pos_score, neg_scores, k_values=[1, 3, 10]):
    """
    Calculate Reciprocal Rank (RR) and Hits@K for a single positive score against a list of negative scores.

    Args:
        pos_score (float): The score of the positive sample.
        neg_scores (list of float): A list of scores for the negative samples.
        k_values (list): A list of integers for K in Hits@K.

    Returns:
        dict: A dictionary containing the Reciprocal Rank ('rr') and Hits@K values for the given sample.
    """
    # If there are no negative scores, the rank is 1 (the best possible).
    if not neg_scores:
        rank = 1
    else:
        # The rank is 1 + the number of negative scores greater than the positive score.
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

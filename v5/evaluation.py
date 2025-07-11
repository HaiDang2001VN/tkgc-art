#!/usr/bin/env python
# eval.py - Manual evaluation script for link prediction tasks with MRR and Hits@K

import numpy as np
import pandas as pd


def calculate_metrics(group):
    """
    Calculates MRR and Hits@K for a group of predictions for a single query.
    The group contains one 'true_link' and multiple 'false_link' rows.
    A lower 'path_length' is considered a better score.
    """
    true_link = group[group['label'] == 1]
    if true_link.empty or true_link['length'].min() == 0:
        return pd.Series({
            'rank': 0, 'mrr': 0, 'hits@1': 0, 'hits@3': 0, 'hits@10': 0
        })

    # Lower length is better.
    true_path_length = true_link['length'].min()

    # Rank is 1 + number of negative samples with a better (smaller) or equal length.
    # We use '<=' because if scores are tied, the true link does not get the best rank.
    rank = 1 + group[(group['label'] == 0) &
                     (group['length'] < true_path_length)].shape[0]

    mrr = 1.0 / rank
    hits_at_1 = 1.0 if rank <= 1 else 0.0
    hits_at_3 = 1.0 if rank <= 3 else 0.0
    hits_at_10 = 1.0 if rank <= 10 else 0.0

    return pd.Series({
        'rank': rank,
        'mrr': mrr,
        'hits@1': hits_at_1,
        'hits@3': hits_at_3,
        'hits@10': hits_at_10
    })


def evaluate(all_items, verbose=True, k_values=[1, 3, 10]):
    df = pd.DataFrame(all_items)
    groups = df.groupby(['u', "v_pos", "edge_type", "ts"])
    metrics_df = groups.apply(calculate_metrics, include_groups=False).reset_index()

    # Calculate overall metrics
    metrics = ['rank', 'mrr'] + [f'hits@{k}' for k in k_values]
    overall_metrics = metrics_df[metrics].mean()

    if verbose:
        print("Evaluation Results:")
        print(overall_metrics)

    return {
        k: v for k, v in overall_metrics.items()
    }, metrics_df

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

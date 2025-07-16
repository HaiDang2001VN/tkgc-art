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
    if true_link.empty or (true_link['length'].min() == 0):
        min_neg = 0 if group.loc[group['label']
                                 == 1, 'length'].min() > 0 else 1
        return pd.Series({
            'rank': min_neg-1, 'mrr': min_neg, 'hits@1': min_neg, 'hits@3': min_neg, 'hits@10': min_neg
        })

    # # Lower length is better.
    # true_length = true_link['length'].min()

    # # Rank is 1 + number of negative samples with a better (smaller) or equal length.
    # # We use '<=' because if scores are tied, the true link does not get the best rank.
    # rank = 1 + group[(group['label'] == 0) &
    #                  (group['length'] < true_length)].shape[0]
    
    # Score-based comparison
    true_score = true_link['score'].mean()

    # Rank is 1 + number of negative samples with a better (smaller) or equal score.
    # We use '<=' because if scores are tied, the true link does not get the best rank.
    rank = 1 + group[(group['label'] == 0) &
                     (group['score'] < true_score)].shape[0]

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


def evaluate(df, verbose=True, k_values=[1, 3, 10]):
    """
    Evaluate link prediction performance using MRR and Hits@K metrics.
    
    Args:
        df: DataFrame containing prediction data. It must have columns:
            'u', 'v_pos', 'edge_type', 'ts', 'label', 'length', and 'score'.
        verbose: Whether to print evaluation results.
        k_values: List of K values for Hits@K calculation.
        
    Returns:
        Tuple: (overall_metrics_dict, metrics_dataframe)
    """
    if df.empty:
        if verbose:
            print("Warning: Input DataFrame `df` is empty. Returning zero metrics.")
        zero_metrics = {'rank': 0, 'mrr': 0, **{f'hits@{k}': 0 for k in k_values}}
        return zero_metrics, pd.DataFrame()
    
    # Ensure required columns exist, adding them with default values if missing.
    required_cols = ['u', 'v_pos', 'edge_type', 'ts', 'label', 'length', 'score']
    for col in required_cols:
        if col not in df.columns:
            if verbose:
                print(f"Warning: Missing required column '{col}'. Adding with default values.")
            df[col] = 0

    # Group by the unique query identifiers.
    groups = df.groupby(['u', "v_pos", "edge_type", "ts"])
    
    # Apply the existing calculate_metrics function to each group.
    metrics_df = groups.apply(calculate_metrics, include_groups=False).reset_index()

    if metrics_df.empty:
        if verbose:
            print("Warning: No valid groups found for evaluation. Returning zero metrics.")
        zero_metrics = {'rank': 0, 'mrr': 0, **{f'hits@{k}': 0 for k in k_values}}
        return zero_metrics, pd.DataFrame()

    # Calculate overall metrics by averaging the results from each group.
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

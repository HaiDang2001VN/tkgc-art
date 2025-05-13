#!/usr/bin/env python
# eval.py - Evaluation script for OGB and TGB link prediction tasks

import torch
import numpy as np


def get_evaluator(dataset_name):
    """
    Get the appropriate evaluator for the given dataset by checking the dataset_name prefix.
    Args:
        dataset_name (str): Name of the dataset
    Returns:
        evaluator: OGB or TGB Evaluator instance
    """
    if "ogb" in dataset_name:
        try:
            from ogb.linkproppred import Evaluator as OGB_Evaluator
        except ImportError:
            raise ImportError(
                "OGB library not found. Please install it using 'pip install ogb'.")
        return OGB_Evaluator(name=dataset_name)
    else:  # TGB evaluator
        try:
            from tgb.linkproppred.evaluate import Evaluator as TGB_Evaluator
        except ImportError:
            raise ImportError(
                "TGB library not found. Please install it using 'pip install py-tgb'.")
        return TGB_Evaluator(name=dataset_name)


def evaluate(dataset_name, pos_scores, neg_scores, verbose=True):
    """
    Evaluate link prediction performance using the appropriate evaluator.
    Args:
        dataset_name (str): Name of the dataset
        pos_scores: Positive edge scores (torch.Tensor or numpy.ndarray)
        neg_scores: Negative edge scores (torch.Tensor or numpy.ndarray)
        verbose (bool, optional): Whether to print information. Defaults to True.
    Returns:
        dict: Evaluation results
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

    # Get the evaluator
    evaluator = get_evaluator(dataset_name)

    # Check tensor shapes based on metric
    eval_metric = evaluator.eval_metric if hasattr(
        evaluator, 'eval_metric') else "unknown"

    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"Evaluation metric: {eval_metric}")
        print(f"Positive scores shape: {pos_scores.shape}")
        print(f"Negative scores shape: {neg_scores.shape}")

    # Validate input shapes
    if "mrr" in eval_metric and neg_scores.ndim == 1:
        raise ValueError(f"For MRR evaluation, neg_scores should be a 2D tensor. "
                         f"Got shape {neg_scores.shape} instead.")

    # Create input dictionary for evaluation
    input_dict = {
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores
    }

    # Perform evaluation
    results = evaluator.eval(input_dict)

    # Return results directly without saving to file
    return results


if __name__ == "__main__":
    # This section only runs when the script is executed directly
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate link prediction performance')
    parser.add_argument('--dataset', type=str,
                        required=True, help='Dataset name')
    parser.add_argument('--pos-scores', type=str, required=True,
                        help='Path to positive scores file')
    parser.add_argument('--neg-scores', type=str, required=True,
                        help='Path to negative scores file')
    args = parser.parse_args()

    # Load scores from files
    pos_scores = np.load(args.pos_scores)
    neg_scores = np.load(args.neg_scores)

    # Perform evaluation
    results = evaluate(args.dataset, pos_scores, neg_scores)
    print(f"Evaluation results: {results}")

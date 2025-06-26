#!/usr/bin/env python3
"""
Test script for manual evaluation functionality.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append('/Users/haidang/Library/CloudStorage/GoogleDrive-danghailamhuynh@gmail.com/My Drive/Study/HCMUS/Grad/Master/Thesis/Codes/code/src/v5')

def test_manual_evaluation():
    """Test the manual evaluation implementation."""
    try:
        from evaluation import evaluate, calculate_mrr_hits
        
        print("🧪 Testing manual evaluation functionality...")
        
        # Create test data
        # 5 edges, 10 negatives per edge
        num_edges = 5
        num_neg = 10
        
        # Positive scores: [0.9, 0.8, 0.7, 0.6, 0.5]
        pos_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32)
        
        # Negative scores: random values between 0 and 1
        torch.manual_seed(42)  # For reproducible results
        neg_scores = torch.rand(num_edges, num_neg, dtype=torch.float32)
        
        print(f"📊 Test data:")
        print(f"  Positive scores shape: {pos_scores.shape}")
        print(f"  Negative scores shape: {neg_scores.shape}")
        print(f"  Positive scores: {pos_scores.tolist()}")
        print(f"  Sample negative scores (first edge): {neg_scores[0].tolist()[:5]}...")
        
        # Test the evaluation function
        print(f"\n🔍 Running evaluation...")
        results = evaluate(pos_scores, neg_scores, verbose=True)
        
        print(f"\n✅ Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Validate results make sense
        assert 0 <= results['mrr'] <= 1, f"MRR should be between 0 and 1, got {results['mrr']}"
        assert 0 <= results['hits@1'] <= 1, f"Hits@1 should be between 0 and 1, got {results['hits@1']}"
        assert 0 <= results['hits@3'] <= 1, f"Hits@3 should be between 0 and 1, got {results['hits@3']}"
        assert 0 <= results['hits@10'] <= 1, f"Hits@10 should be between 0 and 1, got {results['hits@10']}"
        
        # Hits@K should be non-decreasing
        assert results['hits@1'] <= results['hits@3'], "Hits@1 should be <= Hits@3"
        assert results['hits@3'] <= results['hits@10'], "Hits@3 should be <= Hits@10"
        
        print(f"\n🎯 Manual calculation verification:")
        # Manually verify for first edge
        pos_score_0 = pos_scores[0].item()
        neg_scores_0 = neg_scores[0]
        rank_0 = (neg_scores_0 > pos_score_0).sum().item() + 1
        print(f"  Edge 0: pos_score={pos_score_0:.3f}, rank={rank_0}, reciprocal_rank={1.0/rank_0:.4f}")
        
        # Test with numpy arrays
        print(f"\n🔄 Testing with numpy arrays...")
        pos_scores_np = pos_scores.numpy()
        neg_scores_np = neg_scores.numpy()
        results_np = evaluate(pos_scores_np, neg_scores_np, verbose=False)
        
        # Results should be the same
        for metric in results:
            assert abs(results[metric] - results_np[metric]) < 1e-6, f"Results differ for {metric}"
        
        print(f"✅ Numpy array test passed!")
        
        # Test edge cases
        print(f"\n🧩 Testing edge cases...")
        
        # Case: All negatives have lower scores (perfect ranking)
        perfect_pos = torch.tensor([1.0, 1.0, 1.0])
        perfect_neg = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        perfect_results = evaluate(perfect_pos, perfect_neg, verbose=False)
        
        print(f"  Perfect case - MRR: {perfect_results['mrr']:.4f}, Hits@1: {perfect_results['hits@1']:.4f}")
        assert perfect_results['mrr'] == 1.0, "Perfect case should have MRR = 1.0"
        assert perfect_results['hits@1'] == 1.0, "Perfect case should have Hits@1 = 1.0"
        
        # Case: All negatives have higher scores (worst ranking)
        worst_pos = torch.tensor([0.1, 0.1, 0.1])
        worst_neg = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [0.2, 0.3, 0.4]])
        worst_results = evaluate(worst_pos, worst_neg, verbose=False)
        
        print(f"  Worst case - MRR: {worst_results['mrr']:.4f}, Hits@1: {worst_results['hits@1']:.4f}")
        assert worst_results['hits@1'] == 0.0, "Worst case should have Hits@1 = 0.0"
        
        print(f"\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing manual evaluation implementation...")
    success = test_manual_evaluation()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

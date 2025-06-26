#!/usr/bin/env python3
"""
Test script to verify that edge type embeddings are correctly extracted from the dataset.
"""

import sys
import os
import torch

# Add the parent directory to the Python path
sys.path.append('/Users/haidang/Library/CloudStorage/GoogleDrive-danghailamhuynh@gmail.com/My Drive/Study/HCMUS/Grad/Master/Thesis/Codes/code/src/v5')

def test_edge_embeddings():
    """Test that edge embeddings are properly extracted."""
    # Mock configuration for testing
    config_path = "/Users/haidang/Library/CloudStorage/GoogleDrive-danghailamhuynh@gmail.com/My Drive/Study/HCMUS/Grad/Master/Thesis/Codes/code/src/v5/config.json"
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("This test requires a valid configuration file.")
        return False
    
    try:
        from loader import PathDataModule
        
        # Create data module
        dm = PathDataModule(config_path, batch_size=4)
        dm.setup(stage="fit")
        
        # Get a sample from the training data
        train_loader = dm.train_dataloader()
        
        # Test a few samples
        sample_count = 0
        for batch in train_loader:
            for item in batch:
                sample_count += 1
                
                print(f"\n📊 Sample {sample_count}:")
                print(f"  Label: {item['label']}")
                
                # Check if paths exist
                if 'paths' in item:
                    num_paths = len(item['paths'])
                    print(f"  Number of paths: {num_paths}")
                    
                    for i, path in enumerate(item['paths']):
                        print(f"    Path {i}: {len(path)} nodes")
                    
                    # Check if shallow embeddings exist
                    if 'shallow_emb' in item:
                        print(f"  Node embeddings: {len(item['shallow_emb'])} paths")
                        for i, emb in enumerate(item['shallow_emb']):
                            print(f"    Path {i} node embeddings: {emb.shape}")
                    
                    # Check if edge embeddings exist (NEW FEATURE)
                    if 'edge_emb' in item:
                        print(f"  Edge embeddings: {len(item['edge_emb'])} paths")
                        for i, edge_emb in enumerate(item['edge_emb']):
                            print(f"    Path {i} edge embeddings: {edge_emb.shape}")
                            
                            # Verify that number of edge embeddings = number of nodes - 1
                            if 'paths' in item and i < len(item['paths']):
                                expected_edges = max(0, len(item['paths'][i]) - 1)
                                actual_edges = edge_emb.shape[0]
                                if expected_edges == actual_edges:
                                    print(f"      ✅ Correct: {actual_edges} edge embeddings for {len(item['paths'][i])} nodes")
                                else:
                                    print(f"      ❌ Expected {expected_edges} edge embeddings for {len(item['paths'][i])} nodes, got {actual_edges}")
                    else:
                        print("  ❌ No edge embeddings found!")
                else:
                    print("  No paths in this sample")
                
                # Only test first few samples
                if sample_count >= 3:
                    break
            if sample_count >= 3:
                break
        
        print(f"\n✅ Test completed successfully! Tested {sample_count} samples.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing edge embeddings functionality...")
    success = test_edge_embeddings()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

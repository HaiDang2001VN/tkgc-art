import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from model import TransformerModel
from dist import NTPLoss
from data import TemporalDataset

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_iterations = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        paths = batch['paths'].to(device)
        masks = batch['masks'].to(device) 
        labels = batch['labels'].squeeze(-1).float().to(device)
        tokens = batch['tokens'].to(device)
        
        batch_size, seq_len, embed_dim = tokens.shape
        
        # Create input and target sequences
        src_tokens = tokens[:, :-1]  # All but last token
        tgt_tokens = tokens[:, 1:]   # All but first token
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            src_tokens.permute(1, 0, 2),  # [seq_len-1, batch, embed_dim]
            tgt_tokens.permute(1, 0, 2)   # [seq_len-1, batch, embed_dim]
        )
        
        # Calculate per-sample losses without aggregation
        losses = loss_fn(
            outputs.permute(1, 0, 2),  # [batch, seq_len-1, embed_dim]
            tgt_tokens,                # [batch, seq_len-1, embed_dim]
            labels
        )
        
        # Compute mean loss for backpropagation
        loss = losses.mean()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track loss per iteration (not multiplied by batch size)
        total_loss += loss.item()
        total_iterations += 1
        
    return total_loss / total_iterations

def evaluate(model, dataset, device):
    """Evaluate the model on the test dataset"""
    model.eval()
    
    result_dict = {
        "y_pred_pos": [],
        "y_pred_neg": [],
    }
    
    with torch.no_grad():
        for data in tqdm(dataset, desc="Testing"):
            # Get the data
            edge = data["central_edge"].to(device)
            paths = data['paths'].to(device)
            masks = data['masks'].to(device)
            labels = data['labels'].squeeze(1).to(device)
            tokens = data['tokens'].to(device)
            
            if paths.shape[0] == 0:
                continue
                
            batch_size, seq_len, embed_dim = tokens.shape
            
            # Create input and target sequences for scoring
            src_tokens = tokens[:, :-1]
            tgt_tokens = tokens[:, 1:]
            
            # Get model outputs
            outputs = model(
                src_tokens.permute(1, 0, 2),
                tgt_tokens.permute(1, 0, 2)
            )
            
            # Calculate per-sample losses without aggregation
            losses = loss_fn(
                outputs.permute(1, 0, 2),
                tgt_tokens,
                torch.ones_like(labels),  # Use all-ones to get raw distances
            )
            
            # Compute percentile value of central edge according to losses
            # Argsort the losses
            sorted_indices = torch.argsort(losses)
            
            # Get the index of the central edge in sorted losses
            central_pos = edge[sorted_indices].argmax()
            
            # Get the percentile of central edge
            percentile = (central_pos + 1) / sorted_indices.size(0)
            
            iteration += 1
            
            if iteration % steps_per_checkpoint == 0:
                print(f'[Iteration #{iteration}] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
                print(f'Hit rate: {checkpoint_pos_found/max(1, checkpoint_pos_count):.5f} ({checkpoint_pos_found}/{checkpoint_pos_count}), Cumulative hit rate: {positive_count/max(1, total_pos_count):.5f} ({positive_count}/{total_pos_count})\n')
                checkpoint_pos_count = checkpoint_pos_found = 0
    
    print(f'[Final] Total number of samples: {total}, Positive: {positive_count}, Negative: {negative_count}')
    print(f'Cumulative hit rate: {positive_count/max(1, total_pos_count):.5f} ({positive_count}/{total_pos_count})\n')
    
    return positive_count / max(1, total_pos_count)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate temporal graph model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--model_path', type=str, help='Path to saved model for evaluation')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Set device
    device = torch.device(config['training']['devices'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TemporalDataset(config, k=3, m_d=50)
    train_dataset.split = 'train'
    
    test_dataset = TemporalDataset(config, k=3, m_d=50)
    test_dataset.split = 'test'
    
    # Create data loader for training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda batch: {
            'paths': torch.stack([item['paths'] for item in batch]),
            'masks': torch.stack([item['masks'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'positives_count': sum(item['positives_count'] for item in batch),
            'positives_found': sum(item['positives_found'] for item in batch)
        }
    )
    
    # Get embed_dim from the dataset
    embed_dim = train_dataset.embedding_manager.node_dim
    
    # Initialize model
    model = TransformerModel(
        vocab_size=train_dataset.graph['num_nodes'],
        embed_dim=embed_dim,
        nhead=config['models']['DGT']['nhead'],
        num_encoder_layers=config['models']['DGT']['num_layers'] // 2,
        num_decoder_layers=config['models']['DGT']['num_layers'] // 2,
        dim_feedforward=config['models']['DGT']['dim_ffn'],
        dropout=0.1,
        max_seq_length=config['training']['k_hops'] + 1
    ).to(device)
    
    # Initialize loss function and optimizer
    # Creating a modified loss function that can return non-aggregated losses
    loss_fn = NTPLoss(margin=1.0, distance_metric=config['training'].get('similarity_type', 'l2'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load model if evaluating only
    if args.eval_only and args.model_path:
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(model, test_dataset, device)
        return
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Training loss: {train_loss:.6f}")
        
        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")
        
        # Evaluate every few epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print("Evaluating model...")
            hit_rate = evaluate(model, test_dataset, device)
            print(f"Hit Rate: {hit_rate:.5f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
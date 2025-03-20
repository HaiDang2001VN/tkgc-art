# train.py
import os
import json
import argparse
import shutil
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import wandb
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.metrics import roc_auc_score
from ogb.linkproppred import Evaluator

from data import TemporalDataset
from models import DGT, PGT, TemporalEmbeddingManager
from loss import compute_dgt_loss, compute_pgt_loss, adaptive_update_multi_layer

class SyncedGraphDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load dataset immediately (previously in prepare_data)
        print("Loading dataset...")
        self.main_dataset = TemporalDataset(
            root=self.config['data']['path'],
            config=self.config
        )
        print("Dataset loaded.")
        print("Number of nodes: ", self.main_dataset.num_nodes)
        self.num_nodes = self.main_dataset.num_nodes
        
        # Initialize datasets (previously in setup)
        print("Initializing dataset splits...")
        self.train_dataset = self.main_dataset.clone_for_split('train')
        self.val_dataset = self.main_dataset.clone_for_split('valid')
        self.test_dataset = self.main_dataset.clone_for_split('test')
        print("Dataset splits initialized.")

    def train_dataloader(self):
        print("Creating train dataloader...")
        loader = self.create_dataloader(self.train_dataset)
        print(f"Train dataloader created with {len(self.train_dataset)} batches")
        return loader

    def val_dataloader(self):
        print("Creating val dataloader...")
        loader = self.create_dataloader(self.val_dataset)
        print(f"Validation dataloader created with {len(self.val_dataset)} batches")
        return loader

    def test_dataloader(self):
        print("Creating test dataloader...")
        loader = self.create_dataloader(self.test_dataset)
        print(f"Test dataloader created with {len(self.test_dataset)} batches")
        return loader

    def create_dataloader(self, dataset):
        print("Creating dataloader...")
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.config['training']['num_workers'],
            persistent_workers=self.config['training'].get('persistent_workers', True),
            pin_memory=self.config['training'].get('pin_memory', True),
            shuffle=False
        )

class UnifiedTrainer(L.LightningModule):
    def __init__(self, config, num_nodes):
        super().__init__()
        self.config = config
        print("Creating models with config: ", config)
        self.dgt = DGT(config['models']['DGT'])
        self.predictive = config['training']['predictive']
        if self.predictive:
            self.pgt = PGT(config['models']['PGT'])
        else:
            print("PGT not created.")
        print("Models created.")
        print("Creating embedding manager with num_nodes: ", num_nodes)
        print("node_dim = ", config['models']['DGT']['d_model'])
        self.emb_manager = TemporalEmbeddingManager(
            num_nodes=num_nodes,
            node_dim=config['models']['DGT']['d_model']
        )
        print("Embedding manager created.")
        self.val_emb_manager = None  # Validation manager placeholder
        
        # Track original dimensions for validation copies
        self._num_nodes = num_nodes
        self._node_dim = config['models']['DGT']['d_model']
        self.last_layer = max(self.dgt.intermediate_layers.keys())
        print("Last layer: ", self.last_layer)

    def on_train_epoch_start(self):
        """Reset embeddings at epoch start"""
        self.emb_manager.reset()

    def on_validation_start(self):
        """Called by Lightning before validation begins"""
        self.on_start()
        print("Starting validation phase")
        
        # Create validation embedding manager
        self.val_emb_manager = TemporalEmbeddingManager(
            self._num_nodes,
            self._node_dim
        )
        self.val_emb_manager.load_state_dict(
            self.emb_manager.state_dict()
        )
        self.val_emb_manager.to(self.device)

    def step(self, batch, emb_manager=None):
        """
        Common processing logic for both training and validation steps
        
        Args:
            batch: The input batch data
            emb_manager: Embedding manager to use (defaults to self.emb_manager if None)
            
        Returns:
            A dictionary containing losses and computed values
        """
        if emb_manager is None:
            emb_manager = self.emb_manager
        
        # Get similarity type from config
        similarity_type = self.config['training'].get('similarity_type', 'inner')
        
        # Process DGT embeddings with similarity type
        dgt_loss, mean_diff = self._dgt_forward(batch['dgt'], emb_manager, similarity_type)
        
        if self.predictive:
            # Process PGT embeddings with similarity type
            pgt_loss, pgt_scores, raw_z_score = self._pgt_forward(batch['pgt'], emb_manager, similarity_type)
        else:
            pgt_loss = torch.tensor(0.0)
            pgt_scores = {}
            raw_z_score = torch.tensor(0.0)

        # Handle timestamp transitions
        if batch['meta']['is_group_end']:
            emb_manager.transition_timestamp()

        # Compute total loss
        total_loss = dgt_loss + pgt_loss
        
        return {
            'dgt_loss': dgt_loss,
            'pgt_loss': pgt_loss,
            'total_loss': total_loss,
            'pgt_scores': pgt_scores,
            'mean_diff': mean_diff,
            'raw_z_score': raw_z_score
        }

    def training_step(self, batch, batch_idx):
        # Use the common step function
        results = self.step(batch)
        
        # Extract timestamp from batch (mean of all edge timestamps in batch)
        batch_timestamp = batch['edge_time'].float().mean().item()
        
        # Log with train prefix including new metrics
        self.log_dict({
            'train_total_loss': results['total_loss'],
            'train_dgt_loss': results['dgt_loss'],
            'train_pgt_loss': results['pgt_loss'],
            'train_dgt_mean_diff': results['mean_diff'],
            'train_pgt_z_score': results['raw_z_score'],
            'train_timestamp': batch_timestamp
        }, prog_bar=True, sync_dist=True, batch_size=self.config['training']['batch_size'])
        
        return results['total_loss']
    
    def on_train_epoch_end(self):
        """Save checkpoints for models and embedding manager at the end of each training epoch"""
        epoch = self.current_epoch
        
        # Check if we should save a checkpoint this epoch
        ckpt_freq = self.config['training'].get('ckpt_freq', 1)
        if epoch % ckpt_freq != 0:
            return
        
        # Get checkpoint parameters from config
        ckpt_dir = self.config['training'].get('ckpt_dir', '../checkpoints')
        ckpt_name = self.config['training'].get('ckpt_name', 'model')
        ckpt_keep = self.config['training'].get('ckpt_keep', 1)
        
        # Create epoch-specific checkpoint directory
        checkpoint_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save DGT model
        dgt_path = os.path.join(checkpoint_dir, f"{ckpt_name}_dgt.pt")
        torch.save(self.dgt.state_dict(), dgt_path)
        
        # Save PGT model if in predictive mode
        if self.predictive:
            pgt_path = os.path.join(checkpoint_dir, f"{ckpt_name}_pgt.pt")
            torch.save(self.pgt.state_dict(), pgt_path)
        
        # Save embedding manager
        emb_path = os.path.join(checkpoint_dir, f"{ckpt_name}_embedding_manager.pt")
        torch.save(self.emb_manager.state_dict(), emb_path)
        
        print(f"Saved model and embedding manager checkpoints for epoch {epoch} to {checkpoint_dir}")
        
        # Implement checkpoint retention logic (keep only the most recent ckpt_keep checkpoints)
        if ckpt_keep > 0:
            # List all epoch checkpoint directories
            all_ckpts = []
            for dirname in os.listdir(ckpt_dir):
                if dirname.startswith('epoch_') and os.path.isdir(os.path.join(ckpt_dir, dirname)):
                    try:
                        epoch_num = int(dirname.split('_')[1])
                        all_ckpts.append((epoch_num, dirname))
                    except (IndexError, ValueError):
                        pass
            
            # Sort by epoch number (descending)
            all_ckpts.sort(reverse=True)
            
            # Remove older checkpoints beyond the keep limit
            for epoch_num, dirname in all_ckpts[ckpt_keep:]:
                try:
                    import shutil
                    shutil.rmtree(os.path.join(ckpt_dir, dirname))
                    print(f"Removed old checkpoint: {dirname}")
                except Exception as e:
                    print(f"Error removing old checkpoint {dirname}: {e}")

    def validation_step(self, batch, batch_idx):
        # Use the step function with validation embedding manager
        results = self.step(batch, self.val_emb_manager)
        
        # Extract timestamp from batch (mean of all edge timestamps in batch)
        batch_timestamp = batch['edge_time'].float().mean().item()
        
        # Log with validation prefix including new metrics
        self.log_dict({
            'val_total_loss': results['total_loss'],
            'val_dgt_loss': results['dgt_loss'],
            'val_pgt_loss': results['pgt_loss'],
            'val_dgt_mean_diff': results['mean_diff'],
            'val_pgt_z_score': results['raw_z_score'],
            'val_timestamp': batch_timestamp
        }, prog_bar=True, sync_dist=True, batch_size=self.config['training']['batch_size'])
        
        # For evaluation metrics, return scores and labels
        return {
            'loss': results['total_loss'],
            'pgt_scores': results['pgt_scores'],
            'labels': batch.get('labels', None)
        }

    def _dgt_forward(self, batch, emb_manager=None, similarity_type='inner'):
        """
        Forward pass for DGT model using adaptive weighted embeddings
        
        Args:
            batch: List of dictionaries with graph data
            emb_manager: Embedding manager for retrieving and updating node embeddings
            similarity_type: Type of similarity to use ('inner' or 'cosine')
            
        Returns:
            Mean loss across batch (scalar)
            Mean difference metric
        """
        if emb_manager is None:
            emb_manager = self.emb_manager
            
        losses = []
        mean_diffs = []  # Track mean differences
        for item in batch:
            nodes = item['nodes']  # List of node IDs
            adj = item['adj']  # Adjacency matrix [num_nodes, num_nodes]
            dist = item['dist']  # Distance weights [num_nodes]
            mask = item['central_mask']  # Boolean mask for central nodes [num_nodes]
            
            # Only for training or when we know it's a positive edge
            is_positive = True
            if 'label' in item:
                is_positive = item['label'] > 0
            
            # Get old embeddings [num_nodes, dim]
            old_emb = emb_manager.get_embedding(nodes)
            
            # Get intermediate outputs from transformer
            # Returns dictionary {layer_idx: tensor of shape [1, num_nodes, dim]}
            intermediate = self.dgt(old_emb.unsqueeze(0))
            
            # Apply adaptive weighting to all layers and convert to tensor format
            weighted_embs, layer_weight_tensor, layer_indices = adaptive_update_multi_layer(
                old_emb, 
                intermediate, 
                dist,
                self.dgt.intermediate_layers
            )
            
            # Compute loss using the adaptively weighted embeddings
            # Returns both loss and mean diff
            result = compute_dgt_loss(
                weighted_embs,  # [num_layers, num_nodes, dim]
                adj,  # [num_nodes, num_nodes]
                layer_weight_tensor,  # [num_layers]
                similarity_type  # Pass similarity type
            )
            
            if result is not None:
                loss_val, mean_diff = result
                
                # For positive edges, update the embeddings using the last layer
                if is_positive:
                    # Find the index of the last layer in our ordered layers
                    last_layer_idx = layer_indices.index(self.last_layer)
                    # Use the weighted embeddings for the last layer [num_nodes, dim]
                    last_layer_weighted = weighted_embs[last_layer_idx]
                    
                    for i, node in enumerate(nodes):
                        emb_manager.update_embeddings(node, last_layer_weighted[i])
                
                # print(loss_val)
                losses.append(loss_val)
                mean_diffs.append(mean_diff)
            
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(mean_diffs))

    def _pgt_forward(self, batch, emb_manager=None, similarity_type='inner'):
        if emb_manager is None:
            emb_manager = self.emb_manager
            
        final_embs = []
        masks = []
        labels = []  # Store labels for each item
        
        for item in batch:
            nodes = item['nodes']
            mask = item['central_mask']
            
            # Store label if available (for validation/test)
            if 'label' in item:
                labels.append(item['label'])
            
            old_emb = emb_manager.get_embedding(nodes)
            final_out = self.pgt(old_emb.unsqueeze(0))
            res = final_out.squeeze(0)
            final_embs.append(res)
            masks.append(mask)

        # Compute PGT loss and get edge scores and raw z-score
        pgt_loss, edge_scores, raw_z_score = compute_pgt_loss(final_embs, masks, self.config['models']['PGT']['d_model'], similarity_type)
        
        # Return labels along with scores if available
        if labels:
            return pgt_loss, {'scores': edge_scores, 'labels': labels}, raw_z_score
        else:
            return pgt_loss, {'scores': edge_scores}, raw_z_score

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                              lr=self.config['training']['lr'])

    def on_validation_epoch_end(self):
        """Calculate and log link prediction metrics using OGB evaluator"""
        # Skip evaluation if not in predictive mode
        if not self.predictive:
            self.log('val_status', 0.0, prog_bar=True)  # Indicator that evaluation was skipped
            print("Skipping validation metrics: predictive mode is disabled")
            return
            
        # Collect all outputs from validation steps
        outputs = self.validation_step_outputs
        
        # Extract all scores and labels
        all_scores = []
        all_labels = []
        
        for output in outputs:
            scores_dict = output.get('pgt_scores', {})
            # Check if scores_dict has both required keys
            if (scores_dict and 'scores' in scores_dict and 'labels' in scores_dict and 
                scores_dict['labels'] is not None and scores_dict['scores'] is not None):
                
                scores = scores_dict['scores']
                labels = scores_dict['labels']
                
                # Extend our lists with this batch's data
                all_scores.extend([score.item() for score in scores])
                all_labels.extend([label.item() for label in labels])
        
        # Check if we have enough data to calculate metrics
        if len(all_scores) == 0 or len(all_labels) == 0:
            print("WARNING: No validation data available for metrics calculation")
            self.log('val_no_data', -1.0, prog_bar=True)
            return
            
        # Convert to numpy arrays
        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels)
        
        # Separate positive and negative edges
        pos_indices = labels_np == 1
        neg_indices = labels_np == 0
        
        pos_edge_scores = scores_np[pos_indices]
        neg_edge_scores = scores_np[neg_indices]
        
        # Check if we have both positive and negative examples
        if len(pos_edge_scores) == 0 or len(neg_edge_scores) == 0:
            print(f"WARNING: Missing examples for evaluation. Positive: {len(pos_edge_scores)}, Negative: {len(neg_edge_scores)}")
            self.log('val_imbalanced_data', -1.0, prog_bar=True)
            return
        
        # For OGB evaluator
        y_pred_pos = torch.tensor(pos_edge_scores)
        y_pred_neg = torch.tensor(neg_edge_scores)
        
        # Create evaluator input
        input_dict = {
            "y_pred_pos": y_pred_pos,
            "y_pred_neg": y_pred_neg,
        }
        
        # Use OGB evaluator
        try:
            # Get dataset name from config
            d_name = self.config['data'].get('name', 'ogbl-collab')  # Default to collab if not specified
            evaluator = Evaluator(name=d_name)
            result_dict = evaluator.eval(input_dict)
            
            # Log results
            for metric_name, value in result_dict.items():
                self.log(f'val_{metric_name}', value, prog_bar=True)
                
            # Also calculate AUC as a common metric
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones_like(pos_edge_scores), np.zeros_like(neg_edge_scores)])
            y_pred = np.concatenate([pos_edge_scores, neg_edge_scores])
            
            auc_score = roc_auc_score(y_true, y_pred)
            self.log('val_auc', auc_score, prog_bar=True)
            
            # Log data statistics for debugging
            self.log('val_pos_samples', len(pos_edge_scores))
            self.log('val_neg_samples', len(neg_edge_scores))
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            self.log('val_error', -1.0, prog_bar=True)

    def on_validation_epoch_start(self):
        """Initialize collection for validation outputs"""
        # Fix: Don't call parent method that doesn't exist
        # super().on_validation_start()  # This was incorrect
        self.validation_step_outputs = []

    def validation_step_end(self, outputs):
        """Collect validation step outputs"""
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_start(self):
        """
        Common setup code for all start hooks.
        This ensures components are on the correct device.
        """
        # Move embedding manager to the same device as the model
        self.emb_manager.to(self.device)
        
        # Log device information for debugging
        print(f"Components running on device: {self.device}")
        print(f"DGT is on device: {next(self.dgt.parameters()).device}")
        if self.predictive:
            print(f"PGT is on device: {next(self.pgt.parameters()).device}")
        print(f"Embedding manager moved to device: {self.device}")

    def on_train_start(self):
        """Called by Lightning before training begins"""
        self.on_start()
        print("Starting training phase")

    def on_test_start(self):
        """Called by Lightning before test begins"""
        self.on_start()
        print("Starting test phase")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train temporal graph models')
    parser.add_argument('--config', type=str, default='src/config.json',
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration from specified path
    with open(args.config) as f:
        config = json.load(f)
        
    print("Config: ", config)
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'temporal_graph_learning')
    log_dir = os.path.join("logs", f"{experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Extract wandb configuration or use defaults
    wandb_config = {
        "project": config['training'].get('wandb_project', "temporal-graph-learning"),
        "name": config['training'].get('wandb_name', f"{experiment_name}_{timestamp}"),
        "group": config['training'].get('wandb_group', None),
        "tags": config['training'].get('wandb_tags', []),
        "log_model": config['training'].get('wandb_log_model', "all"),
        "save_dir": log_dir,
    }
    
    # Create wandb logger
    logger = WandbLogger(**wandb_config)
    
    # Log hyperparameters to wandb
    logger.log_hyperparams(config)
    
    print(f"Logging metrics to wandb project: {wandb_config['project']}, run: {wandb_config['name']}")
    
    print("Creating datamodule...")
    datamodule = SyncedGraphDataModule(config)
    print("Preparing data...")
    datamodule.prepare_data()
    print("Creating model...")
    model = UnifiedTrainer(config, datamodule.num_nodes)
    print("Training...")
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        log_every_n_steps=config['training']["log_freq"],
        logger=logger,  # Use wandb logger
        enable_checkpointing=True,
        default_root_dir=log_dir
    )
    print("Fitting model...")
    trainer.fit(model, datamodule=datamodule)
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Optionally save configuration with the results
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Close wandb run
    wandb.finish()

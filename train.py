# train.py
import lightning as L
import torch
from torch.utils.data import DataLoader
from data import TemporalDataset
from models import DGT, PGT, TemporalEmbeddingManager
from loss import compute_dgt_loss, compute_pgt_loss, adaptive_update
from ogb.linkproppred import Evaluator
import numpy as np
from lightning.pytorch.loggers import CSVLogger  # Import CSVLogger
import os
from datetime import datetime

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
        
        # Process DGT embeddings
        dgt_loss = self._dgt_forward(batch['dgt'], emb_manager)
        
        if self.predictive:
            # Process PGT embeddings
            pgt_loss, pgt_scores = self._pgt_forward(batch['pgt'], emb_manager)
        else:
            pgt_loss = torch.tensor(0.0)
            pgt_scores = {}

        # Handle timestamp transitions
        if batch['meta']['is_group_end']:
            emb_manager.transition_timestamp()

        # Compute total loss
        total_loss = dgt_loss + pgt_loss
        
        return {
            'dgt_loss': dgt_loss,
            'pgt_loss': pgt_loss,
            'total_loss': total_loss,
            'pgt_scores': pgt_scores
        }

    def training_step(self, batch, batch_idx):
        # Use the common step function
        results = self.step(batch)
        
        # Extract timestamp from batch (mean of all edge timestamps in batch)
        batch_timestamp = batch['edge_time'].float().mean().item()
        
        # Log with train prefix
        self.log_dict({
            'train_total_loss': results['total_loss'],
            'train_dgt_loss': results['dgt_loss'],
            'train_pgt_loss': results['pgt_loss'],
            'train_timestamp': batch_timestamp  # Add timestamp logging
        }, prog_bar=True, sync_dist=True, batch_size=self.config['training']['batch_size'])
        
        return results['total_loss']

    def validation_step(self, batch, batch_idx):
        # Use the step function with validation embedding manager
        results = self.step(batch, self.val_emb_manager)
        
        # Extract timestamp from batch (mean of all edge timestamps in batch)
        batch_timestamp = batch['edge_time'].float().mean().item()
        
        # Log with validation prefix
        self.log_dict({
            'val_total_loss': results['total_loss'],
            'val_dgt_loss': results['dgt_loss'],
            'val_pgt_loss': results['pgt_loss'],
            'val_timestamp': batch_timestamp  # Add timestamp logging
        }, prog_bar=True, sync_dist=True, batch_size=self.config['training']['batch_size'])
        
        # For evaluation metrics, return scores and labels
        return {
            'loss': results['total_loss'],
            'pgt_scores': results['pgt_scores'],
            'labels': batch.get('labels', None)
        }

    def _dgt_forward(self, batch, emb_manager=None):
        if emb_manager is None:
            emb_manager = self.emb_manager
            
        losses = []
        for item in batch:
            nodes = item['nodes']
            adj = item['adj']
            dist = item['dist']
            mask = item['central_mask']
            
            # Only for training or when we know it's a positive edge
            # During validation/test we check for labels if available
            is_positive = True
            if 'label' in item:
                is_positive = item['label'] > 0
            
            # print("Nodes shape: ", nodes.shape)
            # print("Adj shape: ", adj.shape)
            old_emb = emb_manager.get_embedding(nodes)
            # print("Old emb shape: ", old_emb.shape)
            intermediate = self.dgt(old_emb.unsqueeze(0))
            
            # Only update embeddings for positive items
            if is_positive:
                new_emb = adaptive_update(
                    old_emb,
                    intermediate[self.last_layer].squeeze(0),
                    dist
                )
                
                for i, node in enumerate(nodes):
                    emb_manager.update_embeddings(node, new_emb[i])
            
            loss = compute_dgt_loss(intermediate, adj, 
                                  self.dgt.intermediate_layers)
            # print("Loss: ", loss.item())
            losses.append(loss)
            
        return torch.mean(torch.stack(losses))

    def _pgt_forward(self, batch, emb_manager=None):
        if emb_manager is None:
            emb_manager = self.emb_manager
            
        final_embs = []
        masks = []
        labels = []  # Store labels for each item
        
        for item in batch:
            nodes = item['nodes']
            mask = item['central_mask']
            
            # print("Nodes shape: ", nodes.shape)
            # print("Nodes: ", nodes)
            # print("Mask shape: ", mask.shape)
            # print("Mask: ", mask)
            
            # Store label if available (for validation/test)
            if 'label' in item:
                labels.append(item['label'])
            
            old_emb = emb_manager.get_embedding(nodes)
            # print("Old emb shape: ", old_emb.shape)
            # print("Emb: ", old_emb.unsqueeze(0).shape)
            final_out = self.pgt(old_emb.unsqueeze(0))
            # print("Final out shape: ", final_out.shape)
            res = final_out.squeeze(0)
            # print("Final out shape: ", res.shape)
            final_embs.append(res)
            masks.append(mask)

        # Compute PGT loss and get edge scores
        pgt_loss, edge_scores = compute_pgt_loss(final_embs, masks, self.config['models']['PGT']['d_model'])
        
        # Return labels along with scores if available
        if labels:
            return pgt_loss, {'scores': edge_scores, 'labels': labels}
        else:
            return pgt_loss, {'scores': edge_scores}

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
    import json
    with open("config.json") as f:
        config = json.load(f)
        
    print("Config: ", config)
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'temporal_graph_learning')
    log_dir = os.path.join("logs", f"{experiment_name}_{timestamp}")
    
    # Create CSV logger
    logger = CSVLogger(
        save_dir=log_dir,
        name="metrics",
        flush_logs_every_n_steps=config['training']['log_flush']  # Flush to disk frequently
    )
    print(f"Logging metrics to: {logger.log_dir}")
    
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
        logger=logger,  # Add CSV logger
        enable_checkpointing=True,
        default_root_dir=log_dir  # Store checkpoints in the same directory
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

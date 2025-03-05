# train.py
import lightning as L
import torch
from torch.utils.data import DataLoader
from data import TemporalDataset
from models import DGT, PGT, TemporalEmbeddingManager
from loss import compute_dgt_loss, compute_pgt_loss, adaptive_update
from ogb.linkproppred import Evaluator
import numpy as np

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
        self.val_dataset = self.main_dataset.clone_for_split('val')
        self.test_dataset = self.main_dataset.clone_for_split('test')
        print("Dataset splits initialized.")

    def train_dataloader(self):
        print("Creating train dataloader...")
        return self.create_dataloader(self.train_dataset)

    def val_dataloader(self):
        print("Creating val dataloader...")
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        print("Creating test dataloader...")
        return self.create_dataloader(self.test_dataset)

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
        self.pgt = PGT(config['models']['PGT'])
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

    def on_train_epoch_start(self):
        """Reset embeddings at epoch start"""
        self.emb_manager.reset()

    def on_validation_start(self):
        """Initialize validation embeddings from current state"""
        self.val_emb_manager = TemporalEmbeddingManager(
            self._num_nodes,
            self._node_dim
        )
        self.val_emb_manager.load_state_dict(
            self.emb_manager.state_dict()
        )

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
        
        # Process PGT embeddings
        pgt_loss, pgt_scores = self._pgt_forward(batch['pgt'], emb_manager)

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
        
        # Log with train prefix
        self.log_dict({
            'train_total_loss': results['total_loss'],
            'train_dgt_loss': results['dgt_loss'],
            'train_pgt_loss': results['pgt_loss']
        })
        
        return results['total_loss']

    def validation_step(self, batch, batch_idx):
        # Use the step function with validation embedding manager
        results = self.step(batch, self.val_emb_manager)
        
        # Log with validation prefix
        self.log_dict({
            'val_total_loss': results['total_loss'],
            'val_dgt_loss': results['dgt_loss'],
            'val_pgt_loss': results['pgt_loss']
        })
        
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
            
            old_emb = emb_manager.get_embedding(nodes)
            intermediate = self.dgt(old_emb.unsqueeze(1))
            
            # Only update embeddings for positive items
            if is_positive:
                new_emb = adaptive_update(
                    old_emb,
                    intermediate[max(self.dgt.intermediate_layers.keys())].squeeze(1),
                    dist
                )
                
                for i, node in enumerate(nodes):
                    emb_manager.update_embeddings(node, new_emb[i])
            
            loss = compute_dgt_loss(intermediate, adj, 
                                  self.dgt.intermediate_layers)
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
            
            # Store label if available (for validation/test)
            if 'label' in item:
                labels.append(item['label'])
            
            old_emb = emb_manager.get_embedding(nodes)
            final_out = self.pgt(old_emb.unsqueeze(1))[-1].squeeze(1)
            final_embs.append(final_out)
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
        # Collect all outputs from validation steps
        outputs = self.validation_step_outputs
        
        # Extract all scores and labels
        all_scores = []
        all_labels = []
        
        for output in outputs:
            scores_dict = output['pgt_scores']
            if 'labels' in scores_dict and scores_dict['labels'] is not None:
                scores = scores_dict['scores']
                labels = scores_dict['labels']
                
                # Extend our lists with this batch's data
                all_scores.extend([score.item() for score in scores])
                all_labels.extend([label.item() for label in labels])
        
        # Convert to numpy arrays
        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels)
        
        # Separate positive and negative edges
        pos_indices = labels_np == 1
        neg_indices = labels_np == 0
        
        pos_edge_scores = scores_np[pos_indices]
        neg_edge_scores = scores_np[neg_indices]
        
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
                self.log(f'val_{metric_name}', value)
                
            # Also calculate AUC as a common metric
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones_like(pos_edge_scores), np.zeros_like(neg_edge_scores)])
            y_pred = np.concatenate([pos_edge_scores, neg_edge_scores])
            try:
                auc_score = roc_auc_score(y_true, y_pred)
                self.log('val_auc', auc_score)
            except ValueError as e:
                self.log('val_auc_error', -1.0)
                print(f"Error calculating AUC: {e}")
                
        except Exception as e:
            print(f"Error in evaluator: {e}")
            # Log dummy metric to avoid failure
            self.log('val_metric_error', -1.0)

    def on_validation_epoch_start(self):
        """Initialize validation embeddings and prepare for collection"""
        super().on_validation_start()  # Call existing method
        self.validation_step_outputs = []
        
    def validation_step_end(self, outputs):
        """Collect validation step outputs"""
        self.validation_step_outputs.append(outputs)
        return outputs

if __name__ == "__main__":
    import json
    with open("config.json") as f:
        config = json.load(f)
        
    print("Config: ", config)
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
        log_every_n_steps=5
    )
    print("Fitting model...")
    trainer.fit(model, datamodule=datamodule)

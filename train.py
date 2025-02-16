# train.py
import lightning as L
import torch
from torch.utils.data import DataLoader
from data import TemporalCoraDataset
from models import DGT, PGT, TemporalEmbeddingManager
from loss import compute_dgt_loss, compute_pgt_loss, adaptive_update

class SyncedGraphDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_nodes = None

    def prepare_data(self):
        self.main_dataset = TemporalCoraDataset(
            root=self.config['data']['path'],
            config=self.config
        )
        self.num_nodes = self.main_dataset.num_nodes

    def train_dataloader(self):
        if not self.train_dataset:
            self.train_dataset = self.main_dataset.clone_for_split('train')
        return self.create_dataloader(self.train_dataset)

    def val_dataloader(self):
        if not self.val_dataset:
            self.val_dataset = self.main_dataset.clone_for_split('val')
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        if not self.test_dataset:
            self.test_dataset = self.main_dataset.clone_for_split('test')
        return self.create_dataloader(self.test_dataset)

    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.config['training']['num_workers'],
            persistent_workers=True,
            pin_memory=True,
            shuffle=False
        )

class UnifiedTrainer(L.LightningModule):
    def __init__(self, config, num_nodes):
        super().__init__()
        self.config = config
        self.dgt = DGT(config['models']['DGT'])
        self.pgt = PGT(config['models']['PGT'])
        self.emb_manager = TemporalEmbeddingManager(
            num_nodes=num_nodes,
            node_dim=config['models']['DGT']['d_model']
        )
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

    def training_step(self, batch, batch_idx):
        # Process DGT embeddings
        dgt_loss = self._dgt_forward(batch['dgt'])
        
        # Process PGT embeddings
        pgt_loss = self._pgt_forward(batch['pgt'])

        # Handle timestamp transitions
        if batch['meta']['is_group_end']:
            self.emb_manager.transition_timestamp()

        total_loss = dgt_loss + pgt_loss
        self.log_dict({
            'train_total_loss': total_loss,
            'train_dgt_loss': dgt_loss,
            'train_pgt_loss': pgt_loss
        })
        return total_loss

    def _dgt_forward(self, batch):
        losses = []
        for nodes, adj, dist, mask in zip(batch['nodes'], batch['adj'], 
                                       batch['dist'], batch['central_mask']):
            old_emb = self.emb_manager.get_embedding(nodes)
            intermediate = self.dgt(old_emb.unsqueeze(1))
            
            new_emb = adaptive_update(
                old_emb,
                intermediate[max(self.dgt.intermediate_layers.keys())].squeeze(1),
                dist
            )
            
            for i, node in enumerate(nodes):
                self.emb_manager.update_embeddings(node, new_emb[i])
            
            loss = compute_dgt_loss(intermediate, adj, 
                                  self.dgt.intermediate_layers)
            losses.append(loss)
            
        return torch.mean(torch.stack(losses))

    def _pgt_forward(self, batch):
        final_embs = []
        masks = []
        for nodes, adj, dist, mask in zip(batch['nodes'], batch['adj'],
                                        batch['dist'], batch['central_mask']):
            old_emb = self.emb_manager.get_embedding(nodes)
            final_out = self.pgt(old_emb.unsqueeze(1))[-1].squeeze(1)
            final_embs.append(final_out)
            masks.append(mask)  # Removed explicit .to(self.device)

        return compute_pgt_loss(final_embs, masks,
                                self.config['models']['PGT']['d_model'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                              lr=self.config['training']['lr'])

if __name__ == "__main__":
    import json
    with open("config.json") as f:
        config = json.load(f)
        
    datamodule = SyncedGraphDataModule(config)
    datamodule.prepare_data()
    model = UnifiedTrainer(config, datamodule.num_nodes)
    
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator="auto",
        devices=1,
        log_every_n_steps=5
    )
    trainer.fit(model, datamodule=datamodule)

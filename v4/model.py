#!/usr/bin/env python3
import argparse
import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch.multiprocessing as mp

# Evaluation utility
from evaluation import evaluate  # assumes eval.py provides evaluate()

# Import from utils
from utils import norm as utils_norm

# Assumes PathDataModule code is saved in path_datamodule.py
from loader import PathDataModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PathPredictor(LightningModule):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        lp_norm: int = None,
        max_hops: int = 10,
        max_adjust: float = 0.1,
        norm_fn=None,
        adjust_no_neg_paths_samples=True,
        lr=1e-4,
        scale_loss=False,
        chi2=False,
        **kwargs  # Additional hyperparameters
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Use lp_norm if specified, otherwise use custom norm_fn if provided
        if lp_norm is not None:
            self.norm_fn = lambda tensor, dim: torch.norm(tensor, p=lp_norm, dim=dim)
        elif norm_fn is not None:
            self.norm_fn = norm_fn
        else:
            # Fallback to L2 norm if neither is provided
            self.norm_fn = lambda tensor, dim: torch.norm(tensor, p=2, dim=dim)
            
        self.scale_loss = scale_loss
            
        # project input embeddings to transformer hidden size and back
        self.input_proj = nn.Linear(
            self.hparams.emb_dim, self.hparams.hidden_dim)
        self.pos_encoder = PositionalEncoding(
            self.hparams.hidden_dim, dropout, max_len=max_hops + 1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=self.hparams.get('norm_first', False)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(
            self.hparams.hidden_dim, self.hparams.emb_dim)
        
        # valid and test step outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, src_emb: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(src_emb)
        h = self.pos_encoder(h)
        
        # Generate causal mask for the sequence length
        seq_len = h.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=h.device
        )
        
        # Pass the mask to the transformer instead of is_causal=True
        h = self.transformer(h, mask=causal_mask)
        pred_emb = self.output_proj(h)
        return pred_emb

    def _prepare_batch(self, batch: list[dict]):
        all_emb = []
        meta_info = []
        for sample in batch:
            if 'paths' not in sample or sample['paths'] is None:
                meta_info.append((sample["label"], ))
                continue

            paths = sample['paths']
            num_paths = len(paths)
            max_len = max(len(p) for p in paths) if num_paths else 0
            # print(sample.keys())
            # raise ValueError("Debugging paths: " + str(paths))
            
            if 'shallow_emb' in sample:
                meta_info.append((num_paths, max_len - 1, sample["label"]))
                for idx in range(num_paths):
                    emb = sample['shallow_emb'][idx]
                    if sample.get('features') is not None:
                        feat = sample['features'][idx]
                        emb = torch.cat([emb, feat], dim=-1)
                    all_emb.append(emb)
                
        if len(all_emb) == 0:
            return [], [], meta_info

        src_seq = [e[:-1] for e in all_emb]
        tgt_seq = [e[1:] for e in all_emb]
        src_emb = pad_sequence(src_seq, batch_first=True,
                               padding_value=0.0, padding_side="right").detach()
        tgt_emb = pad_sequence(tgt_seq, batch_first=True,
                               padding_value=0.0, padding_side="right").detach()
        return src_emb, tgt_emb, meta_info

    def _predict(self, batch: list[dict]):
        src_emb, tgt_emb, meta = self._prepare_batch(batch)
        if len(meta) == 0:
            return None, None
        
        pred_emb = self(src_emb)
        diff = self.norm_fn(pred_emb - tgt_emb, dim=-1)
        
        return diff, meta

    def training_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        if meta is None:
            print(f"meta is None at {batch_idx} batch_idx")
            return None
        
        losses, ptr = [], 0
        for info in meta:
            if len(info) == 1:
                print(f"Skipping single label sample at {batch_idx} batch_idx")
                continue
            
            num_paths, length, label = info
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            if neg.numel():
                mean, std = neg.mean(0), neg.std(0, unbiased=False)
                z = (pos - mean)/(std+1e-8)
                mean_z = z.mean() if label else -z.mean()

                # Apply arcsinh scaling if specified
                loss = torch.asinh(mean_z) if self.scale_loss else mean_z
                losses.append(loss)
            
            ptr += num_paths
        loss = torch.stack(losses).mean() # diff pos lower better so z_pos must be to left of z_neg
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        
        return loss

    def _evaluation_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        if meta is None:
            print(f"meta is None at {batch_idx} batch_idx")
            return None, None
        
        batch_items = []  # Store structured items for each sample
        losses = []
                
        ptr = 0
        for meta_info in meta:
            if len(meta_info) == 1:
                item = {
                    "score": 0,
                    "label": meta_info[0],  # Single label for this sample
                    "length": 0,
                    "has_neg": False
                }
                batch_items.append(item)
                continue
            
            num_paths, length, label = meta_info
            
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            
            if neg.numel():
                mean, std = neg.mean(0), neg.std(0, unbiased=False)
                z_pos = (pos - mean)/(std+1e-8)
                mean_z_pos = z_pos.mean()
                
                if self.hparams.chi2:
                    # Chi-squared statistic from z-scores
                    chi2_stat = torch.sum(z_pos**2)
                    
                    # Degrees of freedom = hidden_dim of the model
                    df = self.hparams.hidden_dim
                    
                    # CDF of chi-squared distribution
                    chi2_dist = torch.distributions.chi2.Chi2(df)
                    cdf_val = chi2_dist.cdf(chi2_stat)
                    percentile_pos = 1 - cdf_val.item()
                else:
                    # Convert mean z-score to percentile for positive sample using normal CDF
                    percentile_pos = 1 - torch.special.ndtr(mean_z_pos).item()
                
                # Calculate mean z-score for loss
                loss = torch.asinh(mean_z_pos) if self.scale_loss else mean_z_pos
                losses.append(loss if label else -loss)
            else:
                percentile_pos = 1.0

            # Create item with organized structure
            item = {
                'score': percentile_pos,  # Single percentile from mean z-score
                'length': length,  # Path length for this sample
                'label': label,  # Label for this sample
                'has_neg': neg.numel() > 0
            }
            batch_items.append(item)
            ptr += num_paths
        
        # Use negated z-scores for loss, similar to training_step
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        
        return loss, batch_items

    def validation_step(self, batch, batch_idx):
        loss, batch_items = self._evaluation_step(batch, batch_idx)
        if loss is not None:
            self.validation_step_outputs.append({'loss': loss, 'items': batch_items})
        return loss

    def test_step(self, batch, batch_idx):
        loss, batch_items = self._evaluation_step(batch, batch_idx)
        if loss is not None:
            self.test_step_outputs.append({'loss': loss, 'items': batch_items})
        return loss

    def _on_evaluation_epoch_end(self, outputs, stage):
        if not outputs:
            print(f"Warning: No outputs found in {stage} step. Skipping {stage} epoch end.")
            return

        # Calculate epoch-level loss
        epoch_losses = []
        for output in outputs:
            if output and 'loss' in output and output['loss'] is not None:
                epoch_losses.append(output['loss'])
        
        if epoch_losses:
            mean_epoch_loss = torch.stack(epoch_losses).mean()
            self.log(f'{stage}_loss', mean_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"{stage.capitalize()} loss: {mean_epoch_loss.item()}")
        else:
            print(f"Warning: No valid losses found in {stage} step. Skipping {stage} loss logging.")
        
        # Extract and organize values for evaluation
        scores, lengths, labels, has_neg = [], [], [], []
        
        for output in outputs:
            if output and 'items' in output:
                for item in output['items']:
                    scores.append(item['score'])
                    lengths.append(item.get('length', 0))
                    labels.append(item['label'])
                    has_neg.append(item.get('has_neg', False))
        
        if not scores:
            print(f"Warning: No items with scores found in {stage} outputs. Skipping evaluation.")
            return

        scores = torch.tensor(scores)
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)
        has_neg = torch.tensor(has_neg)
        
        max_hops = self.hparams.max_hops
        max_adjust = self.hparams.get('max_adjust', 1.0)
        
        min_len = lengths.min() if lengths.numel() > 0 else 0
        ratios = 1 - ((lengths.float() - min_len) / (max_hops - min_len + 1e-8))
        
        if self.hparams.get('adjust_no_neg_paths_samples', True):
            adjusted_scores = scores + (ratios * max_adjust)
        else:
            adjusted_scores = scores.clone()
            if has_neg.any():
                adjustment = ratios[has_neg] * max_adjust
                adjusted_scores[has_neg] = scores[has_neg] + adjustment
        
        dataset_name = self.trainer.datamodule.dataset
        results = evaluate(dataset_name, adjusted_scores[labels == 1], adjusted_scores[labels == 0])
        
        for k, v in results.items():
            self.log(f'{stage}_{k}', v, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self._on_evaluation_epoch_end(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self._on_evaluation_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    # Set start method to 'spawn' before any other multiprocessing code runs
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Path prediction with Transformer encoder")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    # override max_epochs from config if available
    max_epochs = cfg.get('max_epochs', cfg.get('epochs', 10))

    dm = PathDataModule(
        config_path=args.config,
        batch_size=cfg.get('batch_size', 32),
        shuffle=cfg.get('shuffle', False)
    )
    
    # Get model_name from embedding config for normalization
    model_name = None
    emb_dim = None
    embedding_config_path = cfg.get('embedding_config')
    if embedding_config_path and os.path.exists(embedding_config_path):
        try:
            with open(embedding_config_path, 'r') as f:
                embedding_config = json.load(f)
                model_name = embedding_config.get('model_name', 'transe')
                emb_dim = embedding_config.get('hidden_channels', 128)  # Default to 128 if not specified
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load embedding config: {e}")
    
    hidden_dim = cfg.get('hidden_dim', emb_dim)
    
    # Check if lp_norm is in config
    has_lp_norm = 'lp_norm' in cfg
    
    # Initialize model parameters
    model_params = {
        'emb_dim': emb_dim,
        'hidden_dim': hidden_dim,
        'nhead': cfg.get('nhead', 8),
        'num_layers': cfg.get('num_layers', 4),
        'dim_feedforward': cfg.get('dim_feedforward', 512),
        'dropout': cfg.get('dropout', 0.1),
        'max_hops': cfg.get('max_hops', 10),
        'max_adjust': cfg.get('max_adjust', 1.0),
        'adjust_no_neg_paths_samples': cfg.get('adjust_no_neg_paths_samples', True),
        'lr': cfg.get('lr', 1e-4),
        'scale_loss': cfg.get('scale_loss', False),
        'chi2': cfg.get('chi2', False),
    }
    
    # If lp_norm is in config, use it, otherwise use norm_fn from embedding config
    if has_lp_norm:
        model_params['lp_norm'] = cfg['lp_norm']
    else:
        # Use utils.norm with model_name from embedding config
        norm_fn = lambda tensor, dim: utils_norm(tensor, model=None, model_name=model_name, dim=dim)
        model_params['norm_fn'] = norm_fn

    model = PathPredictor(**model_params)

    # Extract storage directory
    storage_dir = cfg.get('storage_dir', 'runs')
    
    # Get embedding model name from embedding config file
    embedding_model = "default_model"  # Default fallback value
    embedding_config_path = cfg.get('embedding_config')
    if embedding_config_path and os.path.exists(embedding_config_path):
        try:
            with open(embedding_config_path, 'r') as f:
                embedding_config = json.load(f)
                embedding_model = embedding_config.get('model_name', embedding_model)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load embedding config: {e}")
    
    # Get dataset name
    dataset_name = dm.dataset  # Or cfg.get('dataset', 'default_dataset')
    
    # Create the log directory with the specified pattern
    log_dir = os.path.join(storage_dir, embedding_model, dataset_name)
    print(f"Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure CSV logger
    logger = CSVLogger(
        save_dir=log_dir,
        name="lightning_logs",
        version=None  # Auto-increment version
    )
    
    ckpt = ModelCheckpoint(monitor='val_loss', save_top_k=cfg.get('num_ckpt', 1), dirpath=log_dir, mode='min')
    trainer = Trainer(max_epochs=max_epochs, callbacks=[ckpt], logger=logger)
    trainer.fit(model, dm)

    # Run the test stage after training is complete, using the best checkpoint
    print("\n--- Running Test Stage ---")
    trainer.test(model, datamodule=dm, ckpt_path='best')
    
    # Run the test-time evaluation
    print("\n--- Running Test-Time Evaluation ---")
    test_epochs = cfg.get('test_time', 0)
    if test_epochs > 0:
        dm.test_time = True
        test_trainer = Trainer(max_epochs=test_epochs, logger=logger, callbacks=[ckpt])
        test_trainer.fit(model, dm)


if __name__ == '__main__':
    main()

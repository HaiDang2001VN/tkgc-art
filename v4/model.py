#!/usr/bin/env python3
import argparse
import json
import os
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
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
        positive_deviation=False,
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

    def _predict(self, batch: list[dict]):
        # This method combines the logic of _prepare_batch and the original _predict
        all_emb = []
        meta_info = []
        for sample in batch:
            if 'paths' not in sample or not sample['paths']:
                meta_info.append((sample["label"],))
                continue

            paths = sample['paths']
            num_paths = len(paths)
            max_len = max(len(p) for p in paths) if num_paths else 0
            
            if 'shallow_emb' in sample:
                meta_info.append((num_paths, max_len - 1, sample["label"]))
                for idx in range(num_paths):
                    emb = sample['shallow_emb'][idx]
                    if sample.get('features') is not None:
                        feat = sample['features'][idx]
                        emb = torch.cat([emb, feat], dim=-1)
                    all_emb.append(emb)
        
        if not all_emb:
            # No paths found in the batch, return None for diff but keep meta_info
            return None, meta_info

        src_seq = [e[:-1] for e in all_emb]
        tgt_seq = [e[1:] for e in all_emb]
        src_emb = pad_sequence(src_seq, batch_first=True, padding_value=0.0, padding_side="right").detach()
        tgt_emb = pad_sequence(tgt_seq, batch_first=True, padding_value=0.0, padding_side="right").detach()

        try:
            pred_emb = self(src_emb)
            diff = self.norm_fn(pred_emb - tgt_emb, dim=-1)
        except TypeError as e:
            print(f"Error during prediction: {e} with src_emb: {type(src_emb)}, tgt_emb: {type(tgt_emb)}")
            print(f"src_emb shape: {len(src_emb)}, tgt_emb shape: {len(tgt_emb)}")
            diff = None
            meta_info = None
        
        return diff, meta_info

    def training_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        losses, ptr = [], 0
        for info in meta:
            if len(info) == 1:
                # This sample has no paths, skip loss calculation
                continue
            
            num_paths, length, label = info
            # This part of the loop is only reachable if diff is not None
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            if neg.numel():
                refs = slice_diff if self.hparams.positive_deviation else neg
                
                mean, std = refs.mean(0), refs.std(0, correction=0)
                z = (pos - mean)/(std+1e-8)
                mean_z = z.mean() if label else -z.mean()

                # Apply arcsinh scaling if specified
                loss = torch.asinh(mean_z) if self.scale_loss else mean_z
                losses.append(loss)
            
            ptr += num_paths
        
        if not losses:
            # No loss was computed for this batch (e.g., no paths or no negative samples)
            return None

        loss = torch.stack(losses).mean()
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        
        return loss

    def _evaluation_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        batch_items = []  # Store structured items for each sample
        losses = []
                
        ptr = 0
        for meta_info in meta:
            if len(meta_info) == 1:
                item = {
                    "score": 0.0,
                    "length": None,
                    "label": meta_info[0],  # Single label for this sample
                }
                batch_items.append(item)
                continue
            
            num_paths, length, label = meta_info
            
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            loss, mean_z_pos = None, None
            
            if neg.numel() > 0:
                refs = slice_diff if self.hparams.positive_deviation else neg
                
                mean, std = refs.mean(0), refs.std(0, correction=0)
                z_pos = (pos - mean)/(std+1e-8)
                mean_z_pos = z_pos.mean()
                
                if self.hparams.chi2:
                    # Chi statistic from z-scores (L2 norm of z-score vector)
                    chi_stat = torch.norm(z_pos)
                    
                    # Degrees of freedom = path length
                    df = torch.tensor(length, device=self.device, dtype=torch.float)
                    
                    # CDF of chi distribution
                    chi_dist = torch.distributions.chi.Chi(df)
                    cdf_val = chi_dist.cdf(chi_stat)
                    percentile_pos = 1.0 - cdf_val.item()
                else:
                    # Convert mean z-score to percentile for positive sample using normal CDF
                    percentile_pos = 1.0 - torch.special.ndtr(mean_z_pos).item()
                
                # Calculate mean z-score for loss
                loss = torch.asinh(mean_z_pos) if self.scale_loss else mean_z_pos
                losses.append(loss if label else -loss)
            else:
                percentile_pos = 1.0

            # Create item with organized structure
            item = {
                'score': percentile_pos,  # Single percentile from mean z-score
                'pos_dist': pos.detach().cpu().numpy(),  # Distance for positive sample
                'neg_dist': neg.detach().cpu().numpy() if neg.numel() > 0 else None,  # Distances for negative samples
                'length': length,  # Path length for this sample
                'label': label,  # Label for this sample
                'has_neg': neg.numel() > 0,
                'loss': loss,  # Loss for this sample'
                'mean_z_pos': mean_z_pos
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
        pos_losses = []
        neg_losses = []
        for output in tqdm(outputs, desc=f"Processing {stage} losses"):
            if output and 'loss' in output and output['loss'] is not None and 'items' in output:
                for item in output['items']:
                    if 'loss' in item and item['loss'] is not None:
                        if item['label'] == 1:
                            pos_losses.append(item['loss'])
                        elif item['label'] == 0:
                            neg_losses.append(item['loss'])
        
        if pos_losses:
            pos_epoch_loss = torch.stack(pos_losses).mean()
            self.log(f'{stage}_pos_loss', pos_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"{stage.capitalize()} positive loss: {pos_epoch_loss.item()}")
        else:
            pos_epoch_loss = torch.tensor(0.0, device=self.device)
            self.log(f'{stage}_pos_loss', pos_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"Warning: No positive losses found in {stage} step. Skipping {stage} positive loss logging.")
        
        if neg_losses:
            neg_epoch_loss = torch.stack(neg_losses).mean()
            self.log(f'{stage}_neg_loss', neg_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"{stage.capitalize()} negative loss: {neg_epoch_loss.item()}")
        else:
            neg_epoch_loss = torch.tensor(0.0, device=self.device)
            self.log(f'{stage}_neg_loss', neg_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"Warning: No negative losses found in {stage} step. Skipping {stage} negative loss logging.")
            
        if pos_losses or neg_losses:
            mean_epoch_loss = pos_epoch_loss + neg_epoch_loss
            self.log(f'{stage}_loss', mean_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"{stage.capitalize()} loss: {mean_epoch_loss.item()}")
        else:
            print(f"Warning: No valid losses found in {stage} step. Skipping {stage} loss logging.")
        
        # Extract and organize values for evaluation
        scores, lengths, labels, has_neg, pos_dist, neg_dists, mean_z, losses = [], [], [], [], [], [], [], []
        
        max_hops = self.hparams.max_hops
        max_adjust = self.hparams.get('max_adjust', 1.0)
        
        for output in tqdm(outputs, desc=f"Processing {stage} outputs"):
            if output and 'items' in output:
                for item in output['items']:
                    scores.append(item['score'])
                    lengths.append(item['length'] if item['length'] is not None else max_hops + 2)
                    labels.append(item['label'])
                    has_neg.append(item.get('has_neg', False))
                    pos_dist.append(item.get('pos_dist', None))
                    neg_dists.append(item.get('neg_dist', None))
                    mean_z.append(item.get('mean_z_pos', None))
                    losses.append(item.get('loss', None))
        
        if not scores:
            print(f"Warning: No items with scores found in {stage} outputs. Skipping evaluation.")
            return

        scores = torch.tensor(scores)
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)
        has_neg = torch.tensor(has_neg)
        
        min_len = lengths.min() if lengths.numel() > 0 else 0
        ratios = 1 - ((lengths.float() - min_len) / (max_hops - min_len + 1)) # max_hops = max length - 1
        
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

        # --- Export as JSON: a list of dicts, one per item ---
        try:
            epoch = self.trainer.current_epoch
        except Exception:
            epoch = "unknown"
        log_dir = getattr(self.trainer.logger, "save_dir", "logs")
        if hasattr(self.trainer.logger, "name"):
            log_dir = os.path.join(log_dir, self.trainer.logger.name)
        if hasattr(self.trainer.logger, "version"):
            log_dir = os.path.join(log_dir, str(self.trainer.logger.version))
        os.makedirs(log_dir, exist_ok=True)
        # Determine prefix: "train" if not test_time, "test" if test_time
        test_prefix = "test" if getattr(self.trainer.datamodule, "test_time", False) else "train"
        export_path = os.path.join(log_dir, f"{test_prefix}_{stage}_{epoch}_raw.json")

        # Prepare the list of dicts for export
        export_items = []
        for i in range(len(scores)):
            export_items.append({
                "score": float(scores[i]),
                "length": int(lengths[i]),
                "label": int(labels[i]),
                "has_neg": bool(has_neg[i]),
                "pos_dist": pos_dist[i].tolist() if pos_dist[i] is not None else None,
                "neg_dist": neg_dists[i].tolist() if neg_dists[i] is not None else None,
                "adjusted_score": float(adjusted_scores[i]),
                "mean_z": float(mean_z[i]) if mean_z[i] is not None else None,
                "loss": float(losses[i]) if losses[i] is not None else None,
            })

        with open(export_path, "w") as f:
            json.dump(export_items, f, indent=2)
        print(f"Exported raw evaluation items to {export_path}")

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
        'positive_deviation': cfg.get('positive_deviation', False),
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
    
    # Configure Wandb logger
    logger = WandbLogger(
        name=f"{embedding_model}_{dataset_name}",
        save_dir=log_dir,
        project=cfg.get("wandb_project", "thesis-graph"),
        entity=cfg.get("wandb_entity", None),
        log_model=True
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

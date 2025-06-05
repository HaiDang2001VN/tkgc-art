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

# Evaluation utility
from evaluation import evaluate  # assumes eval.py provides evaluate()

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
        lp_norm: int = 2,
        max_hops: int = 10, # Added max_hops
        norm_fn=None
    ):
        super().__init__()
        self.save_hyperparameters()
        # normalization function: use kge_proxy.norm if provided, else torch.norm
        self.norm_fn = norm_fn or (lambda tensor, dim: torch.norm(
            tensor, p=self.hparams.lp_norm, dim=dim))
        # project input embeddings to transformer hidden size and back
        self.input_proj = nn.Linear(
            self.hparams.emb_dim, self.hparams.hidden_dim)
        self.pos_encoder = PositionalEncoding(
            self.hparams.hidden_dim, dropout, max_len=max_hops + 1, batch_first=True) # Use max_hops + 1
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

    def forward(self, src_emb: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(src_emb)
        h = self.pos_encoder(h)
        h = self.transformer(h, is_causal=True)
        pred_emb = self.output_proj(h)
        return pred_emb

    def _prepare_batch(self, batch: list[dict]):
        all_emb = []
        meta_info = []
        for sample in batch:
            if sample is None:
                continue

            paths = sample['paths']
            num_paths = len(paths)
            max_len = max(p.size(0) for p in paths) if num_paths else 0
            
            if 'shallow_emb' in sample:
                meta_info.append((num_paths, max_len - 1))
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
        src_emb = torch.tensor(pad_sequence(src_seq, batch_first=True, padding_value=0.0))
        tgt_emb = torch.tensor(pad_sequence(tgt_seq, batch_first=True, padding_value=0.0))
        return src_emb, tgt_emb, meta_info

    def _predict(self, batch: list[dict]):
        src_emb, tgt_emb, meta = self._prepare_batch(batch)
        if len(meta) == 0:
            return None, None
        
        pred_emb = self(src_emb)
        diff = self.norm_fn(pred_emb - tgt_emb, dim=-1)
        return diff, meta

    def on_validation_epoch_start(self):
        self.pos_scores = []
        self.neg_scores = []

    def training_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        if meta is None:
            return None
        
        losses, ptr = [], 0
        for num_paths, length in meta:
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            if neg.numel():
                mean, std = neg.mean(0), neg.std(0, unbiased=False)
                z = (pos - mean)/(std+1e-8)
            else:
                z = pos*0
            losses.append(z.mean())
            ptr += num_paths
        loss = -torch.stack(losses).mean()
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        diff, meta = self._predict(batch)
        
        if meta is None:
            return None, None
        
        ptr = 0
        for num_paths, length in meta:
            slice_diff = diff[ptr:ptr + num_paths, :length]
            pos, neg = slice_diff[0], slice_diff[1:]
            if neg.numel():
                mean, std = neg.mean(0), neg.std(0, unbiased=False)
                z = (pos - mean)/(std+1e-8)
                neg_z = (neg-mean)/(std+1e-8)
            else:
                z = pos*0
                neg_z = neg
            self.pos_scores.extend(z.tolist())
            self.neg_scores.extend(neg_z.flatten().tolist())
            ptr += num_paths
        loss = -torch.tensor(self.pos_scores).mean()
        
        self.log('val_loss', loss, on_step=True, prog_bar=True)
        
        return loss, torch.tensor(self.pos_scores)

    def on_validation_epoch_end(self, outputs):
        # perform full evaluation using external evaluate()
        dataset_name = self.trainer.datamodule.dataset
        results = evaluate(dataset_name, torch.tensor(
            self.pos_scores), torch.tensor(self.neg_scores))
        for k, v in results.items():
            self.log(k, v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
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
    dm.prepare_data()
    dm.setup()

    norm_fn = dm.kge_proxy.norm if dm.kge_proxy is not None else None
    emb_dim = dm.emb_dim
    hidden_dim = cfg.get('hidden_dim', emb_dim)
    lp = cfg.get('lp_norm', 2)

    model = PathPredictor(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        nhead=cfg.get('nhead', 8),
        num_layers=cfg.get('num_layers', 4),
        dim_feedforward=cfg.get('dim_feedforward', 512),
        dropout=cfg.get('dropout', 0.1),
        lp_norm=lp,
        max_hops=cfg.get('max_hops', 10), # Pass max_hops
        norm_fn=norm_fn
    )

    ckpt = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    trainer = Trainer(max_epochs=max_epochs, callbacks=[ckpt])
    trainer.fit(model, dm)

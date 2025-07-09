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

# Assumes PathDataModule code is saved in path_datamodule.py
from loader import PathDataModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine sequence length based on batch_first
        seq_len = x.size(1) if self.batch_first else x.size(0)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device)
        
        # Get positional embeddings
        pos_embeddings = self.pos_embedding(positions)
        
        # Reshape for broadcasting
        if self.batch_first:
            pos_embeddings = pos_embeddings.unsqueeze(0)  # (1, seq_len, d_model)
        else:
            pos_embeddings = pos_embeddings.unsqueeze(1)  # (seq_len, 1, d_model)
            
        x = x + pos_embeddings
        return self.dropout(x)


class PathPredictor(LightningModule):
    """
    PathPredictor model that supports both legacy and new prediction formats.
    
    The model now supports:
    1. Legacy prediction (_predict): Uses causal masking for sequence-to-sequence prediction
    2. New prediction (_predict_new): Uses non-causal transformer to score path prefixes
    
    New features added:
    - score_proj: Linear layer that maps embeddings to single scores
    - Non-causal forward pass option
    - Prefix-length-grouped batch processing
    - Z-score calculation across prefix lengths
    """
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_hops: int = 10,
        max_adjust: float = 0.1,
        adjust_no_neg_paths_samples=True,
        lr=1e-4,
        scale_loss=False,
        chi2=False,
        positive_deviation=False,
        loss_decay: float = 0.5,
        **kwargs  # Additional hyperparameters
    ):
        super().__init__()
        self.save_hyperparameters()
            
        self.scale_loss = scale_loss
            
        # project input embeddings to transformer hidden size and back
        self.input_proj = nn.Linear(
            self.hparams.emb_dim, self.hparams.hidden_dim)
        self.pos_encoder = PositionalEncoding(
            self.hparams.hidden_dim, dropout, max_len=2 * (max_hops + 1), batch_first=True)
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
        
        # Add scoring layer for single score prediction
        self.score_proj = nn.Linear(self.hparams.emb_dim, 1)
        
        # valid and test step outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, src_emb: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        h = self.input_proj(src_emb)
        h = self.pos_encoder(h)
        
        # Generate causal mask for the sequence length only if requested
        if use_causal_mask:
            seq_len = h.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=h.device
            )
            # Pass the mask to the transformer
            h = self.transformer(h, mask=causal_mask)
        else:
            # No mask for non-causal prediction
            h = self.transformer(h)
            
        pred_emb = self.output_proj(h)
        return pred_emb

    def _predict(self, batch):
        """
        Prediction function that handles prefix-length-grouped batch format from the new collate function.
        
        Args:
            batch: dict: {prefix_length: [node_embeddings, edge_embeddings, meta]} from collate_by_prefix_length
        
        Returns:
            z_scores tensor and meta_info for scoring approach
        """
        # Collection for tracking unique samples and their scores
        all_samples = []  # List of unique samples identified by (u, v, ts)
        
        # Sort prefix lengths to process from shorter to longer
        sorted_prefix_lengths = sorted(batch.keys())
        
        # First pass: collect all unique samples from meta_data across all prefix lengths
        seen_samples = set()
        for prefix_len in sorted_prefix_lengths:
            if not batch[prefix_len] or len(batch[prefix_len]) < 3:  # Empty result or incomplete data
                continue
                
            # Unpack the batch components
            _, _, meta_data = batch[prefix_len]
            
            # Process each sample's metadata
            for sample_meta in meta_data:
                # Create a unique key for this sample using .item() only for metadata
                u = sample_meta.get('u', -1).item() if hasattr(sample_meta.get('u', -1), 'item') else sample_meta.get('u', -1)
                v = sample_meta.get('v', -1).item() if hasattr(sample_meta.get('v', -1), 'item') else sample_meta.get('v', -1)
                ts = sample_meta.get('ts', -1).item() if hasattr(sample_meta.get('ts', -1), 'item') else sample_meta.get('ts', -1)
                sample_key = (u, v, ts)
                
                if sample_key not in seen_samples:
                    seen_samples.add(sample_key)
                    all_samples.append(sample_meta)  # Store the full metadata for this sample
    
        outputs = {}
        
        # Process each prefix length
        for prefix_len in sorted_prefix_lengths:
            if not batch[prefix_len] or len(batch[prefix_len]) < 3:  # Empty result or incomplete data
                continue
                
            # Unpack the batch components
            node_embeddings, edge_embeddings, meta_data = batch[prefix_len]
            
            # Skip if no embeddings
            if node_embeddings.numel() == 0:
                continue
                
            # Create tensor of type_embedding along the node_embeddings
            type_embeddings = []
            num_paths = []
            
            for sample_meta in meta_data:
                type_embedding = sample_meta.get('type_embedding', None)
                num_paths.append(sample_meta.get('num_paths', 1))
                if type_embedding is not None:
                    type_embeddings.extend([type_embedding] * num_paths[-1])

            if not type_embeddings:
                # No type embeddings available, skip this prefix length
                continue
            
            # Use tensor operations for embeddings
            type_embeddings = torch.stack(type_embeddings, dim=0)
            edge_embeddings = torch.cat([type_embeddings, edge_embeddings], dim=1)
            
            # Create interleaved edge-node embeddings
            batch_size, length, emb_dim = node_embeddings.size()
            embeddings = torch.zeros(batch_size, 2 * length, emb_dim, dtype=node_embeddings.dtype,
                                    device=node_embeddings.device)
            
            # Edge embeddings start first - use tensor operations
            embeddings[:, 0::2, :] = edge_embeddings
            embeddings[:, 1::2, :] = node_embeddings
            
            # Prediction with gradient
            pred_emb = self.forward(embeddings, use_causal_mask=False)
            pred_scores = self.score_proj(pred_emb[:, 0, :])
            
            # Compute z-scores while preserving gradients
            cur_idx = 0
            z_scores_list = []
            
            for num_path in num_paths:
                # Get the current sample's prefix scores
                scores = pred_scores[cur_idx:cur_idx + num_path]
                cur_idx += num_path
                
                # Calculate mean and std while preserving gradients
                mean = scores.mean(0)
                # Use unbiased=False to match previous behavior
                std = scores.std(0, unbiased=False)
                
                # Calculate z-score for first score (positive sample)
                z = (scores[0] - mean)/(std + 1e-8)
                z_scores_list.append(z)
            
            # Stack z_scores into single torch tensor - maintains gradients
            z_scores = torch.stack(z_scores_list)
            
            # Create unified outputs from z_scores and meta
            outputs[prefix_len] = {
                'z_scores': z_scores,  # This now preserves gradients
                'meta': meta_data,
                'raw_scores': pred_scores  # Include raw scores for debugging
            }
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self._predict(batch)

        if not outputs:  # Check if outputs is empty
            return None
            
        # Get all prefix lengths from the outputs
        prefix_lengths = sorted([int(k) for k in outputs.keys() if k.isdigit()])
        if not prefix_lengths:
            return None  # No valid prefix lengths
        
        # First, collect all unique samples across all prefix lengths
        all_samples = []  # List of unique samples identified by (u, v, ts)
        sample_to_idx = {}  # Maps sample key to index in all_samples
        
        # First pass: collect all unique samples and map to indices
        for prefix_len in prefix_lengths:
            if prefix_len not in outputs or 'meta' not in outputs[prefix_len]:
                continue
                
            meta_data = outputs[prefix_len]['meta']
            
            for meta in meta_data:
                # Create a unique key for this sample
                u = meta.get('u', -1).item() if hasattr(meta.get('u', -1), 'item') else meta.get('u', -1)
                v = meta.get('v', -1).item() if hasattr(meta.get('v', -1), 'item') else meta.get('v', -1)
                ts = meta.get('ts', -1).item() if hasattr(meta.get('ts', -1), 'item') else meta.get('ts', -1)
                sample_key = (u, v, ts)
                
                if sample_key not in sample_to_idx:
                    sample_to_idx[sample_key] = len(all_samples)
                    all_samples.append(meta)  # Store the full metadata for future reference
    
        if not all_samples:
            return None  # No samples found
        
        # Create tensors for z-scores, mask, weights, and labels
        num_samples = len(all_samples)
        num_prefix_lengths = len(prefix_lengths)
        
        z_scores = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        mask = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        weights = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        
        # Extract labels from all unique samples
        labels = []
        for meta in all_samples:
            label = meta.get('label', 0)
            if isinstance(label, torch.Tensor):
                labels.append(label)
            else:
                labels.append(torch.tensor(label, device=self.device))
        labels = torch.stack(labels)
        
        # Second pass: fill z-scores and mask tensors
        for i, prefix_len in enumerate(prefix_lengths):
            if prefix_len not in outputs or 'z_scores' not in outputs[prefix_len] or 'meta' not in outputs[prefix_len]:
                continue
                
            prefix_scores = outputs[prefix_len]['z_scores']  # These now have gradients preserved
            meta_data = outputs[prefix_len]['meta']
            
            # Fill z-scores tensor for this prefix length
            for j, (score, meta) in enumerate(zip(prefix_scores, meta_data)):
                # It's okay to use .item() for metadata since they're just for lookup
                u = meta.get('u', -1).item() if hasattr(meta.get('u', -1), 'item') else meta.get('u', -1)
                v = meta.get('v', -1).item() if hasattr(meta.get('v', -1), 'item') else meta.get('v', -1)
                ts = meta.get('ts', -1).item() if hasattr(meta.get('ts', -1), 'item') else meta.get('ts', -1)
                sample_key = (u, v, ts)
                
                sample_idx = sample_to_idx.get(sample_key, -1)
                if sample_idx >= 0:
                    # Directly assign the score tensor to preserve gradients
                    z_scores[sample_idx, i] = score
                    mask[sample_idx, i] = 1.0

        # Calculate weights with exponential decay
        decay_rate = self.hparams.loss_decay
        for i, prefix_len in enumerate(prefix_lengths):
            # Higher weights for longer prefixes
            weights[:, i] = torch.exp(-decay_rate * (prefix_lengths[-1] - prefix_len))
        
        # Apply scaling if configured (MOVED HERE - before label adjustment)
        if self.scale_loss:
            z_scores = torch.asinh(z_scores)
        
        # Adjust z-scores based on labels (we want to maximize for positive, minimize for negative)
        adjusted_z_scores = z_scores * (2 * labels.unsqueeze(1) - 1)  # Maps 0->-1, 1->1
        
        # Calculate weighted sum
        weighted_sum = (weights * mask * adjusted_z_scores).sum(dim=1)
        weight_sum = (weights * mask).sum(dim=1) + 1e-8  # Avoid division by zero
        
        # Get average score per sample
        weighted_avg = weighted_sum / weight_sum
        
        # Overall loss is negative mean of weighted averages (we want to maximize scores)
        loss = -weighted_avg.mean()
        
        # Remove scaling from here since it's now applied to z-scores directly
        # if self.scale_loss:
        #     loss = torch.asinh(loss)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def _evaluation_step(self, batch, batch_idx):
        outputs = self._predict(batch)

        if not outputs:
            return torch.tensor(0.0, device=self.device), []

        # --- Loss calculation similar to training_step ---
        prefix_lengths = sorted([int(k) for k in outputs.keys() if k.isdigit()])
        if not prefix_lengths:
            return torch.tensor(0.0, device=self.device), []

        all_samples = []
        sample_to_idx = {}
        
        for prefix_len in prefix_lengths:
            if prefix_len not in outputs or 'meta' not in outputs[prefix_len]:
                continue
            meta_data = outputs[prefix_len]['meta']
            for meta in meta_data:
                u = meta.get('u', -1).item() if hasattr(meta.get('u', -1), 'item') else meta.get('u', -1)
                v = meta.get('v', -1).item() if hasattr(meta.get('v', -1), 'item') else meta.get('v', -1)
                ts = meta.get('ts', -1).item() if hasattr(meta.get('ts', -1), 'item') else meta.get('ts', -1)
                v_pos = meta.get('v_pos', v).item() if hasattr(meta.get('v_pos', v), 'item') else meta.get('v_pos', v)
                sample_key = (u, v, ts, v_pos)  # Include v_pos in the key
                if sample_key not in sample_to_idx:
                    sample_to_idx[sample_key] = len(all_samples)
                    all_samples.append(meta)

        if not all_samples:
            return torch.tensor(0.0, device=self.device), []

        num_samples = len(all_samples)
        num_prefix_lengths = len(prefix_lengths)
        
        z_scores = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        mask = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        weights = torch.zeros(num_samples, num_prefix_lengths, device=self.device)
        
        # Tensors to store the final raw score and length for each sample's longest prefix
        score_for_longest_prefix = torch.zeros(num_samples, device=self.device)
        length_for_longest_prefix = torch.zeros(num_samples, dtype=torch.int, device=self.device)
        
        labels = []
        for meta in all_samples:
            label = meta.get('label', 0)
            if isinstance(label, torch.Tensor):
                labels.append(label)
            else:
                labels.append(torch.tensor(label, device=self.device))
        labels = torch.stack(labels)
        
        # First pass: fill tensors and determine longest prefix raw score/length
        for i, prefix_len in enumerate(prefix_lengths):
            if prefix_len not in outputs or 'z_scores' not in outputs[prefix_len] or 'meta' not in outputs[prefix_len]:
                continue
            prefix_z_scores = outputs[prefix_len]['z_scores']
            prefix_raw_scores_flat = outputs[prefix_len]['raw_scores']
            meta_data = outputs[prefix_len]['meta']

            raw_score_cursor = 0
            for z_score, meta in zip(prefix_z_scores, meta_data):
                # Get sample_idx
                u = meta.get('u', -1).item() if hasattr(meta.get('u', -1), 'item') else meta.get('u', -1)
                v = meta.get('v', -1).item() if hasattr(meta.get('v', -1), 'item') else meta.get('v', -1)
                ts = meta.get('ts', -1).item() if hasattr(meta.get('ts', -1), 'item') else meta.get('ts', -1)
                v_pos = meta.get('v_pos', v).item() if hasattr(meta.get('v_pos', v), 'item') else meta.get('v_pos', v)
                sample_key = (u, v, ts, v_pos)  # Include v_pos in the key
                sample_idx = sample_to_idx.get(sample_key, -1)

                if sample_idx >= 0:
                    # Fill z_scores and mask
                    z_scores[sample_idx, i] = z_score
                    mask[sample_idx, i] = 1.0

                    # Since prefix_lengths are sorted, this update will be for the longest prefix.
                    # The positive sample's score is at the current cursor position.
                    score_for_longest_prefix[sample_idx] = prefix_raw_scores_flat[raw_score_cursor]
                    length_for_longest_prefix[sample_idx] = prefix_len
                
                # Move cursor to the start of the next sample's scores in the flat tensor
                raw_score_cursor += meta.get('num_paths', 1)

        decay_rate = self.hparams.loss_decay
        for i, prefix_len in enumerate(prefix_lengths):
            weights[:, i] = torch.exp(-decay_rate * (prefix_lengths[-1] - prefix_len))
        
        scaled_z_scores = torch.asinh(z_scores) if self.scale_loss else z_scores
        
        adjusted_z_scores = scaled_z_scores * (2 * labels.unsqueeze(1) - 1)
        
        weighted_sum = (weights * mask * adjusted_z_scores).sum(dim=1)
        weight_sum = (weights * mask).sum(dim=1) + 1e-8
        
        weighted_avg = weighted_sum / weight_sum
        
        loss = -weighted_avg.mean()

        # --- Create batch_items for evaluation ---
        batch_items = []
        for i, meta in enumerate(all_samples):
            u = meta.get('u', -1).item() if hasattr(meta.get('u', -1), 'item') else meta.get('u', -1)
            v = meta.get('v', -1).item() if hasattr(meta.get('v', -1), 'item') else meta.get('v', -1)
            ts = meta.get('ts', -1).item() if hasattr(meta.get('ts', -1), 'item') else meta.get('ts', -1)
            edge_type = meta.get('edge_type')
            if hasattr(edge_type, 'item'):
                edge_type = edge_type.item()
                
            # Get v_pos value, defaulting to v for positive edges
            v_pos = meta.get('v_pos', v)
            if hasattr(v_pos, 'item'):
                v_pos = v_pos.item()

            item = {
                'u': u,
                'v': v,
                'v_pos': v_pos,  # Add v_pos to the item
                'ts': ts,
                'edge_type': edge_type,
                'label': labels[i].item(),
                'loss': -weighted_avg[i],  # Keep as tensor for epoch-end aggregation
                'score': score_for_longest_prefix[i].item(),
                'length': length_for_longest_prefix[i].item(),
            }
            batch_items.append(item)
        
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

        from collections import defaultdict

        # --- Aggregate all items and losses from outputs ---
        all_items = []
        pos_losses = []
        neg_losses = []
        for output in tqdm(outputs, desc=f"Processing {stage} outputs"):
            if output and 'items' in output:
                for item in output['items']:
                    all_items.append(item)
                    if 'loss' in item and item['loss'] is not None:
                        if item['label'] == 1:
                            pos_losses.append(item['loss'])
                        else:
                            neg_losses.append(item['loss'])
        
        if not all_items:
            print(f"Warning: No items found in {stage} outputs. Skipping evaluation.")
            return

        # --- Log epoch-level losses ---
        if pos_losses:
            pos_epoch_loss = torch.stack(pos_losses).mean()
            self.log(f'{stage}_pos_loss', pos_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
        if neg_losses:
            neg_epoch_loss = -torch.stack(neg_losses).mean()
            self.log(f'{stage}_neg_loss', neg_epoch_loss, prog_bar=True, on_step=False, on_epoch=True)
        if pos_losses or neg_losses:
            total_loss = torch.stack(pos_losses + neg_losses).mean()
            self.log(f'{stage}_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
            print(f"[{stage.upper()}] Loss: {total_loss:.4f}")

        # --- Group scores by edge for evaluation ---
        edge_groups = defaultdict(lambda: {'pos_score': None, 'neg_scores': []})
        for item in all_items:
            # Use v_pos as the target node for grouping, which is the true target for all edges
            v_for_grouping = item.get('v_pos', item['v'])
            key = (item['u'], item.get('edge_type'), v_for_grouping, item['ts'])
            score = item['score']
            
            if item['label'] == 1:
                edge_groups[key]['pos_score'] = score
            else:
                edge_groups[key]['neg_scores'].append(score)

        pos_scores_list = []
        neg_scores_list = []
        for group in edge_groups.values():
            if group['pos_score'] is not None and group['neg_scores']:
                pos_scores_list.append(group['pos_score'])
                neg_scores_list.append(group['neg_scores'])

        if pos_scores_list:
            max_neg_len = max(len(negs) for negs in neg_scores_list)
            neg_scores_padded = [negs + [0.0] * (max_neg_len - len(negs)) for negs in neg_scores_list]
            
            pos_scores = torch.tensor(pos_scores_list, device=self.device)
            neg_scores = torch.tensor(neg_scores_padded, device=self.device)

            results = evaluate(pos_scores, neg_scores, verbose=False)
            for k, v in results.items():
                self.log(f'{stage}_{k}', v, on_step=False, on_epoch=True)
            print(f"[{stage.upper()}] MRR: {results.get('mrr', 0):.4f}, Hits@1: {results.get('hits@1', 0):.4f}, Hits@10: {results.get('hits@10', 0):.4f}")
        else:
            print(f"Warning: No positive samples with corresponding negative samples found in {stage}. Skipping metric evaluation.")

        # --- Export raw results to JSON ---
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
        test_prefix = "test" if getattr(self.trainer.datamodule, "test_time", False) else "train"
        export_path = os.path.join(log_dir, f"{test_prefix}_{stage}_{epoch}_raw.json")

        export_items = []
        for item in all_items:
            export_item = item.copy()
            if isinstance(export_item.get('loss'), torch.Tensor):
                export_item['loss'] = export_item['loss'].item()
            export_items.append(export_item)

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

    def predict(self, batch):
        """
        Public method for prediction that returns z_scores and meta_info.
        
        Args:
            batch: dict: {prefix_length: [list of samples]} from collate_by_prefix_length
            
        Returns:
            z_scores tensor and meta_info
        """
        return self._predict(batch)

# Example usage of the new prediction format:
#
# # For prefix-grouped batch format:
# batch_dict = {
#     "2": [sample1, sample2],  # samples with prefix length 2
#     "3": [sample1, sample3],  # samples with prefix length 3
# }
# z_scores, meta = model.predict_with_new_format(batch_dict)
# 
# # For legacy list format:
# batch_list = [sample1, sample2, sample3]
# z_scores, meta = model.predict_with_new_format(batch_list)
#
# # z_scores will be a tensor of shape (num_unique_samples,) 
# # containing z-scores for each sample

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
        'loss_decay': cfg.get('loss_decay', 0.5),
    }
    
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
    train_logger = WandbLogger(
        name=f"{embedding_model}_{dataset_name}_train",
        save_dir=log_dir,
        project=cfg.get("wandb_project", "thesis-graph"),
        entity=cfg.get("wandb_entity", None),
        log_model=True
    )

    ckpt = ModelCheckpoint(monitor='val_loss', save_top_k=cfg.get('num_ckpt', 1), dirpath=log_dir, mode='min')
    trainer = Trainer(max_epochs=max_epochs, callbacks=[ckpt], logger=train_logger)
    trainer.fit(model, dm)

    # Run the test stage after training is complete, using the best checkpoint
    print("\n--- Running Test Stage ---")
    trainer.test(model, datamodule=dm, ckpt_path='best')
    
    # Test logger
    test_logger = WandbLogger(
        name=f"{embedding_model}_{dataset_name}_test",
        save_dir=log_dir,
        project=cfg.get("wandb_project", "thesis-graph"),
        entity=cfg.get("wandb_entity", None),
        log_model=True
    )
    
    # Run the test-time evaluation
    print("\n--- Running Test-Time Evaluation ---")
    test_epochs = cfg.get('test_time', 0)
    if test_epochs > 0:
        dm.test_time = True
        test_trainer = Trainer(max_epochs=test_epochs, logger=test_logger, callbacks=[ckpt])
        test_trainer.fit(model, dm)


if __name__ == '__main__':
    main()

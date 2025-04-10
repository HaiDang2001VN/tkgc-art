import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

# Import your custom modules
from model import TransformerModel   # your Transformer model implementation
from dist import compute_distance             # custom loss module
from data import TemporalDataset     # your dataset class
from ogb.linkproppred import Evaluator  # Optional: for evaluation

############################################
# Custom Collate Function
############################################


def custom_collate_fn(batch):
    """
    Custom collate function to combine batch items for the TemporalDataset.
    Assumes each item is a dictionary with keys 'paths', 'masks', 'labels', 'tokens'.
    """
    batch_data = {
        'paths': torch.stack([item['paths'] for item in batch]),
        'masks': torch.stack([item['masks'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'tokens': torch.stack([item['tokens'] for item in batch]),
    }
    
    if "label" in batch[0]:
        batch_data["label"] = torch.stack([item['label'] for item in batch])
    
    return batch_data

############################################
# Lightning Module for Model & Training
############################################


class LinkPredTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # save entire config in hparams for logging and checkpointing
        self.save_hyperparameters(config)
        self.config = config
        self.dataset_name = config["data"].get("name", "ogbl-collab")

        # Use the "DGT" model hyperparameters from config
        dgt_conf = config["models"]["DGT"]
        # For example, we assume your TransformerModel constructor takes the following arguments:
        # vocab_size, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
        # dropout, max_seq_length.
        # Note: here we set vocab_size from the data config (if available); here we assume it's provided
        # by the dataset's property. Otherwise you might adjust accordingly.
        # default to 10000 if not provided
        vocab_size = config["data"].get("num_nodes", 10000)
        # Here embed_dim equals d_model in config
        embed_dim = dgt_conf["d_model"]
        nhead = dgt_conf["nhead"]
        num_layers = dgt_conf["num_layers"]
        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers // 2
        dim_ffn = dgt_conf["dim_ffn"]

        self.model = TransformerModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ffn,
            dropout=0.1,
            max_seq_length=config["training"]["k_hops"] + 1
        )
        
        # Use the distance metric from the config
        self.distance_metric = config["training"].get("similarity_type", "cosine")
        
        self.validation_dict = None
        
    def forward(self, src_tokens, tgt_tokens):
        """
        Forward pass expects src_tokens and tgt_tokens of shape [batch, seq_len, embed_dim].
        TransformerModel expects [seq_len, batch, embed_dim] so we permute accordingly.
        """
        src = src_tokens.permute(1, 0, 2)
        tgt = tgt_tokens.permute(1, 0, 2)
        outputs = self.model(src, tgt)
        # Permute back to shape [batch, seq_len, vocab_size] (or embed_dim if using embeddings)
        return outputs.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        # Retrieve batch items and move them to the proper device
        # [batch, seq_len, embed_dim]
        tokens = batch['tokens'].to(self.device)
        # [batch]
        labels = batch['labels'].squeeze(-1).int().to(self.device)
        # Binary masks
        masks = batch['masks'].to(self.device)

        # Separate losses for positive and negative examples
        pos_mask = labels > 0  # Positive examples
        neg_mask = ~pos_mask   # Negative examples

        # Teacher forcing: use tokens[:, :-1] as source and tokens[:, 1:] as target
        src_tokens = tokens[:, :-1]
        tgt_tokens = tokens[:, 1:]

        # Forward pass (model expects [seq_len, batch, embed_dim])
        # shape: [batch, seq_len-1, output_dim]
        pred = self.forward(src_tokens, tgt_tokens)

        # Calculate per-sample losses without aggregation
        dist = distance(pred, tgt_tokens, masks)

        # Get the losses for positive and negative examples, only 1 positive per batch
        pos_dist = dist[pos_mask].mean()
        neg_dist = dist[neg_mask]

        # Mean and std of negative examples
        mean_neg = neg_dist.mean()
        std_neg = neg_dist.std() + 1e-6
        
        # Calculate the z-like loss
        loss = pos_dist - mean_neg / std_neg
        
        # Log the loss
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        
        # Return the loss
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_dict = {
            "y_pred": [],
            "y_label": []
        }
    
    def validation_step(self, batch, batch_idx):
        tokens = batch['tokens'].to(self.device)
        labels = batch['labels'].squeeze(-1).int().to(self.device)
        # Binary masks
        masks = batch['masks'].to(self.device)
        # Central label
        label = batch['label'].squeeze(-1).int().to(self.device)

        # Separate losses for positive and negative examples
        pos_mask = labels > 0  # Positive examples
        neg_mask = ~pos_mask   # Negative examples

        src_tokens = tokens[:, :-1]
        tgt_tokens = tokens[:, 1:]

        pred = self.forward(src_tokens, tgt_tokens)
        # For validation, use a version of loss with all-ones (raw distance) if desired;
        # Calculate per-sample losses without aggregation
        dist = distance(pred, tgt_tokens, masks)

        # Get the losses for positive and negative examples, only 1 positive per batch
        pos_dist = dist[pos_mask].mean()
        neg_dist = dist[neg_mask]
        
        # Compute quantile of central edge according to losses
        high_neg_count = (neg_dist > pos_dist).sum()
        pos_quantile = (high_neg_count + 1) / neg_dist.size(0)
        
        # Store the predictions and labels
        self.validation_dict["y_pred"].append(pos_quantile)
        self.validation_dict["y_label"].append(label)
        
        # Compute quantile loss
        mean_neg = neg_dist.mean()
        std_neg = neg_dist.std() + 1e-6
        loss = pos_dist - mean_neg / std_neg
        
        # Log the loss
        self.log("val_loss", loss, prog_bar=True)
        
        return {"val_loss": loss, "labels": label}
    
    def on_validation_epoch_end(self, outputs):
        # Concatenate all predictions and labels
        y_pred = torch.cat(self.validation_dict["y_pred"])
        y_label = torch.cat(self.validation_dict["y_label"])
        
        # Use OGB Evaluator
        evaluator = Evaluator(name=self.dataset_name)
        eval_dict = {
            "y_pred_pos": y_pred[y_label == 1],
            "y_pred_neg": y_pred[y_label == 0],
        }
        
        result_dict = evaluator.eval(eval_dict)
        
        # Log the evaluation results
        for metric_name, value in result_dict.items():
            self.log(f"val_{metric_name}", value, prog_bar=True)
        return result_dict

    def configure_optimizers(self):
        lr = self.config["training"].get("lr", 1e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


############################################
# Lightning Data Module for Data Loading
############################################

class TemporalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Initialize training and validation datasets with the TemporalDataset class.
        self.train_dataset = TemporalDataset(self.config, k=3, m_d=50)
        self.train_dataset.split = 'train'
        self.val_dataset = TemporalDataset(self.config, k=3, m_d=50)
        self.val_dataset.split = 'test'

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            persistent_workers=self.config["training"].get(
                "persistent_workers", False),
            pin_memory=self.config["training"].get("pin_memory", False),
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            persistent_workers=self.config["training"].get(
                "persistent_workers", False),
            pin_memory=self.config["training"].get("pin_memory", False),
            collate_fn=custom_collate_fn
        )

############################################
# Main function: argument parsing & training
############################################


def main():
    parser = ArgumentParser(
        description="Lightning Trainer for Temporal Graph Transformer")
    # Only one argument is required: --config (path to config JSON)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file (JSON)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run in evaluation-only mode")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained checkpoint for evaluation")
    args = parser.parse_args()

    # Read configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create data module from config
    dm = TemporalDataModule(config)

    # Initialize Lightning module for model and loss
    model = LinkPredTrainer(config)

    # Load model weights if eval_only mode and model_path is provided
    if args.eval_only and args.model_path:
        print(f"Loading model checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Create a ModelCheckpoint callback using configuration from config["training"]
    ckpt_callback = ModelCheckpoint(
        dirpath=config["training"].get("ckpt_dir", "checkpoints"),
        filename=config["training"].get(
            "ckpt_name", "model") + "_epoch{epoch}",
        save_top_k=config["training"].get("ckpt_keep", 1),
        every_n_epochs=config["training"].get("ckpt_freq", 1),
        verbose=True,
    )

    # Initialize the PyTorch Lightning Trainer using config parameters.
    trainer = pl.Trainer(
        max_epochs=config["training"].get("num_epochs", 50),
        accelerator=config["training"].get("accelerator", "auto"),
        devices=1 if torch.cuda.is_available() else None,
        precision=config["training"].get("precision", 32),
        callbacks=[ckpt_callback],
        log_every_n_steps=config["training"].get("log_freq", 5),
        default_root_dir=config["training"].get("log_dir", "../model")
    )

    if not args.eval_only:
        trainer.fit(model, datamodule=dm)

    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()

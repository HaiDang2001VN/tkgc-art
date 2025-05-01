import os
import json
import argparse

import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn.kge import DistMult, ComplEx, RotatE, TransE


class KGEModelProxy(nn.Module):
    MODEL_MAP = {
        'distmult': DistMult,
        'complex': ComplEx,
        'rotate': RotatE,
        'transe': TransE
    }

    def __init__(self, cfg: dict, state_dict_path: str = None):
        super().__init__()

        self.cfg = cfg
        self.model_name = cfg.get('model_name', 'transe').lower()
        if self.model_name not in self.MODEL_MAP:
            raise ValueError(
                f"Model {self.model_name} not supported. Choose from: {list(self.MODEL_MAP.keys())}")

        # Extract common parameters from cfg
        num_nodes = cfg.get('num_nodes')
        num_relations = cfg.get('num_relations', 1)
        hidden_channels = cfg.get('hidden_channels', 100)

        # Prepare model-specific parameters
        model_args = {}

        # Common optional parameter across models
        sparse = cfg.get('sparse', False)
        model_args['sparse'] = sparse

        # Model-specific parameters
        if self.model_name == 'distmult':
            model_args['margin'] = cfg.get('margin', 1.0)
        elif self.model_name == 'complex':
            pass
        elif self.model_name == 'transe':
            model_args['margin'] = cfg.get('margin', 1.0)
            model_args['p_norm'] = cfg.get('p_norm', 2)
        elif self.model_name == 'rotate':
            model_args['margin'] = cfg.get('margin', 1.0)

        # Initialize model
        model_class = self.MODEL_MAP[self.model_name]
        self.model = model_class(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **model_args
        )

        # Load pretrained weights if provided
        if state_dict_path:
            try:
                state_dict = torch.load(state_dict_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"Loaded model state from {state_dict_path}")
            except Exception as e:
                print(f"Error loading state dict: {e}")

        self.model.eval()

    def forward(self, batched_paths: Tensor) -> Tensor:
        heads = batched_paths[:, -2]
        rels = batched_paths[:, -1]
        tails = batched_paths[:, 0]
        return self.model(heads, rels, tails)

    def train_epoch(self, loader, optimizer, device):
        self.model.train()
        total_loss = 0.0
        for heads, rels, tails in loader:
            heads, rels, tails = heads.to(
                device), rels.to(device), tails.to(device)
            optimizer.zero_grad()
            loss = self.model.loss(heads, rels, tails)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader, device):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for heads, rels, tails in loader:
                heads, rels, tails = heads.to(
                    device), rels.to(device), tails.to(device)
                loss = self.model.loss(heads, rels, tails)
                total_loss += loss.item()
        return total_loss / len(loader)

    @classmethod
    def train_model(cls,
                    train_triples: torch.Tensor,
                    val_triples: torch.Tensor = None,
                    cfg: dict = None,
                    device: torch.device = None,
                    name_suffix: str = ''):
        if cfg is None:
            cfg = {}
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = cfg.get('batch_size', 1024)
        # Train DataLoader
        heads = train_triples[:, 0]
        rels = train_triples[:, 1]
        tails = train_triples[:, 2]
        train_ds = TensorDataset(heads, rels, tails)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)

        # Val DataLoader
        val_loader = None
        if val_triples is not None:
            heads_v = val_triples[:, 0]
            rels_v = val_triples[:, 1]
            tails_v = val_triples[:, 2]
            val_ds = TensorDataset(heads_v, rels_v, tails_v)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False)

        # Instantiate proxy
        proxy = cls(cfg=cfg).to(device)
        optimizer = Adam(proxy.model.parameters(), lr=cfg.get('lr', 0.01))

        best_val_loss = float('inf')
        best_state = None
        patience = 0
        early_stop = cfg.get('early_stopping', float('inf'))

        print(f"Training {name_suffix} model...")
        for epoch in range(1, cfg.get('epochs', 100) + 1):
            train_loss = proxy.train_epoch(train_loader, optimizer, device)
            if val_loader is not None:
                val_loss = proxy.evaluate(val_loader, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = proxy.model.state_dict()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
                print(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")
                best_state = proxy.model.state_dict()

        if best_state:
            proxy.model.load_state_dict(best_state)
        embeddings = proxy.model.node_emb.weight.detach().cpu()
        return proxy, embeddings


def main(config_path: str):
    # Load main config and embedding config
    with open(config_path) as f:
        main_cfg = json.load(f)
    embed_cfg_path = main_cfg.get('embedding_config')
    if not embed_cfg_path:
        raise KeyError("'embedding_config' missing in main config.")
    with open(embed_cfg_path) as f:
        embed_cfg = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read CSV edges
    csv_file = os.path.join(
        main_cfg['storage_dir'], f"{main_cfg['dataset']}_edges.csv")
    df = pd.read_csv(csv_file)
    df_pos = df[df['label'] == 1]

    def triples_for(split: str) -> torch.Tensor:
        d = df_pos[df_pos['split'] == split]
        return torch.tensor(d[['u', 'edge_type', 'v']].values, dtype=torch.long)

    pre = triples_for('pre')
    train = triples_for('train')
    valid = triples_for('valid')
    test = triples_for('test')

    num_nodes = int(df[['u', 'v']].max().max()) + 1
    num_relations = int(df_pos['edge_type'].nunique())
    embed_cfg['num_nodes'] = num_nodes
    embed_cfg['num_relations'] = num_relations

    partitions = [
        ('train', pre, train),
        ('val', torch.cat([pre, train], dim=0), valid),
        ('test', torch.cat([pre, train, valid], dim=0), test),
    ]

    for name, tr, val in partitions:
        proxy_model, embeddings = KGEModelProxy.train_model(
            train_triples=tr,
            val_triples=val,
            cfg=embed_cfg,
            device=device,
            name_suffix=name
        )
        store = main_cfg.get('store', 'embedding')
        suffix = '_embeddings.pt' if store == 'embedding' else '_model.pt'
        model_name = embed_cfg.get('model_name', 'model')
        out_name = f"{model_name}_{main_cfg['dataset']}_{name}{suffix}"
        out_path = os.path.join(main_cfg['storage_dir'], out_name)
        if store == 'embedding':
            torch.save(embeddings, out_path)
            print(f"Saved {name} embeddings to {out_path}")
        else:
            torch.save(proxy_model.model.state_dict(), out_path)
            print(f"Saved {name} model state to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train KGE from CSV via Proxy")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)

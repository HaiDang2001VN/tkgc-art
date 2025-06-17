import os
import json
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn.kge import DistMult, ComplEx, RotatE, TransE
from utils import load_configuration
import numpy as np


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

        num_nodes = int(cfg.get('num_nodes'))        
        num_relations = int(cfg.get('num_relations', 1))
        hidden_channels = int(cfg.get('hidden_channels', 100))
        model_args = {'sparse': cfg.get('sparse', False)}

        if self.model_name in ['distmult', 'transe', 'rotate']:
            model_args['margin'] = float(cfg.get('margin', 1.0))
        if self.model_name == 'transe':
            model_args['p_norm'] = int(cfg.get('p_norm', 2))

        model_class = self.MODEL_MAP[self.model_name]
        self.model = model_class(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **model_args
        )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"KGEModelProxy initialized on {self.device}")

        if state_dict_path:
            try:
                state_dict_map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
                if state_dict_path.endswith('_embeddings.pt'):
                    embeddings = torch.load(
                        state_dict_path, map_location=state_dict_map_location)
                    self.model.node_emb.weight = nn.Parameter(embeddings)
                    print(
                        f"Loaded embeddings from {state_dict_path} to {self.device}")
                else:
                    state_dict = torch.load(
                        state_dict_path, map_location=state_dict_map_location)
                    self.model.load_state_dict(state_dict)
                    print(
                        f"Loaded model state from {state_dict_path} to {self.device}")
            except Exception as e:
                print(f"Error loading state dict: {e}")
        self.model.eval()

    def save_embeddings_as_text(self, out_prefix: str, storage_dir: str):
        """
        Saves node and relation embeddings to word2vec-compatible text files.
        The format is a header line "num_embeddings embedding_dim" followed by embeddings.
        """
        def _save_tensor_as_text(tensor, file_path):
            numpy_array = tensor.detach().cpu().numpy()
            num_vectors, dim = numpy_array.shape
            header = f"{num_vectors} {dim}"
            np.savetxt(file_path, numpy_array, fmt='%.8f',
                       header=header, comments='')

        # Save Node Embeddings
        node_path = os.path.join(storage_dir, f"{out_prefix}_nodes.txt")
        print(f"Saving node embeddings to text file: {node_path}")
        _save_tensor_as_text(self.model.node_emb.weight, node_path)

        # Save Relation Embeddings (if they exist)
        if hasattr(self.model, 'rel_emb') and self.model.rel_emb is not None:
            rel_path = os.path.join(storage_dir, f"{out_prefix}_relations.txt")
            print(f"Saving relation embeddings to text file: {rel_path}")
            _save_tensor_as_text(self.model.rel_emb.weight, rel_path)
        else:
            print(
                f"No relation embeddings found for model {self.model_name}. Skipping text save for relations.")

    def norm(self, tensor: Tensor, dim: int = -1) -> Tensor:
        if hasattr(self.model, 'p_norm'):
            return F.normalize(tensor, p=self.model.p_norm, dim=dim)
        elif self.model_name == 'rotate':
            return torch.linalg.vector_norm(tensor, dim=dim)
        else:
            return tensor.norm(p=2, dim=dim)

    def forward(self, batched_paths: Tensor) -> Tensor:
        heads = batched_paths[:, -1]
        rels = batched_paths[:, -2]
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
    def train_model(cls, train_triples: torch.Tensor, val_triples: torch.Tensor = None, cfg: dict = None, device: torch.device = None, name_suffix: str = ''):
        if cfg is None:
            cfg = {}
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = cfg.get('batch_size', 1024)
        train_ds = TensorDataset(
            train_triples[:, 0], train_triples[:, 1], train_triples[:, 2])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_triples is not None:
            val_ds = TensorDataset(
                val_triples[:, 0], val_triples[:, 1], val_triples[:, 2])
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False)

        proxy = cls(cfg=cfg).to(device)
        optimizer = Adam(proxy.model.parameters(), lr=cfg.get('lr', 0.01))
        best_val_loss, best_state, patience = float('inf'), None, 0
        early_stop = cfg.get('early_stopping', float('inf'))

        print(f"Training {name_suffix} model...")
        for epoch in range(1, cfg.get('epochs', 100) + 1):
            train_loss = proxy.train_epoch(train_loader, optimizer, device)
            if val_loader is not None:
                val_loss = proxy.evaluate(val_loader, device)
                print(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss, best_state, patience = val_loss, proxy.model.state_dict(), 0
                else:
                    patience += 1
                    if patience >= early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")
                best_state = proxy.model.state_dict()

        if best_state:
            proxy.model.load_state_dict(best_state)
        embeddings = proxy.model.node_emb.weight.detach().cpu()
        return proxy, embeddings


def main(config_path: str):
    main_cfg = load_configuration(config_path)
    embed_cfg_path = main_cfg.get('embedding_config')
    if not embed_cfg_path:
        raise KeyError("'embedding_config' missing in main config.")

    with open(embed_cfg_path) as f:
        embed_cfg = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_file = os.path.join(
        main_cfg['storage_dir'], f"{main_cfg['dataset']}_edges.csv")
    df = pd.read_csv(csv_file)
    df_pos = df[df['label'] == 1]

    def triples_for(split: str) -> torch.Tensor:
        d = df_pos[df_pos['split'] == split]
        return torch.tensor(d[['u', 'edge_type', 'v']].values, dtype=torch.long)

    partitions = {
        'train': (triples_for('pre'), triples_for('train')),
        'valid': (torch.cat([triples_for('pre'), triples_for('train')]), triples_for('valid')),
        'test': (torch.cat([triples_for('pre'), triples_for('train'), triples_for('valid')]), triples_for('test'))
    }

    num_nodes = int(df[['u', 'v']].max().max()) + 1
    num_relations = int(df_pos['edge_type'].nunique())
    embed_cfg.update({'num_nodes': num_nodes, 'num_relations': num_relations})
    model_name_from_embed_cfg = embed_cfg.get('model_name', 'model')

    for name, (tr, val) in partitions.items():
        proxy_model, embeddings = KGEModelProxy.train_model(
            train_triples=tr, val_triples=val, cfg=embed_cfg, device=device, name_suffix=name
        )

        store = main_cfg.get('store', 'embedding')
        suffix = '_embeddings.pt' if store == 'embedding' else '_model.pt'
        out_prefix = f"{model_name_from_embed_cfg}_{main_cfg['dataset']}_{name}"
        out_name = f"{out_prefix}{suffix}"
        out_path = os.path.join(main_cfg['storage_dir'], out_name)

        if store == 'embedding':
            torch.save(embeddings, out_path)
            print(f"Saved {name} embeddings to {out_path}")
        else:
            torch.save(proxy_model.model.state_dict(), out_path)
            print(f"Saved {name} model state to {out_path}")

        # NEW: Save embeddings as text if configured
        if main_cfg.get('save_text_embeddings', False):
            proxy_model.save_embeddings_as_text(
                out_prefix, main_cfg['storage_dir'])

        embed_cfg['model_name'] = model_name_from_embed_cfg
        config_out_name = f"{out_prefix}_config.json"
        config_out_path = os.path.join(
            main_cfg['storage_dir'], config_out_name)
        with open(config_out_path, 'w') as f_out:
            json.dump(embed_cfg, f_out, indent=4)
        print(f"Saved {name} embedding config to {config_out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train KGE from CSV via Proxy")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)

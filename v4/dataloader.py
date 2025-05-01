#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Import the KGE proxy for shallow embedding extraction
from embedding import KGEModelProxy


class PathDataset(Dataset):
    def __init__(self, cfg: dict, partition: str):
        # Load main config and partition
        self.cfg = cfg
        self.partition = partition
        storage_dir = cfg.get('storage_dir', '.')
        dataset = cfg['dataset']

        # Load edges CSV and filter by partition
        edges_fp = os.path.join(storage_dir, f"{dataset}_edges.csv")
        self.df = pd.read_csv(edges_fp, index_col='edge_id')
        self.df = self.df[self.df['split'] == partition].copy()

        # Load shortest (positive) paths
        paths_fp = os.path.join(storage_dir, 'paths.json')
        with open(paths_fp) as f:
            self.pos_paths = json.load(f)

        # Load negative paths
        neg_fp = os.path.join(
            storage_dir,
            f"{cfg.get('model_name', 'transe')}_{dataset}_{partition}_neg.json"
        )
        with open(neg_fp) as f:
            self.neg_paths = json.load(f)

        # Load node features if available
        feat_fp = os.path.join(storage_dir, f"{dataset}_features.pt")
        if os.path.exists(feat_fp):
            fm = torch.load(feat_fp)
            self.features_map = fm if isinstance(fm, dict) else {0: fm}
        else:
            self.features_map = None

        # Handle shallow KGE embeddings via proxy if requested
        loader_cfg = cfg.get('loader', {})
        self.use_shallow = loader_cfg.get('shallow', False)
        if self.use_shallow:
            state_path = loader_cfg.get('state_dict_path', None)
            self.kge_proxy = KGEModelProxy(
                loader_cfg, state_dict_path=state_path)
            self.kge_proxy.eval()
        else:
            self.kge_proxy = None

        # (Optional) Load precomputed embeddings or model state
        store = cfg.get('store', 'embedding')
        emb_cfg_path = cfg['embedding_config']
        if not os.path.isabs(emb_cfg_path):
            emb_cfg_path = os.path.join(storage_dir, emb_cfg_path)
        with open(emb_cfg_path) as ef:
            emb_cfg = json.load(ef)
        model_name = emb_cfg.get('model_name', 'model')
        suffix = '_embeddings.pt' if store == 'embedding' else '_model.pt'
        emb_file = os.path.join(
            storage_dir, f"{model_name}_{dataset}_{partition}{suffix}")
        self.precomputed_embed = torch.load(
            emb_file) if os.path.exists(emb_file) else None

        # Prepare edge IDs
        self.edge_ids = self.df.index.tolist()

    def __len__(self):
        return len(self.edge_ids)

    def __getitem__(self, idx):
        eid = self.edge_ids[idx]
        # Positive path nodes
        pos_nodes = self.pos_paths.get(str(eid), {}).get('nodes', [])
        # Negative path nodes
        raw_negs = self.neg_paths.get(str(eid), [])
        neg_nodes = []
        for p in raw_negs:
            # Extract node IDs if interleaved
            if all(isinstance(x, int) for x in p):
                neg_nodes.append(p)
            else:
                neg_nodes.append(p[::2])

        # Combine
        all_paths = [pos_nodes] + neg_nodes
        path_tensor = torch.tensor(all_paths, dtype=torch.long)
        item = {'paths': path_tensor}

        # Shallow KGE embeddings per path
        if self.use_shallow:
            emb_list = []
            for nodes in all_paths:
                node_ids = torch.tensor(nodes, dtype=torch.long)
                # n_nodes x emb_dim
                emb_list.append(self.kge_proxy.model.node_emb(node_ids))
            # (1+neg) x path_len x emb_dim
            item['shallow_emb'] = torch.stack(emb_list, dim=0)

        # Precomputed embeddings/model state
        if self.precomputed_embed is not None:
            item['precomputed_embed'] = self.precomputed_embed

        # Node features
        if self.features_map is not None:
            feats = []
            fmap = self.features_map
            for nodes in all_paths:
                feats.append(torch.stack([fmap[0][n] for n in nodes], dim=0))
            item['features'] = torch.stack(feats, dim=0)

        return item


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch DataLoader for path-based temporal graph dataset"
    )
    parser.add_argument('--config', type=str, required=True,
                        help="Path to main config.json")
    parser.add_argument('--partition', type=str,
                        choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    dataset = PathDataset(cfg, args.partition)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        pin_memory=True
    )
    # Sanity check
    for batch in loader:
        print(batch['paths'].shape)
        if 'shallow_emb' in batch:
            print(batch['shallow_emb'].shape)
        if 'features' in batch:
            print(batch['features'].shape)
        break


if __name__ == '__main__':
    main()

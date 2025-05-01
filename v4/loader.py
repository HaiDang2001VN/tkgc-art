#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

# Proxy for extracting shallow embeddings
from embedding import KGEModelProxy


class EdgeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pos_paths: dict,
        neg_paths: dict,
        features_map: dict | None,
        kge_proxy: KGEModelProxy | None
    ):
        self.edge_ids = df.index.tolist()
        self.pos_paths = pos_paths
        self.neg_paths = neg_paths
        self.features_map = features_map
        self.kge_proxy = kge_proxy

    def __len__(self):
        return len(self.edge_ids)

    def __getitem__(self, idx):
        eid = self.edge_ids[idx]
        # Positive path nodes
        pos = self.pos_paths.get(str(eid), {}).get('nodes', [])
        # Negative path nodes
        raw_negs = self.neg_paths.get(str(eid), [])
        negs = []
        for p in raw_negs:
            if all(isinstance(x, int) for x in p):
                negs.append(p)
            else:
                negs.append(p[::2])
        all_paths = [pos] + negs

        # Paths tensor: (1+neg) x path_len
        paths = torch.tensor(all_paths, dtype=torch.long)
        item = {'paths': paths}

        # Node features if available: (1+neg) x path_len x feat_dim
        if self.features_map is not None:
            nodes_feats = []
            fmap = self.features_map
            for path in all_paths:
                nodes_feats.append(torch.stack(
                    [fmap[0][n] for n in path], dim=0))
            item['features'] = torch.stack(nodes_feats, dim=0)

        # Shallow embeddings if proxy provided
        if self.kge_proxy is not None:
            emb_list = []
            for path in all_paths:
                node_ids = torch.tensor(path, dtype=torch.long)
                emb_list.append(self.kge_proxy.model.node_emb(node_ids))
            item['shallow_emb'] = torch.stack(emb_list, dim=0)

        return item


class PathDataModule(LightningDataModule):
    def __init__(
        self,
        config_path: str,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        super().__init__()
        self.config_path = config_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load config
        cfg = json.load(open(config_path))
        self.cfg = cfg
        self.storage_dir = cfg.get('storage_dir', '.')
        self.dataset = cfg['dataset']

        # Loader-specific settings
        loader_cfg = cfg.get('loader', {})
        self.use_shallow = loader_cfg.get('shallow', False)
        state_dict = loader_cfg.get('state_dict_path', None)
        if self.use_shallow:
            self.kge_proxy = KGEModelProxy(
                loader_cfg, state_dict_path=state_dict)
            self.kge_proxy.eval()
        else:
            self.kge_proxy = None

    def prepare_data(self):
        # No downloads; assumes files already in storage_dir
        pass

    def setup(self, stage: str | None = None):
        # Read and split edges
        edges_fp = os.path.join(self.storage_dir, f"{self.dataset}_edges.csv")
        df = pd.read_csv(edges_fp, index_col='edge_id')
        self.dfs = {
            split: df[df['split'] == split].copy()
            for split in ('train', 'val', 'test')
        }

        # Load paths JSONs
        with open(os.path.join(self.storage_dir, 'paths.json')) as f:
            self.pos_paths = json.load(f)

        def neg_fn(split): return os.path.join(
            self.storage_dir,
            f"{self.cfg.get('model_name','transe')}_{self.dataset}_{split}_neg.json"
        )
        self.neg_paths = {
            split: json.load(open(neg_fn(split)))
            for split in ('train', 'val', 'test')
        }

        # Load features
        feat_fp = os.path.join(self.storage_dir, f"{self.dataset}_features.pt")
        if os.path.exists(feat_fp):
            fm = torch.load(feat_fp)
            self.features_map = fm if isinstance(fm, dict) else {0: fm}
        else:
            self.features_map = None

    def train_dataloader(self):
        return DataLoader(
            EdgeDataset(
                self.dfs['train'], self.pos_paths, self.neg_paths['train'],
                self.features_map, self.kge_proxy
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            EdgeDataset(
                self.dfs['val'], self.pos_paths, self.neg_paths['val'],
                self.features_map, self.kge_proxy
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            EdgeDataset(
                self.dfs['test'], self.pos_paths, self.neg_paths['test'],
                self.features_map, self.kge_proxy
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Lightning DataModule replacing PathDataset"
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    dm = PathDataModule(
        config_path=args.config,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )
    dm.prepare_data()
    dm.setup()

    # Sanity check
    for batch in dm.train_dataloader():
        print(batch['paths'].shape)
        if 'features' in batch:
            print(batch['features'].shape)
        if 'shallow_emb' in batch:
            print(batch['shallow_emb'].shape)
        break

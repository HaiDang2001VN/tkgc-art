#!/usr/bin/env python3
import argparse
import json
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from typing import Union

# Proxy for extracting shallow embeddings
from embedding import KGEModelProxy

# Custom collate that returns list of samples (avoids stacking variable-length tensors)


def collate_to_list(batch: list[dict]) -> list[dict]:
    return batch


class EdgeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pos_paths: dict,
        neg_paths: dict,
        features_map: Union[dict, None],
        kge_proxy: Union[KGEModelProxy, None],
        num_neg: Union[int, None] = None
    ):
        self.df = df
        self.edge_ids = df.index.tolist()
        self.pos_paths = pos_paths
        self.neg_paths = neg_paths
        self.features_map = features_map
        self.kge_proxy = kge_proxy
        self.num_neg = num_neg

    def __len__(self):
        return len(self.edge_ids)

    def __getitem__(self, idx):
        eid = self.edge_ids[idx]
        label = self.df.at[eid, 'label']
        pos = self.pos_paths.get(str(eid), {}).get('nodes')
        
        if pos is None:
            return None
        
        raw_negs = self.neg_paths.get(str(eid), [])
        if self.num_neg is not None and len(raw_negs) > self.num_neg:
            raw_negs = random.sample(raw_negs, self.num_neg)
        negs = []
        for p in raw_negs:
            if all(isinstance(x, int) for x in p):
                negs.append(p)
            else:
                negs.append(p[::2])
        all_paths = [pos] + negs
        item = {'paths': torch.tensor(all_paths, dtype=torch.long)}
        item['label'] = torch.tensor(label, dtype=torch.long)
        if self.features_map is not None:
            feats = []
            fmap = self.features_map
            for path in all_paths:
                ft = torch.stack([torch.tensor(fmap[0][n]) for n in path], dim=0)
                feats.append(ft)
            item['features'] = torch.stack(feats, dim=0)
        if self.kge_proxy is not None:
            embs = []
            for path in all_paths:
                node_ids = torch.tensor(path, dtype=torch.long)
                embs.append(self.kge_proxy.model.node_emb(node_ids))
            item['shallow_emb'] = torch.stack(embs, dim=0)
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
        cfg = json.load(open(config_path))
        self.cfg = cfg
        self.storage_dir = cfg.get('storage_dir', '.')
        self.dataset = cfg['dataset']
        self.num_neg = cfg.get('num_neg', None)
        loader_cfg = cfg.get('loader', {})
        self.loader_cfg = loader_cfg
        # Internal shallow flag
        self._use_shallow = loader_cfg.get('shallow', False)
        self.kge_proxy = None

    @property
    def use_shallow(self) -> bool:
        """Whether shallow embeddings will be used (considering features)."""
        return self._use_shallow

    @property
    def emb_dim(self) -> int:
        """
        Returns the total embedding dimension: feature dim + shallow KGE dim (if used).
        """
        dim = 0
        if hasattr(self, 'features_map') and self.features_map is not None:
            first = next(iter(self.features_map.values()))
            dim += first.shape[1]
        if self.use_shallow and self.kge_proxy is not None:
            emb_dim = self.loader_cfg.get(
                'hidden_channels') or self.kge_proxy.cfg.get('hidden_channels', 0)
            dim += emb_dim
        return dim

    def prepare_data(self):
        pass

    def setup(self, stage: Union[str, None] = None):
        edges_fp = os.path.join(self.storage_dir, f"{self.dataset}_edges.csv")
        df = pd.read_csv(edges_fp, index_col='edge_id')
        self.dfs = {s: df[df['split'] == s].copy()
                    for s in ('train', 'val', 'test')}
        with open(os.path.join(self.storage_dir, 'paths.json')) as f:
            self.pos_paths = json.load(f)

        def neg_fn(s): return os.path.join(self.storage_dir,
                                           f"{self.cfg.get('model_name','transe')}_{self.dataset}_{s}_neg.json")
        self.neg_paths = {s: json.load(open(neg_fn(s)))
                          for s in ('train', 'val', 'test')}
        feat_fp = os.path.join(self.storage_dir, f"{self.dataset}_features.pt")
        if os.path.exists(feat_fp):
            fm = torch.load(feat_fp, weights_only=False)
            if isinstance(fm, dict):
                first = next(iter(fm.values()))
                fmap = fm
            else:
                first = fm
                fmap = {0: fm}
            if first.shape[1] == 0:
                self.features_map = None
            else:
                self.features_map = fmap
        else:
            self.features_map = None
        # force shallow if no features
        if self.features_map is None:
            self._use_shallow = True
        if self.use_shallow and self.kge_proxy is None:
            state = self.loader_cfg.get('state_dict_path', None)
            self.kge_proxy = KGEModelProxy(
                self.loader_cfg, state_dict_path=state)
            self.kge_proxy.eval()

    def _dataloader(self, split: str, shuffle: bool):
        ds = EdgeDataset(
            self.dfs[split], self.pos_paths, self.neg_paths[split],
            self.features_map, self.kge_proxy, num_neg=self.num_neg
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          pin_memory=True, collate_fn=collate_to_list)

    def train_dataloader(self):
        return self._dataloader('train', self.shuffle)

    def val_dataloader(self):
        return self._dataloader('val', False)

    def test_dataloader(self):
        return self._dataloader('test', False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DataModule with use_shallow property")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    dm = PathDataModule(args.config, args.batch_size, args.shuffle)
    dm.prepare_data()
    dm.setup()
    print(f"Use shallow: {dm.use_shallow}")
    print(f"Embedding dimension: {dm.emb_dim}")

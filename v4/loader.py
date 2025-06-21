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
import multiprocessing as mp
from tqdm import tqdm

# Proxy for extracting shallow embeddings
from embedding import KGEModelProxy


# Custom collate function remains unchanged
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
        label = self.df.at[eid, 'label'].astype(int)
        # Positive paths are expected to be lists of nodes already
        pos_nodes = self.pos_paths.get(str(eid), {}).get('nodes')
        
        item = {}
        # Create label tensor on CPU
        item['label'] = torch.tensor(label, dtype=torch.long)  # CPU default
        
        if pos_nodes is None: # If no positive path, skip this item
            return item # Still returns the label in order for later evaluation if needed
        
        # Negative paths from preprocess.cpp are interleaved: [node0, edge_type1, node1, ...]
        raw_neg_interleaved_paths = self.neg_paths.get(str(eid), [])
        
        if self.num_neg is not None and len(raw_neg_interleaved_paths) > self.num_neg:
            raw_neg_interleaved_paths = random.sample(raw_neg_interleaved_paths, self.num_neg)
        
        negs_nodes_only = []
        for p_interleaved in raw_neg_interleaved_paths:
            # Extract nodes (elements at even indices)
            # e.g., [n0, r1, n1, r2, n2] -> [n0, n1, n2]
            nodes_in_path = p_interleaved[::2]
            negs_nodes_only.append(nodes_in_path)
            
        # all_paths will be a list of (list of node_ids)
        # e.g., [[pos_n1, pos_n2], [neg1_n1, neg1_n2, neg1_n3], [neg2_n1, neg2_n2]]
        all_paths_nodes_only = [pos_nodes] + negs_nodes_only
        
        
        # Store paths as a list of lists of integers. Tensor conversion (if needed) happens later,
        # possibly after padding in the model or a more sophisticated collate_fn.
        item['paths'] = all_paths_nodes_only
        
        # if self.features_map is not None:
        #     feats_for_all_paths = [] # This will be a list of tensors
        #     fmap = self.features_map # Assuming fmap[0] contains node_id to feature tensor mapping
        #     for node_list_for_one_path in all_paths_nodes_only:
        #         if not node_list_for_one_path: # Handle empty path if it can occur
        #             # Append a zero-size tensor or handle as per model requirements
        #             # For now, let's assume paths are non-empty or model handles it.
        #             # If features are essential, an empty path might be an issue.
        #             # Example: feats_for_all_paths.append(torch.empty((0, feature_dim)))
        #             pass # Or raise error, or skip
                
        #         # Create a tensor of features for the current path
        #         # Each feature fmap[0][n] is already a tensor or array-like
        #         try:
        #             path_features_tensor = torch.stack([torch.as_tensor(fmap[0][n], dtype=torch.float) for n in node_list_for_one_path], dim=0)
        #             feats_for_all_paths.append(path_features_tensor)
        #         except KeyError as e:
        #             # print(f"Warning: Feature key error {e} for eid {eid}. Path: {node_list_for_one_path}. Skipping feature for this path or item.")
        #             # Decide handling: skip item, skip path's features, or use placeholder
        #             return None # Simplest: skip item if features are crucial and missing

        #     # item['features'] is a list of tensors, e.g. [(path1_len, feat_dim), (path2_len, feat_dim)]
        #     item['features'] = feats_for_all_paths

        if self.kge_proxy is not None:
            embs_for_all_paths = [] # This will be a list of tensors
            # Get device of KGE proxy model for intermediate operations
            device = self.kge_proxy.model.device
            
            for node_list_for_one_path in all_paths_nodes_only:
                if not node_list_for_one_path:
                    pass # Similar handling as features for empty paths

                # Create node_ids_tensor on the same device as the KGE model
                node_ids_tensor = torch.tensor(node_list_for_one_path, dtype=torch.long, device=device)
                
                # Wrap the embedding extraction in no_grad context:
                with torch.no_grad():
                    # kge_proxy.model.node_emb should return a tensor of shape (path_len, kge_dim)
                    path_kge_embs_tensor = self.kge_proxy.model.node_emb(node_ids_tensor)
                    
                    # Move embeddings to CPU before adding to list
                    path_kge_embs_tensor = path_kge_embs_tensor.cpu()
                    embs_for_all_paths.append(path_kge_embs_tensor)
            
            # item['shallow_emb'] is a list of tensors, e.g. [(path1_len, kge_dim), (path2_len, kge_dim)]
            item['shallow_emb'] = embs_for_all_paths
            
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
        # Get num_threads for DataLoader num_workers
        self.num_workers = cfg.get('num_threads', mp.cpu_count())
        # Update pre_scan to accept a list of split names
        self.filter_splits = cfg.get('pre_scan', [])
        # Convert to list if a string was provided
        if isinstance(self.filter_splits, str):
            self.filter_splits = [self.filter_splits]
    
        # Internal shallow flag
        self._use_shallow = cfg.get('shallow', False)
        self.df = None
        self.split_map = {}
        self.data = {}
        self.pos_paths = {}
        self.neg_paths = {}
        self.kge_proxy = {}
        self.features_map = {}

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
        # if hasattr(self, 'features_map') and self.features_map[0] is not None:
        #     first = self.features_map[0]
        #     dim += first.shape[1]
        if self.use_shallow and self.kge_proxy["train"] is not None:
            emb_dim = self.cfg.get('hidden_channels') or self.kge_proxy["train"].cfg.get('hidden_channels', 0)
            dim += emb_dim
        return dim

    def prepare_data(self):
        pass

    def setup(self, stage: Union[str, None] = None):
        if stage == "fit":
            print(f"Setting up data for stage: {stage}")
            edges_fp = os.path.join(self.storage_dir, f"{self.dataset}_edges.csv")
            if self.df is None:
                self.df = pd.read_csv(edges_fp, index_col='edge_id')            
                self.split_map = {str(idx): row['split'] for idx, row in self.df.iterrows()}

            for split in ['train', 'valid', 'test']:
                print(f"Setting up data for split: {split}")
                self.data[split] = self.df[self.df['split'] == split].copy()

                pos_paths = {}
                with open(os.path.join(self.storage_dir, f"{self.cfg['dataset']}_paths.txt")) as f:
                    n = int(f.readline())
                    for _ in range(n):
                        eid = f.readline().strip()
                        hops = int(f.readline())
                        nodes = [int(u) for u in f.readline().split()]
                        node_types = [int(t) for t in f.readline().split()]
                        edge_types = f.readline().split()

                        if self.split_map[eid] == split:
                            pos_paths[eid] = {
                                "hops": hops,
                                "nodes": nodes,
                                "node_types": node_types,
                                "edge_types": edge_types
                            }

                self.pos_paths[split] = pos_paths
                neg_fn = os.path.join(self.storage_dir, f"{self.cfg.get('model_name','transe')}_{self.dataset}_{split}_neg.json")
                self.neg_paths[split] = json.load(open(neg_fn))

                # Check if this split should be pre-scanned
                if split in self.filter_splits:
                    print(f"Pre-scan enabled for {split} split. Running full data validation...")
                    self.filter_edges(split)
                else:
                    print(f"Pre-scan not configured for {split} split. Skipping data validation.")

                # Continue with the rest of setup...
                feat_fp = os.path.join(self.storage_dir, f"{self.dataset}_features.pt")
                # if os.path.exists(feat_fp):
                if False:
                    fm = torch.load(feat_fp, weights_only=False)
                    if isinstance(fm, dict):
                        first = next(iter(fm.values()))
                        fmap = fm
                    else:
                        first = fm
                        fmap = {0: fm}
                    if first.shape[1] == 0:
                        self.features_map[split] = None
                    else:
                        self.features_map[split] = fmap
                else:
                    self.features_map[split] = None

                if self.features_map[split] is None:
                    self._use_shallow = True
                
                embedding_config_path = self.cfg.get('embedding_config', 'full_embedding.json')
                print(f"Use shallow embeddings: {self.use_shallow} at config {embedding_config_path}")
                if self.use_shallow and os.path.exists(embedding_config_path):
                    store = self.cfg.get('store', 'embedding')
                    config_suffix = '_config.json'
                    
                    embedding_config = json.load(open(embedding_config_path))
                    model_name = embedding_config.get('model_name', 'transe')

                    config_prefix = f"{model_name}_{self.dataset}_{split}"
                    config_name = f"{config_prefix}{config_suffix}"
                    config_path = os.path.join(self.storage_dir, config_name)

                    state_suffix = "_embeddings.pt" if store == 'embedding' else "_model.pt"
                    state_dict_path = os.path.join(self.storage_dir, f"{config_prefix}{state_suffix}")

                    if os.path.exists(config_path):
                        print(f"Loading KGE model proxy for {split} split from {config_path}")
                        config = json.load(open(config_path))
                        self.kge_proxy[split] = KGEModelProxy(config, state_dict_path=state_dict_path)
                        self.kge_proxy[split].eval()
                        print(f"Device for KGE model: {self.kge_proxy[split].model.device}")

                print(f"Loaded {len(self.data[split])} edges for {split} split.")
        else:
            print(f"Data already setup during fit stage, skipping setup for stage: {stage}")
                    

    def _dataloader(self, split: str, shuffle: bool):
        ds = EdgeDataset(
            self.data[split], self.pos_paths[split], self.neg_paths[split],
            self.features_map[split], self.kge_proxy[split], num_neg=self.num_neg
        )
        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,  # Only enable when using workers
            collate_fn=collate_to_list
        )

    def train_dataloader(self):
        return self._dataloader('train', self.shuffle)

    def val_dataloader(self):
        return self._dataloader('valid', False)

    def test_dataloader(self):
        return self._dataloader('test', False)

    def filter_edges(self, split):
        """
        Pre-scan ALL data points to validate the entire dataset, and filters out invalid edges.
        """
        print(f"\n--- Pre-scanning and filtering {split} data points ---")
        
        initial_edge_count = len(self.data[split])
        edge_ids = self.data[split].index.astype(str).tolist()
        
        print(f"Scanning {initial_edge_count} edges in {split} split...")
        
        valid_edge_ids = []
        results = []
        for eid in tqdm(edge_ids, desc="Scanning edges"):
            # Inline scanning logic (previously in scan_edge_worker)
            result = {
                "valid": False,
                "missing_pos": 0,
                "missing_neg": 0,
                "empty_neg": 0
            }
            
            # Check positive path
            has_valid_pos = False
            if eid in self.pos_paths[split] and self.pos_paths[split][eid].get('nodes'):
                has_valid_pos = True
            else:
                result["missing_pos"] = 1
            
            # Check negative paths
            has_valid_neg = False
            if eid in self.neg_paths[split]:
                if self.neg_paths[split][eid]:  # Check if the list is not empty
                    has_valid_neg = True
                else:
                    result["empty_neg"] = 1
            else:
                result["missing_neg"] = 1
            
            # Check if edge is valid (has both positive and negative paths)
            if has_valid_pos and has_valid_neg:
                result["valid"] = True
                valid_edge_ids.append(eid)
                
            results.append(result)
        
        # Aggregate results
        valid_edges_count = len(valid_edge_ids)
        missing_pos = sum(r["missing_pos"] for r in results)
        missing_neg = sum(r["missing_neg"] for r in results)
        empty_neg = sum(r["empty_neg"] for r in results)
        
        # Calculate statistics
        valid_percent = (valid_edges_count / initial_edge_count) * 100 if initial_edge_count > 0 else 0
        
        # Print summary
        print(f"\nPre-scan Results for {split}:")
        print(f"  Total edges scanned: {initial_edge_count}")
        print(f"  Valid edges (has pos & neg paths): {valid_edges_count} ({valid_percent:.1f}%)")
        print(f"  Missing positive paths: {missing_pos} ({(missing_pos/initial_edge_count)*100:.1f}%)")
        print(f"  Missing negative paths: {missing_neg} ({(missing_neg/initial_edge_count)*100:.1f}%)")
        print(f"  Empty negative paths: {empty_neg} ({(empty_neg/initial_edge_count)*100:.1f}%)")
        
        if valid_edges_count < initial_edge_count:
            print("\n⚠️  WARNING: Some edges are missing required path data!")
            print(f"  Filtering {split} split to keep only {valid_edges_count} valid edges.")
            
            # Filter the data structures
            valid_edge_ids_int = [int(eid) for eid in valid_edge_ids]
            self.data[split] = self.data[split].loc[valid_edge_ids_int]
            self.pos_paths[split] = {eid: self.pos_paths[split][eid] for eid in valid_edge_ids}
            self.neg_paths[split] = {eid: self.neg_paths[split][eid] for eid in valid_edge_ids}
            
            print(f"  New edge count for {split}: {len(self.data[split])}")
        else:
            print("\n✓ All edges have complete path data. No filtering needed.")
        
        print("--- Pre-scan complete ---\n")


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
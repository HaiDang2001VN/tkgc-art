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
import requests  # Add this import at the top if not present

# Proxy for extracting shallow embeddings
from embedding import KGEModelProxy


# Custom collate function that groups samples by prefix length
def collate_by_prefix_length(batch: list[dict]) -> dict:
    """
    Collates batches by prefix length and returns a dict mapping:
    prefix_length -> [node_embeddings, edge_embeddings, meta]
    
    Where:
    - node_embeddings is a tensor containing all node embeddings (positive + negative) for this prefix
    - edge_embeddings is a tensor containing all edge embeddings (positive + negative) for this prefix
    - meta is a list of dicts with metadata for each sample, including:
        - num_paths: number of paths (1 positive + negatives) for this sample
        - u, v, ts, label, edge_type, type_embedding: original sample data
    """
    # Collect all unique prefix lengths across all samples
    all_prefix_lengths = set()
    for item in batch:
        if 'negs_by_prefix_length' in item:
            all_prefix_lengths.update(item['negs_by_prefix_length'].keys())
    
    result = {}
    
    for prefix_len in all_prefix_lengths:
        # Lists to collect all embeddings for this prefix length
        all_node_embs = []
        all_edge_embs = []
        meta_data = []
        
        # Process each sample
        for item in batch:
            eid = item.get('eid', None)
            # if eid == 349288:
            #     print(f"Processing item for eid 349288 with prefix length {prefix_len}: {item['length']}")
            #     print("No pos node embs found")
            
            if 'pos_node_embs' not in item:
                # if eid == 349288:
                #     print("No pos node embs found")
                continue
            
            # Get positive path embeddings and trim them to match the prefix length
            pos_node_embs = item['pos_node_embs']
            pos_edge_embs = item['pos_edge_embs'] if 'pos_edge_embs' in item else None
            length = item['length']
            
            # Trim positive embeddings to match the prefix length for fair comparison
            if pos_node_embs is not None:
                if length > (prefix_len + 1):
                    # Keep only the prefix part of the positive path
                    pos_node_embs = pos_node_embs[:prefix_len + 1]
                elif length < (prefix_len + 1):
                    # If positive path shorter meaning no negative paths, we can skip this sample
                    # if eid == 349288:
                    #     print("Skipping sample with eid 349288 due to shorter positive path than prefix length")
                    continue

            if pos_edge_embs is not None:
                if pos_edge_embs.size(0) > prefix_len:
                    # For edges, we need prefix_len edges to connect prefix_len + 1 nodes
                    pos_edge_embs = pos_edge_embs[:prefix_len]
                elif pos_edge_embs.size(0) < prefix_len:
                    # If positive path edges shorter meaning no negative paths, we can skip this sample
                    # if eid == 349288:
                    #     print("Skipping sample with eid 349288 due to shorter positive path than prefix length")
                    continue

            # Get negative path embeddings for this prefix length
            neg_node_embs = item['neg_node_embs_by_prefix'].get(prefix_len, [])
            neg_edge_embs = item['neg_edge_embs_by_prefix'].get(prefix_len, [])
            
            # Add assertions to verify the trimmed embeddings match the expected structure
            if pos_node_embs is not None and neg_node_embs:
                # After trimming, pos_node_embs should have exactly prefix_len + 1 nodes
                assert pos_node_embs.size(0) == prefix_len + 1, f"Positive node embeddings should have length {prefix_len + 1}, got {pos_node_embs.size(0)}"

                # Each negative path should also have prefix_len + 1 nodes
                for neg_emb in neg_node_embs:
                    assert neg_emb.size(0) == prefix_len + 1, f"Negative node embeddings should have length {prefix_len + 1}, got {neg_emb.size(0)}"
            
            if pos_edge_embs is not None and neg_edge_embs:
                # After trimming, pos_edge_embs should have exactly prefix_len edges
                assert pos_edge_embs.size(0) == prefix_len, f"Positive edge embeddings should have length {prefix_len}, got {pos_edge_embs.size(0)}"

                # Each negative path should also have prefix_len edges
                for neg_emb in neg_edge_embs:
                    if neg_emb.size(0) > 0:  # Only check if there are edge embeddings
                        assert neg_emb.size(0) == prefix_len, f"Negative edge embeddings should have length {prefix_len}, got {neg_emb.size(0)}"
            
            # Combine positive and negative embeddings for this sample
            sample_node_embs = [pos_node_embs] + neg_node_embs
            sample_edge_embs = [pos_edge_embs] + neg_edge_embs if pos_edge_embs is not None else neg_edge_embs
            
            # Add to the collection
            all_node_embs.extend(sample_node_embs)
            all_edge_embs.extend(sample_edge_embs)
            
            # if eid == 349288:
            #     print("Collating sample for eid 349288, pos_node_embs:", len(pos_node_embs), "neg_node_embs:", [len(neg_emb) for neg_emb in neg_node_embs], "prefix_len:", prefix_len, "length:", length)  # noqa: E501length

            # Create metadata entry for this sample with the adjusted path length
            # Since pos_node_embs is already trimmed, we can directly use its length
            meta = {
                'num_paths': len(sample_node_embs),
                'length': length,
            }
            
            # Add edge information to metadata
            for key in ['label', 'u', 'v', 'ts', 'edge_type', 'type_embedding', 'v_pos', 'eid']:
                if key in item:
                    meta[key] = item[key]
            
            meta_data.append(meta)
        
        # Stack all embeddings into a single tensor
        # Since we've ensured all embeddings have the same length for this prefix,
        # we can stack directly without checking or padding
        if all_node_embs:
            # All node embeddings should have the same shape: [prefix_len, embedding_dim]
            try:
                node_embeddings = torch.stack(all_node_embs)
            except RuntimeError as e:
                print(f"Error stacking node embeddings for prefix length {prefix_len}: {e}")
                # Print the shape of embeddings and metadata
                for i, emb in enumerate(all_node_embs):
                    print(f"Node embedding {i} shape: {emb.shape}")
                print(f"Metadata for prefix length {prefix_len}: {meta_data}")
                raise e
        else:
            node_embeddings = torch.tensor([])
        
        # Similarly for edge embeddings - all should have shape [prefix_len-1, embedding_dim]
        if all_edge_embs:
            edge_embeddings = torch.stack(all_edge_embs)
        else:
            edge_embeddings = torch.tensor([])
        
        # Store in result
        result[prefix_len] = [node_embeddings, edge_embeddings, meta_data]
    
    return result


class EdgeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pos_paths: dict,
        neg_paths: dict,
        features_map: Union[dict, None],
        kge_proxy: Union[KGEModelProxy, None],
        split: str = None
    ):
        self.df = df
        self.edge_ids = df.index.tolist()
        self.pos_paths = pos_paths
        self.neg_paths = neg_paths
        self.features_map = features_map
        self.kge_proxy = kge_proxy
        self.split = split

    def __len__(self):
        return len(self.edge_ids)# if self.split != 'train' else 100  # For testing purposes, limit to 100 items in train split

    def __getitem__(self, idx):
        eid = self.edge_ids[idx]
        label = self.df.at[eid, 'label'].astype(int)
        
        # Get positive path info
        pos_path_info = self.pos_paths.get(str(eid), {})
        pos_nodes = pos_path_info.get('nodes')
        pos_edge_types = pos_path_info.get('edge_types', [])
        pos_edge_timestamps = pos_path_info.get('edge_timestamps', [])
        
        item = {'eid': eid}
        # Create label tensor on CPU
        item['label'] = torch.tensor(label, dtype=torch.long)  # CPU default
        
        # Add edge information to the item dictionary
        if 'u' in self.df.columns:
            item['u'] = torch.tensor(self.df.at[eid, 'u'], dtype=torch.long)
        if 'v' in self.df.columns:
            item['v'] = torch.tensor(self.df.at[eid, 'v'], dtype=torch.long)
        
        # Check for timestamp column (could be 'ts' or 'timestamp')
        if 'ts' in self.df.columns:
            item['ts'] = torch.tensor(self.df.at[eid, 'ts'], dtype=torch.long)
        elif 'timestamp' in self.df.columns:
            item['ts'] = torch.tensor(self.df.at[eid, 'timestamp'], dtype=torch.long)
        
        # Extract edge type and its embedding if available
        edge_type = None
        if 'edge_type' in self.df.columns:
            edge_type = int(self.df.at[eid, 'edge_type'])
            item['edge_type'] = torch.tensor(edge_type, dtype=torch.long)
        
        # Add v_pos from the new schema
        if 'v_pos' in self.df.columns:
            v_pos_val = self.df.at[eid, 'v_pos']
            if pd.notna(v_pos_val) and v_pos_val != "None":
                item['v_pos'] = torch.tensor(int(v_pos_val), dtype=torch.long)
                
        # if eid == 349288:
        #     print(f"Debugging item for edge {eid}: {item}")
        #     # Print pos_path_info
        #     print(f"Positive path info: {pos_path_info}")
        #     # Print pos_nodes and pos_edge_types
        #     print(f"Positive nodes: {pos_nodes}")
        #     print(f"Positive edge types: {pos_edge_types}")
            
        # Get length directly from dataframe instead of recalculating
        if 'length' in self.df.columns:
            item['length'] = int(self.df.at[eid, 'length'])
        else:
            # Fallback to original calculation if length column doesn't exist
            if "hops" in pos_path_info:
                item['length'] = pos_path_info["hops"] + 1
                assert item['length'] == len(pos_nodes), "Length mismatch between hops and nodes in positive path"
            elif pos_nodes is not None and len(pos_nodes) > 0:
                item['length'] = len(pos_nodes)
            else:
                item['length'] = 0
        
        if pos_nodes is None: # If no positive path, skip this item
            return item # Still returns the label in order for later evaluation if needed
        
        # Get negative paths from tree-like format
        raw_neg_tree_data = self.neg_paths.get(str(eid), {})
        
        # Process only tree-like format data
        negs_by_prefix_length = {}  # prefix_length -> list of negative paths
        neg_edge_types_by_prefix_length = {}
        neg_timestamps_by_prefix_length = {}
        
        # Process tree-like negative data
        for prefix_len_str, candidates in raw_neg_tree_data.items():
            prefix_len = int(prefix_len_str)
            
            prefix_negs_nodes = []
            prefix_negs_edge_types = []
            prefix_neg_timestamps = []
            
            # For each candidate at this prefix length
            for candidate in candidates:
                if isinstance(candidate, list) and len(candidate) == 2:
                    node_id, timestamp = candidate
                    
                    # Reconstruct the path: take first prefix_len nodes from positive path
                    # and replace the last node with the candidate
                    if pos_nodes and len(pos_nodes) >= prefix_len:
                        neg_path = pos_nodes[:prefix_len] + [node_id]
                        # Edge types: take first prefix_len edge types from positive path
                        neg_edge_types = pos_edge_types[:prefix_len] if pos_edge_types and len(pos_edge_types) >= prefix_len else []
                        # Timestamps: take first prefix_len-1 edge timestamps from positive path, then add the new timestamp
                        neg_path_timestamps = (pos_edge_timestamps[:prefix_len-1] + [timestamp]) if pos_edge_timestamps and len(pos_edge_timestamps) >= prefix_len-1 else [timestamp]
                        
                        prefix_negs_nodes.append(neg_path)
                        prefix_negs_edge_types.append(neg_edge_types)
                        prefix_neg_timestamps.append(neg_path_timestamps)
            
            if prefix_negs_nodes:  # Only store if we have valid negative paths for this prefix length
                negs_by_prefix_length[prefix_len] = prefix_negs_nodes
                neg_edge_types_by_prefix_length[prefix_len] = prefix_negs_edge_types
                neg_timestamps_by_prefix_length[prefix_len] = prefix_neg_timestamps
        
        # Store the prefix-length grouped data in the item
        item['negs_by_prefix_length'] = negs_by_prefix_length
        item['neg_edge_types_by_prefix_length'] = neg_edge_types_by_prefix_length
        item['neg_timestamps_by_prefix_length'] = neg_timestamps_by_prefix_length
        
        # Store positive path info separately for easier access
        item['pos_path'] = pos_nodes
        item['pos_edge_types'] = pos_edge_types
        item['pos_timestamps'] = pos_edge_timestamps
        
        # --- GROUP ALL KGE PROXY ACCESS TOGETHER HERE ---
        if self.kge_proxy is not None:
            # Get device of KGE proxy model for intermediate operations
            device = next(self.kge_proxy.model.parameters()).device
            emb_dim = self.kge_proxy.model.node_emb.weight.size(1)
            
            # 1. First extract edge type embedding if available
            if edge_type is not None and hasattr(self.kge_proxy.model, 'rel_emb') and self.kge_proxy.model.rel_emb is not None:
                with torch.no_grad():
                    edge_type_tensor = torch.tensor([edge_type], dtype=torch.long, device=device)
                    type_embedding = self.kge_proxy.model.rel_emb(edge_type_tensor)
                    # Move to CPU and remove batch dimension
                    item['type_embedding'] = type_embedding.cpu()[0]
            
            # 2. Process positive path embeddings
            pos_node_embs = None
            pos_edge_embs = None
            
            if pos_nodes:
                # Create node_ids_tensor on the same device as the KGE model
                pos_node_tensor = torch.tensor(pos_nodes, dtype=torch.long, device=device)
                
                # Extract embeddings with no_grad
                with torch.no_grad():
                    # Get positive path node embeddings
                    pos_node_embs = self.kge_proxy.model.node_emb(pos_node_tensor).cpu()
                    
                    # Get positive path edge embeddings if available
                    if pos_edge_types and hasattr(self.kge_proxy.model, 'rel_emb') and self.kge_proxy.model.rel_emb is not None:
                        pos_edge_tensor = torch.tensor(pos_edge_types, dtype=torch.long, device=device)
                        pos_edge_embs = self.kge_proxy.model.rel_emb(pos_edge_tensor).cpu()
                    else:
                        pos_edge_embs = torch.empty(0, emb_dim)
            
            # Store positive path embeddings
            item['pos_node_embs'] = pos_node_embs
            item['pos_edge_embs'] = pos_edge_embs
            
            # 3. Process negative path embeddings per prefix length
            neg_node_embs_by_prefix = {}
            neg_edge_embs_by_prefix = {}
            
            for prefix_len, neg_paths in negs_by_prefix_length.items():
                prefix_node_embs = []
                prefix_edge_embs = []
                
                for i, neg_path in enumerate(neg_paths):
                    if not neg_path:
                        prefix_node_embs.append(torch.empty(0, emb_dim))
                        prefix_edge_embs.append(torch.empty(0, emb_dim))
                        continue
                    
                    # Create tensor on device
                    neg_node_tensor = torch.tensor(neg_path, dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        # Get negative path node embeddings
                        neg_node_emb = self.kge_proxy.model.node_emb(neg_node_tensor).cpu()
                        prefix_node_embs.append(neg_node_emb)
                        
                        # Get negative path edge embeddings if available
                        neg_edge_types = neg_edge_types_by_prefix_length.get(prefix_len, [])[i]
                        if neg_edge_types and hasattr(self.kge_proxy.model, 'rel_emb') and self.kge_proxy.model.rel_emb is not None:
                            neg_edge_tensor = torch.tensor(neg_edge_types, dtype=torch.long, device=device)
                            neg_edge_emb = self.kge_proxy.model.rel_emb(neg_edge_tensor).cpu()
                            prefix_edge_embs.append(neg_edge_emb)
                        else:
                            prefix_edge_embs.append(torch.empty(0, emb_dim))
                
                neg_node_embs_by_prefix[prefix_len] = prefix_node_embs
                neg_edge_embs_by_prefix[prefix_len] = prefix_edge_embs
            
            # Store negative path embeddings
            item['neg_node_embs_by_prefix'] = neg_node_embs_by_prefix
            item['neg_edge_embs_by_prefix'] = neg_edge_embs_by_prefix
            
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
        self.shuffle = shuffle
        cfg = json.load(open(config_path))
        self.cfg = cfg
        self.storage_dir = cfg.get('storage_dir', '.')
        self.dataset = cfg['dataset']

        # --- Adjust num_threads and batch_size if set to 'auto' or 'vast' ---
        num_threads = cfg.get('num_threads', mp.cpu_count())
        batch_size_cfg = batch_size

        if isinstance(num_threads, str):
            if num_threads.lower() == 'auto':
                num_threads = max(1, mp.cpu_count() - 2)
            elif num_threads.lower() == 'vast':
                cid = os.getenv("CONTAINER_ID")
                key = os.getenv("CONTAINER_API_KEY")
                assert cid and key, "Not running on a Vast.ai container!"
                resp = requests.get(
                    f"https://console.vast.ai/api/v0/instances/{cid}/",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "accept": "application/json"
                    },
                    timeout=10,
                )
                info = resp.json()
                num_threads = int(info['instances']["cpu_cores_effective"]) - 6
                print("Effective vCPUs (Vast.ai):", num_threads)

        if isinstance(batch_size_cfg, str):
            if batch_size_cfg.lower() == 'auto':
                batch_size = num_threads
            elif batch_size_cfg.lower() == 'vast':
                batch_size = num_threads
            else:
                batch_size = int(batch_size_cfg)
        else:
            batch_size = batch_size_cfg

        self.num_workers = num_threads
        self.batch_size = batch_size

        # Update pre_scan to accept a list of split names
        self.filter_splits = cfg.get('pre_scan', [])
        if isinstance(self.filter_splits, str):
            self.filter_splits = [self.filter_splits]
        self.embedding_used = cfg.get('embedding', None)
        self.test_time = False  # Flag to indicate if this is test time
        
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
        if self.use_shallow and self.kge_proxy["train"] is not None:
            emb_dim = self.cfg.get('hidden_channels') or self.kge_proxy["train"].cfg.get('hidden_channels', 0)
            dim += emb_dim
        return dim

    @property
    def train_data(self) -> pd.DataFrame:
        """train_data df[split] associated with the train phase

        Returns
        -------
        pd.DataFrame
            Returned dataframe of that split
        """
        if not self.test_time:
            if 'train' in self.data:
                return self.data['train']
            else:
                raise ValueError("Train data not set up. Call setup() first.")
        else:
            if 'valid' in self.data:
                return self.data['valid']
            else:
                raise ValueError("Validation data not set up. Call setup() first.")
            
    @property
    def valid_data(self) -> pd.DataFrame:
        """valid_data df[split] associated with the validation phase

        Returns
        -------
        pd.DataFrame
            Returned dataframe of that split
        """
        if not self.test_time:
            if 'valid' in self.data:
                return self.data['valid']
            else:
                raise ValueError("Validation data not set up. Call setup() first.")
        else:
            if 'test' in self.data:
                return self.data['test']
            else:
                raise ValueError("Test data not set up. Call setup() first.")
            
    @property
    def test_data(self) -> pd.DataFrame:
        """test_data df[split] associated with the test phase

        Returns
        -------
        pd.DataFrame
            Returned dataframe of that split
        """
        if 'test' in self.data:
            return self.data['test']
        else:
            raise ValueError("Test data not set up. Call setup() first.")
        
        
    def prepare_data(self):
        pass

    def setup(self, stage: Union[str, None] = None):
        if stage == "fit":
            print(f"Setting up data for stage: {stage}")
            edges_fp = os.path.join(self.storage_dir, f"{self.dataset}_edges.csv")
            if self.df is None:
                self.df = pd.read_csv(edges_fp, index_col='edge_id')            
                self.split_map = {str(idx): row['split'] for idx, row in self.df.iterrows()}

            split_code = {'pre': 0, 'train': 1, 'valid': 2, 'test': 3}
            for split in ['train', 'valid', 'test']:
                print(f"Setting up data for split: {split}")
                
                self.data[split] = self.df[self.df['split'] == split_code[split]].copy()
                # Initialize the length column with zeros
                self.data[split]['length'] = 0

                pos_paths = {}
                with open(os.path.join(self.storage_dir, f"{self.cfg['dataset']}_paths.txt")) as f:
                    n_str = f.readline()
                    n = int(n_str) if n_str and n_str.strip() else 0
                    for _ in range(n):
                        eid = f.readline().strip()
                        if not eid:
                            break
                        hops = int(f.readline())
                        nodes = [int(u) for u in f.readline().split()]
                        node_types = [int(t) for t in f.readline().split()]
                        edge_types_str = f.readline().strip().split()
                        edge_types = [int(et) for et in edge_types_str if et]
                        
                        edge_timestamps_str = f.readline().strip().split()
                        edge_timestamps = [int(ts) for ts in edge_timestamps_str if ts]

                        if self.split_map.get(eid) == split_code[split]:
                            pos_paths[eid] = {
                                "hops": hops,
                                "nodes": nodes,
                                "node_types": node_types,
                                "edge_types": edge_types,
                                "edge_timestamps": edge_timestamps
                            }
                            # Store length (hops + 1) in the dataframe
                            try:
                                int_eid = int(eid)
                                if int_eid in self.data[split].index:
                                    self.data[split].at[int_eid, 'length'] = hops + 1
                            except (ValueError, KeyError) as e:
                                print(f"Warning: Could not update length for eid {eid}: {e}")

                self.pos_paths[split] = pos_paths
                neg_fn = os.path.join(self.storage_dir, f"{self.cfg.get('model_name','transe')}_{self.dataset}_{split}_neg.json")
                
                try:
                    raw_neg_data = json.load(open(neg_fn))
                except FileNotFoundError:
                    print(f"Negative paths file not found for {split} split: {neg_fn}. Skipping negative paths setup.")
                    raw_neg_data = {}
                
                neg_paths = {}
                for eid, data in raw_neg_data.items():
                    # Assume tree-like format (no format checking)
                    neg_paths[eid] = data
                
                self.neg_paths[split] = neg_paths

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

                    config_prefix = f"{model_name}_{self.dataset}_{split if self.embedding_used is None else self.embedding_used}"
                    config_name = f"{config_prefix}{config_suffix}"
                    config_path = os.path.join(self.storage_dir, config_name)

                    state_suffix = "_embeddings.pt" if store == 'embedding' else "_model.pt"
                    state_dict_path = os.path.join(self.storage_dir, f"{config_prefix}{state_suffix}")

                    if os.path.exists(config_path):
                        print(f"Loading KGE model proxy for {split} split from {config_path}")
                        config = json.load(open(config_path))
                        self.kge_proxy[split] = KGEModelProxy(config, state_dict_path=state_dict_path)
                        self.kge_proxy[split].eval()
                        print(f"Device for KGE model: {next(self.kge_proxy[split].model.parameters()).device}")

                print(f"Loaded {len(self.data[split])} edges for {split} split.")
        else:
            print(f"Data already setup during fit stage, skipping setup for stage: {stage}")
                    

    def _dataloader(self, split: str, shuffle: bool):
        ds = EdgeDataset(
            self.data[split], self.pos_paths[split], self.neg_paths[split],
            self.features_map[split], self.kge_proxy[split], split=split
        )
        
        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,  # Only enable when using workers
            collate_fn=collate_by_prefix_length
        )

    def train_dataloader(self):
        if not self.test_time:
            return self._dataloader('train', self.shuffle)
        else:
            return self._dataloader('valid', self.shuffle)

    def val_dataloader(self):
        if not self.test_time:
            return self._dataloader('valid', False)
        else:
            return self._dataloader('test', False)

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
                neg_data = self.neg_paths[split][eid]
                # Tree-like format: check if any prefix has candidates
                has_valid_neg = any(candidates for candidates in neg_data.values())
                if not has_valid_neg:
                    result["empty_neg"] = 1
            else:
                result["missing_neg"] = 1
            
            # Check if edge is valid (has both positive and negative paths)
            if has_valid_pos:# and has_valid_neg:
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
            self.neg_paths[split] = {eid: self.neg_paths[split].get(eid, {}) for eid in valid_edge_ids}
            
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
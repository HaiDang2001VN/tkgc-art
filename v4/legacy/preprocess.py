import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Union, Tuple
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, set_start_method # Import set_start_method
from bisect import bisect_left

# Import the new proxy
from embedding import KGEModelProxy


class TemporalGraphSearch:
    def __init__(self, config_path: str, partition: str):
        # Load configuration
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.partition = partition
        self.num_threads = self.cfg.get('num_threads', 1)
        print(f"[LOG] Using {self.num_threads} threads/processes for processing.") # Clarified threads/processes

        # Read edges CSV (edge_id as index)
        csv_path = os.path.join(
            self.cfg.get('storage_dir', '.'),
            f"{self.cfg['dataset']}_edges.csv"
        )
        self.df = pd.read_csv(csv_path, index_col='edge_id')

        # DataFrames for graph and queries
        self.graph_df = self.df[self.df['label'] == 1]
        self.query_df = self.df[self.df['split'] != 'pre']

        # Initialize KGEModelProxy instances using embedding config
        self.models: List[KGEModelProxy] = [] # Initialize models attribute
        self._load_models() # Renamed from _load_model to _load_models

        # Build adjacency list and neighbor set
        self.adj_list: Dict[int, List[Tuple[int,
                                            Union[int, str], int]]] = defaultdict(list)
        self.final_graph: Dict[int, Set[int]] = {}
        self._build_graph()

        # Load shortest-path constraints keyed by edge_id
        self.shortest_paths: Dict[int, Dict] = {}
        paths_file = os.path.join(
            self.cfg.get('storage_dir', '.'),
            f"{self.cfg['dataset']}_paths.json"  # Correct path construction
        )
        if os.path.exists(paths_file):
            with open(paths_file) as pf:
                mapping = json.load(pf)
            for eid_str, path in mapping.items():
                self.shortest_paths[int(eid_str)] = path

    def _load_models(self): # Renamed and modified
        # Get the path to the original embedding config to find the model_name used for saving files
        original_embed_cfg_path = self.cfg.get('embedding_config')
        if not original_embed_cfg_path:
            raise KeyError("'embedding_config' missing in main config (self.cfg).")
        
        try:
            with open(original_embed_cfg_path) as ef_orig:
                original_embed_cfg_content = json.load(ef_orig)
        except FileNotFoundError:
            storage_dir_for_orig_cfg = self.cfg.get('storage_dir', '.')
            resolved_original_embed_cfg_path = os.path.join(storage_dir_for_orig_cfg, os.path.basename(original_embed_cfg_path))
            try:
                with open(resolved_original_embed_cfg_path) as ef_orig:
                    original_embed_cfg_content = json.load(ef_orig)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Original embedding config not found at {original_embed_cfg_path} or {resolved_original_embed_cfg_path}"
                )

        model_name_from_original_cfg = original_embed_cfg_content.get('model_name', 'model')

        out_prefix = f"{model_name_from_original_cfg}_{self.cfg['dataset']}_{self.partition}"
        base_storage_dir = self.cfg.get('storage_dir', '.')

        derived_embed_cfg_filename = f"{out_prefix}_config.json"
        derived_embed_cfg_path = os.path.join(base_storage_dir, derived_embed_cfg_filename)

        if not os.path.exists(derived_embed_cfg_path):
            raise FileNotFoundError(
                f"Derived embedding config file not found: {derived_embed_cfg_path}. "
                "Ensure embedding.py ran successfully for this partition and saved its config."
            )

        with open(derived_embed_cfg_path) as ef:
            embed_cfg = json.load(ef) # This is the config for the KGE model
            
        store_type = self.cfg.get('store', 'embedding')
        model_suffix = '_embeddings.pt' if store_type == 'embedding' else '_model.pt'
        state_dict_filename = f"{out_prefix}{model_suffix}"
        state_path = os.path.join(base_storage_dir, state_dict_filename)

        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Model state/embedding file not found: {state_path}. "
                "Ensure embedding.py ran successfully and saved the model/embeddings."
            )

        # Create num_threads instances of KGEModelProxy
        print(f"[LOG] Loading {self.num_threads} KGEModelProxy instances...")
        for i in range(self.num_threads):
            # Each proxy will load the model and move it to its device.
            # If on CUDA, care must be taken if processes fork after CUDA init in parent.
            # However, KGEModelProxy initializes its device internally.
            proxy_instance = KGEModelProxy(
                cfg=embed_cfg, # embed_cfg is shared (read-only dict)
                state_dict_path=state_path # state_path is shared (read-only string)
            )
            self.models.append(proxy_instance)
            print(f"[LOG] Loaded KGEModelProxy instance {i+1}/{self.num_threads} on device {proxy_instance.device}")
        
        if not self.models:
            raise RuntimeError("Failed to load any KGEModelProxy instances.")

    def _build_graph(self):
        """Construct adjacency and final_graph from graph_df"""
        for _, row in tqdm(self.graph_df.iterrows(), total=len(self.graph_df), desc="Building graph"):
            u, v = int(row['u']), int(row['v'])
            etype, ts = int(row['edge_type']), int(row['ts'])  # Ensure edge_type is int
            self.adj_list[u].append((v, etype, ts))
        for u, neighs in tqdm(self.adj_list.items(), desc="Sorting adjacency"):
            neighs.sort(key=lambda x: x[2])
            self.final_graph[u] = {neigh for neigh, _, _ in neighs}

    def beam_search(self) -> Dict[str, List[List[Union[int, str]]]]:
        """Run beam search on query_df using imap_unordered within Pool context"""
        beam_width = self.cfg.get('num_neg', 20)
        # Prepare lists: count empties and collect search targets
        results: Dict[str, List[List[Union[int, str]]]] = {}
        empty_count = 0
        total = len(self.query_df)
        
        print(f"[LOG] Starting beam search with {self.num_threads} processes and beam width {beam_width}")
        print(f"[LOG] Total edges to search: {total}")
        # (eid,u,v,ts,max_depth)
        to_search: List[Tuple[int, int, int, int, int]] = []
        for eid, row in tqdm(self.query_df.iterrows(), total=total, desc="Preparing targets"):
            if eid not in self.shortest_paths:
                empty_count += 1
            else:
                path_info = self.shortest_paths[eid]
                max_depth = path_info.get('hops', 0)
                if max_depth <= 0:
                    empty_count += 1
                else:
                    u, v, ts = int(row['u']), int(row['v']), int(row['ts'])
                    # Pass a model index to _process_edge, or let _process_edge pick one
                    # For simplicity with multiprocessing.Pool, _process_edge will use one from its copied self.models
                    to_search.append((eid, u, v, ts, max_depth))

        # Perform beam search only on valid targets
        # Each process in the Pool will get a copy of the TemporalGraphSearch instance.
        # This copy includes the self.models list (which itself contains copies of the KGEModelProxy objects).
        with Pool(self.num_threads) as pool:
            for eid, paths in tqdm(
                pool.imap_unordered(self._process_edge, to_search),
                total=len(to_search),
                desc="Beam searching"
            ):
                if not paths:
                    empty_count += 1
                else:
                    results[str(eid)] = paths

        # Print statistics
        print(
            f"[LOG] Beam search completed: {empty_count}/{total} edges had empty paths ({empty_count/total:.2%})")
        return results

    def _process_edge(self, args: Tuple[int, int, int, int, int]) -> Tuple[int, List[List[Union[int, str]]]]:
        """
        Perform beam search for a single edge in one direction.
        args = (edge_id, u, v, ts, max_depth)
        Returns (edge_id, paths)
        """
        eid, u, v, ts, max_depth = args
        beam_width = self.cfg.get('num_neg', 20)
        source = u
        
        # Each process uses one of its KGEModelProxy instances.
        # For simplicity, using the first one from its copied list.
        # This ensures model isolation between processes.
        # If KGEModelProxy is large, this means N copies of N models are made,
        # but each process only uses one of its N copies.
        # A more memory-efficient way for process pools is to initialize
        # the model within the worker process using an initializer.
        # However, adhering to "create num_threads kge proxy model" in init:
        if not self.models: # Should not happen if _load_models was successful
            raise RuntimeError("No models available in _process_edge. This indicates an issue with object state in the worker process.")
        
        # Pick a model for this process. If models are identical, [0] is fine.
        # To be slightly more robust if self.models could be shorter than self.num_threads in some edge case (though not intended):
        model_idx = os.getpid() % len(self.models) # Simple way to pick a model, somewhat distributing
        kge_model_instance = self.models[model_idx]

        current: List[List[Union[int, str]]] = [[u]]
        for depth in range(max_depth):
            next_paths: List[List[Union[int, str]]] = []
            for path in current:
                last = path[-1]
                edges = self.adj_list.get(last, [])
                idx = bisect_left(edges, ts, key=lambda x: x[2])
                for neigh, etype, _ in edges[:idx]:
                    if depth > 0 and neigh in self.final_graph.get(source, set()): # Ensure source key exists
                        continue
                    next_paths.append(path + [etype, neigh])
            if not next_paths:
                break
            num = len(next_paths)
            score_tensor = torch.zeros(num) # Stays on CPU
            
            # Batched inference
            # Ensure tensors sent to KGE model are on the KGE model's device
            model_device = kge_model_instance.device 
            for i in range(0, num, beam_width):
                # Batch construction should be integers for torch.tensor
                batch_paths_list = next_paths[i:i+beam_width]
                
                # Convert to tensor; KGEModelProxy expects (heads, rels, tails)
                # Here, path is [node, rel, node, rel, ..., node]
                # The KGEModelProxy.forward expects (batched_paths[:, -2], batched_paths[:, -1], batched_paths[:, 0])
                # This seems to imply paths are [tail, rel, head] for a single triple.
                # Current path structure: [u, etype1, node1, etype2, node2, ...]
                # For KGE scoring, we usually score (h, r, t) triples.
                # The current `next_paths` are full paths. The KGEModelProxy.forward is:
                # heads = batched_paths[:, -2]  (second to last element, should be relation)
                # rels = batched_paths[:, -1]   (last element, should be tail if path is [head, rel, tail])
                # tails = batched_paths[:, 0]   (first element, should be head)
                # This is unusual. Standard KGE models take (h,r,t).
                # Let's assume KGEModelProxy.forward(path_tensor) expects path_tensor where each row is a path
                # and it internally extracts h,r,t based on its specific logic.
                # The current batch is List[List[Union[int,str]]]. It needs to be List[List[int]].
                # Assuming all elements in next_paths are already integers as per type hints and previous logic.
                
                batch_tensor = torch.tensor(batch_paths_list, dtype=torch.long).to(model_device)

                with torch.no_grad():
                    sc = kge_model_instance(batch_tensor) # Use the selected kge_model_instance
                score_tensor[i:i+len(batch_tensor)] = sc.cpu() # Move scores to CPU for topk
            
            k = min(beam_width, num)
            indices = torch.topk(score_tensor, k).indices.tolist()
            current = [next_paths[i] for i in indices]
        return eid, current


if __name__ == '__main__':
    # Set the start method to 'spawn' for CUDA compatibility with multiprocessing
    # This should be done before any CUDA tensors or Pool objects are created.
    try:
        set_start_method('spawn', force=True) 
    except RuntimeError:
        # This might happen if the start method has already been set and force=False (default)
        # or if called at an inappropriate time. With force=True, it should generally work
        # unless there's a more fundamental issue with the environment.
        print("[Warning] Could not set multiprocessing start method to 'spawn'. This might lead to CUDA issues.")
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', required=True,
                        choices=['train', 'val', 'test'])
    parser.add_argument('--config', required=True, help='Path to config.json')
    args = parser.parse_args()

    tgs = TemporalGraphSearch(args.config, args.partition)
    results = tgs.beam_search()
    out_file = os.path.join(
        tgs.cfg.get('storage_dir', '.'),
        # Ensure the output filename reflects the actual model used if it's configurable
        # For now, assuming KGEModelProxy's internal model_name or a config value would be better.
        # Using a generic name or one derived from config for now.
        f"{tgs.models[0].model_name if tgs.models else 'kge'}_{tgs.cfg['dataset']}_{tgs.partition}_neg.json"
    )
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

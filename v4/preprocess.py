import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Union, Tuple
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from bisect import bisect_left

# Import the new proxy
from embedding import KGEModelProxy


class TemporalGraphSearch:
    def __init__(self, config_path: str, partition: str):
        # Load configuration
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.partition = partition

        # Read edges CSV (edge_id as index)
        csv_path = os.path.join(
            self.cfg.get('storage_dir', '.'),
            f"{self.cfg['dataset']}_edges.csv"
        )
        self.df = pd.read_csv(csv_path, index_col='edge_id')

        # DataFrames for graph and queries
        self.graph_df = self.df[self.df['label'] == 1]
        self.query_df = self.df[self.df['split'] != 'pre']

        # Initialize KGEModelProxy using embedding config
        self._load_model()

        # Build adjacency list and neighbor set
        self.adj_list: Dict[int, List[Tuple[int,
                                            Union[int, str], int]]] = defaultdict(list)
        self.final_graph: Dict[int, Set[int]] = {}
        self._build_graph()

        # Load shortest-path constraints keyed by edge_id
        self.shortest_paths: Dict[int, Dict] = {}
        paths_file = os.path.join(self.cfg.get(
            'storage_dir', '.'), 'paths.json')
        if os.path.exists(paths_file):
            with open(paths_file) as pf:
                mapping = json.load(pf)
            for eid_str, path in mapping.items():
                self.shortest_paths[int(eid_str)] = path

    def _load_model(self):
        # Load embedding configuration JSON
        embed_cfg_path = self.cfg.get('embedding_config')
        # if not os.path.isabs(embed_cfg_path):
        #     embed_cfg_path = os.path.join(
        #         self.cfg.get('storage_dir', '.'), embed_cfg_path)
        with open(embed_cfg_path) as ef:
            embed_cfg = json.load(ef)
            
        # TODO: save current config with runtime dataset parameters            
            
        # Override with runtime dataset parameters
        embed_cfg['num_nodes'] = self.cfg['num_nodes']
        embed_cfg['num_relations'] = self.cfg.get('num_rels', 1)
        embed_cfg['hidden_channels'] = self.cfg['hidden_dim']
        # Determine state dict path
        model_name = f"transe_{self.cfg['dataset']}_{self.partition}"
        base = self.cfg.get('storage_dir', '.')
        fname = f"{model_name}_model.pt" if self.cfg.get(
            'store') == 'model' else f"{model_name}_embeddings.pt"
        state_path = os.path.join(base, fname)
        # Instantiate proxy
        self.model = KGEModelProxy(
            cfg=embed_cfg,
            state_dict_path=state_path
        )

    def _build_graph(self):
        """Construct adjacency and final_graph from graph_df"""
        for _, row in tqdm(self.graph_df.iterrows(), total=len(self.graph_df), desc="Building graph"):
            u, v = int(row['u']), int(row['v'])
            etype, ts = row['edge_type'], int(row['ts'])
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
                    to_search.append((eid, u, v, ts, max_depth))

        # Perform beam search only on valid targets
        with Pool(cpu_count()) as pool:
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
        current: List[List[Union[int, str]]] = [[u]]
        for depth in range(max_depth):
            next_paths: List[List[Union[int, str]]] = []
            for path in current:
                last = path[-1]
                edges = self.adj_list.get(last, [])
                idx = bisect_left(edges, ts, key=lambda x: x[2])
                for neigh, etype, _ in edges[:idx]:
                    if depth > 0 and neigh in self.final_graph[source]:
                        continue
                    next_paths.append(path + [etype, neigh])
            if not next_paths:
                break
            num = len(next_paths)
            score_tensor = torch.zeros(num)
            for i in range(0, num, beam_width):
                batch = torch.tensor(
                    next_paths[i:i+beam_width], dtype=torch.long)
                with torch.no_grad():
                    sc = self.model(batch)
                score_tensor[i:i+len(batch)] = sc
            k = min(beam_width, num)
            indices = torch.topk(score_tensor, k).indices.tolist()
            current = [next_paths[i] for i in indices]
        return eid, current


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', required=True,
                        choices=['train', 'val', 'test'])
    parser.add_argument('--config', required=True, help='Path to config.json')
    args = parser.parse_args()

    tgs = TemporalGraphSearch(args.config, args.partition)
    results = tgs.beam_search()
    out_file = os.path.join(
        tgs.cfg.get('storage_dir', '.'),
        f"transe_{tgs.cfg['dataset']}_{tgs.partition}_neg.json"
    )
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

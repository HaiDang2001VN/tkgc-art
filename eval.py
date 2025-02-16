# eval.py

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class NegativeSampler:
    def __init__(self, full_edge_index, edge_times, k_hop):
        self.k_hop = k_hop
        self.full_edge_index = full_edge_index.clone()
        self.edge_times = edge_times.clone()
        
        # Initialize data structures
        self.node_first_seen = defaultdict(lambda: float('inf'))
        self.sorted_nodes = torch.tensor([], dtype=torch.long)
        self.sorted_timestamps = torch.tensor([], dtype=torch.float32)
        self.node_to_idx = {}
        self.global_counts = torch.tensor([], dtype=torch.long)
        self.adj = defaultdict(set)
        self.neighborhoods = {}
        self.pivoted_neighbors = {}
        self.negative_counts = {}

        # Initialization sequence
        self._init_first_appearances()
        self._sort_nodes()
        self._precompute_global_counts()
        self._build_adjacency()
        self._precompute_neighborhoods()
        self._calculate_negative_counts()

    def _init_first_appearances(self):
        for i in range(self.full_edge_index.size(1)):
            u = self.full_edge_index[0, i].item()
            v = self.full_edge_index[1, i].item()
            t = self.edge_times[i].item()
            self.node_first_seen[u] = min(self.node_first_seen[u], t)
            self.node_first_seen[v] = min(self.node_first_seen[v], t)

    def _sort_nodes(self):
        nodes = sorted(self.node_first_seen.keys(), 
                      key=lambda n: (self.node_first_seen[n], n))
        self.sorted_nodes = torch.tensor(nodes, dtype=torch.long)
        self.sorted_timestamps = torch.tensor(
            [self.node_first_seen[n] for n in nodes], 
            dtype=torch.float32
        )
        self.node_to_idx = {n: i for i, n in enumerate(nodes)}

    def _precompute_global_counts(self):
        n = len(self.sorted_nodes)
        self.global_counts = torch.zeros(n, dtype=torch.long)
        left = 0
        for right in range(n):
            current_time = self.sorted_timestamps[right]
            while self.sorted_timestamps[left] < current_time:
                left += 1
            self.global_counts[right] = left

    def _build_adjacency(self):
        for u, v in self.full_edge_index.t().tolist():
            self.adj[u].add(v)
            self.adj[v].add(u)

    def _precompute_neighborhoods(self):
        self.neighborhoods = {}
        for node in tqdm(self.sorted_nodes.tolist(), 
                       desc="Precomputing neighborhoods"):
            visited = set()
            queue = [(node, 0)]
            
            while queue:
                current, depth = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    if depth < self.k_hop:
                        queue.extend((n, depth+1) for n in self.adj[current])
            
            global_indices = sorted([self.node_to_idx[n] for n in visited])
            self.neighborhoods[node] = torch.tensor(global_indices, dtype=torch.long)

    def _calculate_negative_counts(self):
        for node in tqdm(self.sorted_nodes.tolist(),
                       desc="Calculating negative counts"):
            neighbors = self.neighborhoods[node]
            # Add pivot and convert to tensor
            self.pivoted_neighbors[node] = torch.cat([
                torch.tensor([-1], dtype=torch.long),
                neighbors
            ])
            # Calculate negative counts with +1 correction
            positions = torch.arange(len(self.pivoted_neighbors[node]))
            self.negative_counts[node] = self.pivoted_neighbors[node] - positions + 1

    def sampling(self, pos_edge_list, pos_edge_time, neg_ratio):
        if pos_edge_list.size(1) != pos_edge_time.size(0):
            raise ValueError("Edge list and timestamps size mismatch")
        if neg_ratio <= 0:
            raise ValueError("neg_ratio must be positive integer")
            
        neg_samples = []
        
        for edge_idx in range(pos_edge_list.size(1)):
            u, v = pos_edge_list[:, edge_idx].tolist()
            t = pos_edge_time[edge_idx].item()
            
            u_size = self._get_negative_set_size(u, t)
            v_size = self._get_negative_set_size(v, t)
            
            u_neg, v_neg = self._distribute_neg_samples(u_size, v_size, neg_ratio)
            
            if u_neg > 0:
                neg_u = self._sample_for_node(u, u_neg, u_size)
                neg_samples.extend([[u, n, t] for n in neg_u.tolist()])
            if v_neg > 0:
                neg_v = self._sample_for_node(v, v_neg, v_size)
                neg_samples.extend([[v, n, t] for n in neg_v.tolist()])
                
        return torch.tensor(neg_samples, dtype=torch.long).T if neg_samples else torch.empty(3, 0)

    def _get_negative_set_size(self, node, t):
        if node not in self.node_to_idx:
            return 0
        node_idx = self.node_to_idx[node]
        if self.sorted_timestamps[node_idx] >= t:
            # If node's first appearance older than t then this is invalid
            # Else if node's first appearance is exactly t then this is first appearance
            # So no negative samples because we dont's evaluate on first appearance
            return 0
            
        t_idx = torch.searchsorted(self.sorted_timestamps, t, side='right').item()
        neighbors = self.neighborhoods[node]
        neighbor_t_idx = torch.searchsorted(
            self.sorted_timestamps[neighbors], t, side='right'
        ).item()
        return max(0, t_idx - neighbor_t_idx)

    def _distribute_neg_samples(self, u_size, v_size, neg_ratio):
        total = u_size + v_size
        if total <= neg_ratio:
            return u_size, v_size
            
        u_quota = min(u_size, neg_ratio // 2)
        v_quota = neg_ratio - u_quota
        
        if u_quota > u_size:
            u_quota = u_size
            v_quota = neg_ratio - u_size
        elif v_quota > v_size:
            v_quota = v_size
            u_quota = neg_ratio - v_size
            
        return u_quota, v_quota

    def _sample_for_node(self, node, num_neg, neg_set_size):
        if num_neg == 0 or neg_set_size == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Generate unique seeds using NumPy for efficiency
        seeds = np.random.choice(neg_set_size, num_neg, replace=False)
        seeds = torch.from_numpy(seeds).long()
        
        # Find positions in precomputed negative counts
        counts = self.negative_counts[node]
        positions = torch.searchsorted(counts, seeds, side='right') - 1
        
        # Calculate global indices using formula
        neighbors = self.pivoted_neighbors[node]
        global_indices = seeds + 1 - counts[positions] + neighbors[positions]
        
        return global_indices

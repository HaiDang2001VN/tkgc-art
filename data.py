import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
from torch.utils.data import IterableDataset
from collections import defaultdict

def temporal_bfs_with_distance(central_edge, max_t, k_hops, edge_index, edge_time, current_inclusive):
    time_op = (lambda x: x <= max_t) if current_inclusive else (lambda x: x < max_t)
    
    visited_nodes = torch.zeros(edge_index.max().item() + 1, dtype=torch.bool)
    visited_edges = torch.zeros_like(edge_time, dtype=torch.bool)
    
    source, dest = central_edge[0].item(), central_edge[1].item()
    nodes = torch.tensor([source, dest], dtype=torch.long)
    distances = torch.tensor([0, 0], dtype=torch.long)
    visited_nodes[source] = True
    visited_nodes[dest] = True
    
    queue = torch.tensor([[source, 0], [dest, 0]], dtype=torch.long)

    while queue.shape[0] > 0:
        current_node, current_dist = queue[0].tolist()
        queue = queue[1:]
        
        if current_dist >= k_hops:
            continue
            
        mask = ((edge_index[0] == current_node) | 
                (edge_index[1] == current_node)) & time_op(edge_time)
        connected_edges = torch.where(mask)[0]

        for e_idx in connected_edges:
            if visited_edges[e_idx]:
                continue
            visited_edges[e_idx] = True

            u, v = edge_index[:, e_idx].tolist()
            neighbor = v if u == current_node else u

            if not visited_nodes[neighbor]:
                visited_nodes[neighbor] = True
                nodes = torch.cat([nodes, torch.tensor([neighbor])])
                distances = torch.cat([distances, torch.tensor([current_dist + 1])])
                queue = torch.cat([queue, torch.tensor([[neighbor, current_dist + 1]])])

    central_mask = (nodes == source) | (nodes == dest)
    if central_mask.sum().item() != 2:
        raise RuntimeError(f"Central mask invalid, expected 2 True values got {central_mask.sum().item()}")

    return (
        nodes,
        torch.where(visited_edges)[0],
        distances,
        central_mask
    )

class TemporalCoraDataset(IterableDataset):
    def __init__(self, root, config, split=None, parent_groups=None):
        self.config = config
        self.split = split
        self.batch_size = config['training']['batch_size']

        if parent_groups is None:
            self.cora = Planetoid(root=root, name=config['data']['name'])[0]
            self._add_temporal_info()
            self.node_mapping = self._create_global_mapping()
            self.num_nodes = len(self.node_mapping)
            self._convert_global_edges()
            self.temporal_groups = self._create_temporal_groups()
            self.split_groups = self._compute_splits()
        else:
            self.cora = parent_groups['cora']
            self.node_mapping = parent_groups['node_mapping']
            self.num_nodes = parent_groups['num_nodes']
            self.temporal_groups = parent_groups['temporal_groups']
            self.split_groups = parent_groups['split_groups']

    def _create_global_mapping(self):
        first_seen = {}
        for group in self.temporal_groups:
            edges = self.cora.edge_index[:, group['edges']]
            for node in torch.unique(edges):
                original_id = node.item()
                if original_id not in first_seen:
                    first_seen[original_id] = group['timestamp']
        
        sorted_nodes = sorted(first_seen.keys(), key=lambda x: (first_seen[x], x))
        return {original_id: idx for idx, original_id in enumerate(sorted_nodes)}

    def _convert_global_edges(self):
        original_edges = self.cora.edge_index.clone()
        self.cora.original_edge_index = original_edges
        self.cora.edge_index = torch.stack([
            torch.tensor([self.node_mapping[n.item()] for n in original_edges[0]]),
            torch.tensor([self.node_mapping[n.item()] for n in original_edges[1]])
        ]).long()

    def _add_temporal_info(self):
        num_edges = self.cora.edge_index.shape[1]
        self.cora.edge_time = torch.arange(num_edges, 0, -1, dtype=torch.float32)
        sorted_idx = torch.argsort(self.cora.edge_time, descending=True)
        self.cora.edge_index = self.cora.edge_index[:, sorted_idx]
        self.cora.edge_time = self.cora.edge_time[sorted_idx]

    def _create_temporal_groups(self):
        groups = defaultdict(list)
        for idx in range(len(self.cora.edge_time)):
            t = self.cora.edge_time[idx].item()
            groups[t].append(idx)
        
        sorted_groups = sorted(groups.items(), key=lambda x: -x[0])
        cumulative_batches = 0
        enhanced = []
        
        for ts, edges in sorted_groups:
            group_size = len(edges)
            num_batches = (group_size + self.batch_size - 1) // self.batch_size
            enhanced.append({
                'timestamp': ts,
                'edges': edges,
                'num_edges': group_size,
                'num_batches': num_batches,
                'start_batch': cumulative_batches,
                'end_batch': cumulative_batches + num_batches
            })
            cumulative_batches += num_batches
        
        return enhanced

    def _compute_splits(self):
        val_frac = self.config['data']['split_frac']['val']
        test_frac = self.config['data']['split_frac']['test']
        total_edges = sum(g['num_edges'] for g in self.temporal_groups)
        
        splits = {'train': [], 'val': [], 'test': []}
        cumulative = 0
        train_limit = int((1 - val_frac - test_frac) * total_edges)

        for group in self.temporal_groups:
            if cumulative + group['num_edges'] > train_limit:
                break
            splits['train'].append(group)
            cumulative += group['num_edges']

        val_limit = int(val_frac * total_edges)
        remaining = [g for g in self.temporal_groups if g not in splits['train']]
        cumulative = 0
        
        for group in remaining:
            if cumulative + group['num_edges'] > val_limit:
                break
            splits['val'].append(group)
            cumulative += group['num_edges']

        splits['test'] = [g for g in self.temporal_groups 
                        if g not in splits['train'] and g not in splits['val']]
        
        return splits

    def clone_for_split(self, split):
        return TemporalCoraDataset(
            root=self.config['data']['path'],
            config=self.config,
            split=split,
            parent_groups={
                'cora': self.cora,
                'node_mapping': self.node_mapping,
                'num_nodes': self.num_nodes,
                'temporal_groups': self.temporal_groups,
                'split_groups': self.split_groups
            }
        )

    def __iter__(self):
        if not self.split:
            raise ValueError("Split must be specified for cloned datasets")
            
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        for group in self.split_groups[self.split]:
            group_batches = range(group['start_batch'], group['end_batch'])
            last_batch = group['end_batch'] - 1
            
            for batch_idx in group_batches:
                if batch_idx % num_workers != worker_id:
                    continue
                
                is_last_in_group = (batch_idx == last_batch)
                local_idx = batch_idx - group['start_batch']
                start = local_idx * self.batch_size
                end = min(start + self.batch_size, group['num_edges'])
                
                batch_data = self.get_synced_subgraphs(
                    group['edges'][start:end],
                    self.config['training']['k_hops']
                )
                batch_data['meta'] = {
                    'is_group_end': is_last_in_group,
                    'num_nodes': self.num_nodes
                }
                yield batch_data

    def get_synced_subgraphs(self, edge_idx, k_hops):
        central_edge = self.cora.edge_index[:, edge_idx]
        central_time = self.cora.edge_time[edge_idx].item()
        
        dgt_data = self._get_subgraph(central_edge, central_time, k_hops, True)
        pgt_data = self._get_subgraph(central_edge, central_time, k_hops, False)
        
        return {
            'dgt': dgt_data,
            'pgt': pgt_data,
            'edge_time': central_time,
            'original_edge': central_edge
        }

    def _get_subgraph(self, central_edge, max_t, k_hops, current_inclusive):
        nodes, edges, distances, central_mask = temporal_bfs_with_distance(
            central_edge=central_edge,
            max_t=max_t,
            k_hops=k_hops,
            edge_index=self.cora.edge_index,
            edge_time=self.cora.edge_time,
            current_inclusive=current_inclusive
        )
        
        edge_pairs = self.cora.edge_index[:, edges]
        source_idx = torch.searchsorted(nodes, edge_pairs[0])
        dest_idx = torch.searchsorted(nodes, edge_pairs[1])
        local_edges = torch.stack([source_idx, dest_idx])

        adj = to_dense_adj(local_edges, max_num_nodes=len(nodes))[0]
        
        return {
            'nodes': nodes,
            'adj': adj,
            'dist': torch.exp(-distances.float().pow(2)),
            'central_mask': central_mask
        }

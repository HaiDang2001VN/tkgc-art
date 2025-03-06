from collections import defaultdict
from tqdm import tqdm
import torch
from torch_geometric.utils import to_dense_adj
from torch.utils.data import IterableDataset
from ogb.linkproppred import PygLinkPropPredDataset


def temporal_bfs_with_distance(central_edge, max_t, k_hops, edge_index, edge_time, current_inclusive):
    time_op = (lambda x: x <= max_t) if current_inclusive else (
        lambda x: x < max_t)

    visited_nodes = torch.zeros(edge_index.max().item() + 1, dtype=torch.bool)
    visited_edges = torch.zeros_like(edge_time, dtype=torch.bool)

    source, dest = central_edge[0].item(), central_edge[1].item()
    nodes = torch.tensor([source, dest], dtype=torch.long)
    distances = torch.tensor([0, 0], dtype=torch.long)
    visited_nodes[source] = True
    visited_nodes[dest] = True

    queue = torch.tensor([[source, 0], [dest, 0]], dtype=torch.long)

    while queue.shape[0] > 0:
        # if not current_inclusive:
        #     print(f"Queue size: {queue.shape[0]}")
        #     print(f"First element: {queue[0]}")
        current_node, current_dist = queue[0].tolist()
        queue = queue[1:]

        if current_dist >= k_hops:
            continue

        # if not current_inclusive:
        #     print(f"Edge index shape: {edge_index.shape}")
        #     print(f"Edge time shape: {edge_time.shape}")
        mask = ((edge_index[0] == current_node) |
                (edge_index[1] == current_node)) & time_op(edge_time)
        connected_edges = torch.where(mask)[0]
        # if not current_inclusive:
        #     print(f"Connected edges: {connected_edges}")

        for e_idx in connected_edges:
            if visited_edges[e_idx]:
                continue
            visited_edges[e_idx] = True

            u, v = edge_index[:, e_idx].tolist()
            neighbor = v if u == current_node else u

            if not visited_nodes[neighbor]:
                visited_nodes[neighbor] = True
                nodes = torch.cat([nodes, torch.tensor([neighbor])])
                distances = torch.cat(
                    [distances, torch.tensor([current_dist + 1])])
                queue = torch.cat(
                    [queue, torch.tensor([[neighbor, current_dist + 1]])])

    central_mask = (nodes == source) | (nodes == dest)
    if central_mask.sum().item() != 2:
        raise RuntimeError(
            f"Central mask invalid, expected 2 True values got {central_mask.sum().item()}")

    return (
        nodes,
        torch.where(visited_edges)[0],
        distances,
        central_mask
    )


class TemporalDataset(IterableDataset):
    def __init__(self, root, config, split=None, parent_groups=None):
        self.config = config
        self.split = split
        self.batch_size = config['training']['batch_size']
        self.edge_time = config['data']['edge_time']

        if parent_groups is None:
            self.data = PygLinkPropPredDataset(
                name=config['data']['name'], root=root)
            self.graph = self._add_temporal_info()
            self.num_nodes = self.graph['num_nodes']
            self.split_groups = self._compute_split_groups()
        else:
            self.data = parent_groups['data']
            self.graph = parent_groups['graph']
            self.num_nodes = self.graph['num_nodes']
            self.split_groups = parent_groups['split_groups']

    def _process_split_with_neg_ratio(self, split_name, split_edges, neg_ratio):
        """Helper function to process validation or test split with negative sampling ratio."""
        # Get positive edges and their timestamps
        pos_edges = split_edges[split_name]['edge']
        pos_times = split_edges[split_name][self.edge_time]
        
        # Get unique timestamps from positive edges
        unique_times = torch.unique(pos_times)
        
        # Cap neg_ratio if it's larger than the number of unique timestamps
        actual_neg_ratio = min(neg_ratio, len(unique_times))
        
        # Get negative edges and repeat them according to actual_neg_ratio
        neg_edges = split_edges[split_name]['edge_neg'].repeat(actual_neg_ratio, 1)
        
        # Generate distinct random permutations for each negative edge
        # This ensures each negative edge gets distinct timestamps
        all_perms = torch.stack([
            torch.randperm(len(unique_times))[:actual_neg_ratio] 
            for _ in range(len(split_edges[split_name]['edge_neg']))
        ])
        
        # Flatten the permutation indices and use them to get timestamps
        neg_times = unique_times[all_perms.view(-1)]
        
        # Combine positive and negative edges, times, and labels
        all_edges = torch.cat([pos_edges, neg_edges], dim=0)
        all_times = torch.cat([pos_times, neg_times])
        all_labels = torch.cat([
            torch.ones_like(pos_times),
            torch.zeros_like(neg_times)
        ])
        
        # Sort by time
        sort_idx = torch.argsort(all_times)
        
        return all_edges, all_times, all_labels, sort_idx, len(neg_edges)

    def _add_temporal_info(self):
        # Get the splits' edges in form of [N, 2] tensor
        split_edges = self.data.get_edge_split()
        
        # Get negative sampling ratio from config
        neg_ratio = int(self.config.get('data', {}).get('neg_ratio', 1))

        num_edges = {
            "train": split_edges['train']['edge'].size(0),
            "valid": split_edges['valid']['edge'].size(0),
            "neg_valid": split_edges['valid']['edge_neg'].size(0) * neg_ratio,
            "test": split_edges['test']['edge'].size(0),
            "neg_test": split_edges['test']['edge_neg'].size(0) * neg_ratio,
            "total": self.data.edge_index.size(0)
        }

        train_idx = torch.argsort(split_edges['train'][self.edge_time])
        
        # Process validation set
        valid_edges, valid_time, valid_label, valid_idx, num_neg_valid = self._process_split_with_neg_ratio(
            'valid', split_edges, neg_ratio
        )
        
        # Process test set
        test_edges, test_time, test_label, test_idx, num_neg_test = self._process_split_with_neg_ratio(
            'test', split_edges, neg_ratio
        )

        pos_edges = torch.cat([
            split_edges['train']['edge'],
            split_edges['valid']['edge'],
            split_edges['test']['edge']
        ], dim=0)
        
        pos_time = torch.cat([
            split_edges['train'][self.edge_time],
            split_edges['valid'][self.edge_time],
            split_edges['test'][self.edge_time]
        ])

        return {
            "edges": {
                "train": {
                    "edge_index": split_edges['train']['edge'][train_idx].t(),
                    "edge_time": split_edges['train'][self.edge_time][train_idx],
                    "num_edges": num_edges['train']
                },
                "valid": {
                    "edge_index": valid_edges[valid_idx].t(),
                    "edge_time": valid_time[valid_idx],
                    "edge_label": valid_label[valid_idx],
                    "num_edges": num_edges['valid'] + num_neg_valid
                },
                "test": {
                    "edge_index": test_edges[test_idx].t(),
                    "edge_time": test_time[test_idx],
                    "edge_label": test_label[test_idx],
                    "num_edges": num_edges['test'] + num_neg_test
                },
                "positive": {
                    "edge_index": pos_edges.t(),
                    "edge_time": pos_time
                }
            },
            "num_nodes": self.data[0].num_nodes,
            "num_edges": num_edges['total']
        }

    def _compute_split_groups(self):
        edges = self.graph['edges']
        split_groups = {}

        print("Computing split groups...")
        for split, split_edges in edges.items():
            if split == 'positive':
                continue
            
            print(f"Processing {split} split...")
            groups = defaultdict(list)

            for idx, edge_time in enumerate(split_edges['edge_time']):
                groups[edge_time].append(idx)

            print(f"Found {len(groups)} unique timestamps")
            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
            cumulative_batches = 0
            split_group = []

            print(f"Processing {len(sorted_groups)} timestamp groups...")
            for ts, edge_indices in tqdm(sorted_groups):
                group_size = len(edge_indices)
                num_batches = (group_size + self.batch_size -
                               1) // self.batch_size

                temporal_group = {
                    'timestamp': ts,
                    'edge_index': split_edges['edge_index'][:, edge_indices],
                    'edge_time': split_edges['edge_time'][edge_indices],
                    'num_edges': group_size,
                    'num_batches': num_batches,
                    'start_batch': cumulative_batches,
                    'end_batch': cumulative_batches + num_batches
                }

                if split != 'train':
                    temporal_group['edge_label'] = split_edges['edge_label'][edge_indices]

                split_group.append(temporal_group)
                cumulative_batches += num_batches

            split_groups[split] = split_group

        return split_groups

    def clone_for_split(self, split):
        return TemporalDataset(
            root=self.config['data']['path'],
            config=self.config,
            split=split,
            parent_groups={
                'data': self.data,
                'graph': self.graph,
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
            
            # Create a random permutation for edges within this temporal group
            # Only shuffle for training to ensure deterministic validation/testing
            if self.split == 'train':
                edge_permutation = torch.randperm(group['num_edges'])
            else:
                # Use identity permutation for validation/test
                edge_permutation = torch.arange(group['num_edges'])

            for batch_idx in group_batches:
                if batch_idx % num_workers != worker_id:
                    continue

                is_last_in_group = (batch_idx == last_batch)
                local_idx = batch_idx - group['start_batch']
                start = local_idx * self.batch_size
                end = min(start + self.batch_size, group['num_edges'])
                
                # Get permuted edge indices for current batch
                batch_edge_indices = edge_permutation[start:end]
                
                batch_data = self.get_synced_subgraphs(
                    group['edge_index'][:, batch_edge_indices],
                    group['edge_time'][batch_edge_indices],
                    self.config['training']['k_hops']
                )
                
                # Add labels for validation and test splits
                if self.split != 'train':
                    batch_data['labels'] = group['edge_label'][batch_edge_indices]
                
                batch_data['meta'] = {
                    'is_group_end': is_last_in_group,
                    'num_nodes': self.graph['num_nodes']
                }
                yield batch_data

    def get_synced_subgraphs(self, central_edges, central_times, k_hops):
        """
        Generate synchronized subgraphs for a batch of edges.
        
        Args:
            central_edges: Tensor of shape [2, batch_size] containing source and target nodes
            central_times: Tensor of shape [batch_size] containing edge timestamps
            k_hops: Number of hops for neighborhood extraction
        """
        batch_size = central_edges.size(1)
        batch_data = {
            'dgt': [],
            'pgt': [],
            'edge_time': [],
            'original_edge': []
        }
        
        for i in range(batch_size):
            edge = central_edges[:, i:i+1]
            time = central_times[i].item()
            
            dgt_data = self._get_subgraph(edge, time, k_hops, True)
            pgt_data = self._get_subgraph(edge, time, k_hops, False)
            
            batch_data['dgt'].append(dgt_data)
            batch_data['pgt'].append(pgt_data)
            batch_data['edge_time'].append(time)
            batch_data['original_edge'].append(edge)
        
        # Collate batch data
        return {
            'dgt': batch_data['dgt'],
            'pgt': batch_data['pgt'],
            'edge_time': torch.tensor(batch_data['edge_time']),
            'original_edge': torch.cat(batch_data['original_edge'], dim=1)
        }

    def _get_subgraph(self, central_edge, max_t, k_hops, current_inclusive):
        # Get the positive edges for subgraph extraction
        positive_edges = self.graph["edges"]["positive"]
        
        nodes, edges, distances, central_mask = temporal_bfs_with_distance(
            central_edge=central_edge,
            max_t=max_t,
            k_hops=k_hops,
            edge_index=positive_edges['edge_index'],
            edge_time=positive_edges['edge_time'],
            current_inclusive=current_inclusive
        )

        edge_pairs = positive_edges['edge_index'][:, edges]
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

    def __len__(self):
        """Return the length of the dataset in batches"""
        if not self.split or self.split not in self.split_groups:
            return 0
            
        # Get the last group of this split which contains the total batches
        groups = self.split_groups[self.split]
        if not groups:
            return 0
            
        last_group = groups[-1]
        return last_group['end_batch']

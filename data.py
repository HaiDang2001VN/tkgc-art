from collections import defaultdict
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

        if parent_groups is None:
            self.data = PygLinkPropPredDataset(
                name=config['data']['name'], root=root)
            self.graph = self._add_temporal_info()
            self.split_groups = self._compute_split_groups()
        else:
            self.data = parent_groups['data']
            self.graph = parent_groups['graph']
            self.split_groups = parent_groups['split_groups']

    def _add_temporal_info(self):
        # Get the splits' edges in form of [N, 2] tensor
        split_edges = self.data.get_idx_split()

        num_edges = {
            "train": split_edges['train']['edge'].size(0),
            "valid": split_edges['valid']['edge'].size(0),
            "neg_valid": split_edges['valid']['edge_neg'].size(0),
            "test": split_edges['test']['edge'].size(0),
            "neg_test": split_edges['test']['edge_neg'].size(0),
            "total": self.data.edge_index.size(0)
        }

        train_idx = torch.argsort(split_edges['train']['year'])

        valid_edges = torch.cat([
            split_edges['valid']['edge'],
            split_edges['valid']['edge_neg']
        ], dim=0)
        val_neg_time = torch.randint(0, len(split_edges['valid']['year']),
                                     (len(split_edges['valid']['edge_neg']),))
        valid_time = torch.cat([
            split_edges['valid']['year'],
            split_edges['valid']['year'][val_neg_time]
        ])
        valid_label = torch.cat([
            torch.ones_like(split_edges['valid']['year']),
            torch.zeros_like(split_edges['valid']['year'][val_neg_time])
        ])
        valid_idx = torch.argsort(valid_time)

        test_edges = torch.cat([
            split_edges['test']['edge'],
            split_edges['test']['edge_neg']
        ], dim=0)
        test_neg_time = torch.randint(0, len(split_edges['test']['year']),
                                      (len(split_edges['test']['edge_neg']),))
        test_time = torch.cat([
            split_edges['test']['year'],
            split_edges['test']['year'][test_neg_time]
        ])
        test_label = torch.cat([
            torch.ones_like(split_edges['test']['year']),
            torch.zeros_like(split_edges['test']['year'][test_neg_time])
        ])
        test_idx = torch.argsort(test_time)

        pos_edges = torch.cat([
            split_edges['train']['edge'],
            split_edges['valid']['edge'],
            split_edges['test']['edge']
        ], dim=0)
        pos_time = torch.cat([
            split_edges['train']['edge_time'],
            split_edges['valid']['year'],
            split_edges['test']['year']
        ])

        return {
            "edges": {
                "train": {
                    "edge_index": split_edges['train']['edge'][train_idx].t(),
                    "edge_time": split_edges['train']['edge_time'][train_idx],
                    "num_edges": num_edges['train']
                },
                "valid": {
                    "edge_index": valid_edges[valid_idx].t(),
                    "edge_time": valid_time[valid_idx],
                    "edge_label": valid_label[valid_idx],
                    "num_edges": num_edges['valid']
                },
                "test": {
                    "edge_index": test_edges[test_idx].t(),
                    "edge_time": test_time[test_idx],
                    "edge_label": test_label[test_idx],
                    "num_edges": num_edges['test']
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

        for split, split_edges in edges.items():
            groups = defaultdict(list)

            for idx, edge_time in enumerate(split_edges['edge_time']):
                groups[edge_time].append(idx)

            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
            cumulative_batches = 0
            split_group = []

            for ts, edge_indices in sorted_groups:
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
        graph = self.graph["edges"]["positive"]
        central_edge = graph['edge_index'][:, edge_idx]
        central_time = graph['edge_time'][edge_idx].item()

        dgt_data = self._get_subgraph(central_edge, central_time, k_hops, True)
        pgt_data = self._get_subgraph(
            central_edge, central_time, k_hops, False)

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
            edge_index=self.graph['edge_index'],
            edge_time=self.graph['edge_time'],
            current_inclusive=current_inclusive
        )

        edge_pairs = self.graph['edge_index'][:, edges]
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

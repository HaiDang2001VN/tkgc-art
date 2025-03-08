from collections import defaultdict, deque
from tqdm import tqdm
import torch
from torch_geometric.utils import to_dense_adj
from torch.utils.data import IterableDataset
from ogb.linkproppred import PygLinkPropPredDataset


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
            
            # Build optimized graph representation
            print("Building optimized temporal graph representation...")
            self._build_temporal_adjacency()
        else:
            self.data = parent_groups['data']
            self.graph = parent_groups['graph']
            self.num_nodes = self.graph['num_nodes']
            self.split_groups = parent_groups['split_groups']
            self.adj_list = parent_groups.get('adj_list')  # Share precomputed adjacency lists
    
    def _build_temporal_adjacency(self):
        """
        Build time-sorted adjacency lists for efficient traversal
        
        Note: This structure is shared between workers and should not be modified
        after initialization.
        """
        positive_edges = self.graph["edges"]["positive"]
        edge_index = positive_edges['edge_index']
        edge_time = positive_edges['edge_time']
        
        # Create adjacency list for each node
        self.adj_list = [[] for _ in range(self.num_nodes)]
        
        # Process all edges
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            t = edge_time[i].item()
            
            # Store (neighbor, time, edge_idx) tuples
            self.adj_list[src].append((dst, t, i))
            self.adj_list[dst].append((src, t, i))  # For undirected graphs
        
        # Sort each list by timestamp for early termination
        for i in range(self.num_nodes):
            self.adj_list[i].sort(key=lambda x: x[1])  # Sort by timestamp
    
    def temporal_bfs(self, central_edge, max_t, k_hops, current_inclusive):
        """Optimized BFS with early termination on time constraint"""
        source, dest = central_edge[0].item(), central_edge[1].item()
        
        # Add validation for source and dest nodes
        if source >= self.num_nodes or dest >= self.num_nodes:
            print(f"WARNING: Edge {source}->{dest} contains node outside graph range (0-{self.num_nodes-1})")
            # Return minimal valid result
            nodes = torch.tensor([source, dest], dtype=torch.long)
            edges = torch.tensor([], dtype=torch.long)
            distances = torch.tensor([0, 0], dtype=torch.long)
            central_mask = torch.tensor([True, True])
            return nodes, edges, distances, central_mask
        
        # Define time comparison operator
        time_op = (lambda x: x <= max_t) if current_inclusive else (lambda x: x < max_t)
        
        # Use efficient data structures for BFS
        visited_nodes = set([source, dest])
        visited_edges = set()
        node_distances = {source: 0, dest: 0}
        queue = deque([(source, 0), (dest, 0)])
        
        while queue:
            current_node, current_dist = queue.popleft()
            
            if current_dist > k_hops:
                continue
                
            # Process edges in time-sorted order with early termination
            for neighbor, t, edge_idx in self.adj_list[current_node]:
                # Early termination: break as soon as we exceed the max timestamp
                if not time_op(t):
                    break  # Since edges are sorted by time, all remaining edges will also fail
                
                if edge_idx in visited_edges:
                    continue
                    
                visited_edges.add(edge_idx)
                
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    node_distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))
        
        # More efficient tensor creation with pre-allocated lists
        nodes_list = list(visited_nodes)
        edges_list = list(visited_edges)
        
        # Convert to tensors efficiently
        nodes = torch.tensor(nodes_list, dtype=torch.long)
        edges = torch.tensor(edges_list, dtype=torch.long)
        
        # More efficient distance calculation
        distances = torch.zeros(len(nodes), dtype=torch.long)
        for i, n in enumerate(nodes):
            distances[i] = node_distances[n.item()]
        
        central_mask = (nodes == source) | (nodes == dest)
        
        return nodes, edges, distances, central_mask

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
                # Ensure we have a plain Python int for dictionary keys
                time_key = int(edge_time.item())  # Convert to Python int explicitly
                groups[time_key].append(idx)

            print(f"Found {len(groups)} unique timestamps")
            print("Top 5 unique timestamps: ", list(groups.keys())[:5])
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
            print("=" * 20 + "Split group information" + "=" * 20)
            print(f"Split: {split}")
            print(f"Number of edges: {cumulative_batches * self.batch_size}")
            print(f"Number of unique timestamps: {len(groups)}")
            print("Cummulative batches: ", cumulative_batches)
            print("=" * 50)

        return split_groups

    def clone_for_split(self, split):
        # Check if adj_list exists before sharing it
        if not hasattr(self, 'adj_list') or self.adj_list is None:
            print("WARNING: Adjacency list not built yet, building now...")
            self._build_temporal_adjacency()
            
        return TemporalDataset(
            root=self.config['data']['path'],
            config=self.config,
            split=split,
            parent_groups={
                'data': self.data,
                'graph': self.graph,
                'split_groups': self.split_groups,
                'adj_list': self.adj_list  # Now safely shared
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
            # print(f"Processing edge {i+1}/{batch_size}: {central_edges[0, i].item()} -> {central_edges[1, i].item()}")
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
        """Extract subgraph using the optimized BFS"""
        # Directly use the member method instead of the external function
        nodes, edges, distances, central_mask = self.temporal_bfs(
            central_edge=central_edge,
            max_t=max_t,
            k_hops=k_hops,
            current_inclusive=current_inclusive
        )

        # Sort nodes for searchsorted operations
        sort_indices = torch.argsort(nodes)
        nodes = nodes[sort_indices]
        distances = distances[sort_indices]
        central_mask = central_mask[sort_indices]

        # Get original edge pairs
        positive_edges = self.graph["edges"]["positive"]
        edge_pairs = positive_edges['edge_index'][:, edges]
        
        # Map to local indices
        source_idx = torch.searchsorted(nodes, edge_pairs[0])
        dest_idx = torch.searchsorted(nodes, edge_pairs[1])
        local_edges = torch.stack([source_idx, dest_idx])

        # Create adjacency matrix
        adj = to_dense_adj(local_edges, max_num_nodes=len(nodes))[0]

        # Debug for empty adjacency matrices (keeping your original debugging logic)
        if adj.sum() == 0 and len(nodes) > 2:
            print("=" * 50)
            print("WARNING: Empty adjacency matrix detected")
            print(f"Max timestamp: {max_t}")
            print(f"Number of nodes: {len(nodes)}")
            print(f"Number of edges before conversion: {len(edges)}")
            print(f"Edge pairs shape: {edge_pairs.shape}")
            print(f"Local edges shape: {local_edges.shape}")
            if local_edges.numel() > 0:
                print(f"Original edge pairs: {edge_pairs[:, :min(5, edge_pairs.shape[1])].tolist()}")
                print(f"First 5 local edges: {local_edges[:, :min(5, local_edges.shape[1])].tolist()}")
            print(f"Original central edge: {central_edge.tolist()}")
            print(f"Source idx: {source_idx.shape}, Dest idx: {dest_idx.shape}")
            if source_idx.numel() > 0:
                matching_indices = (source_idx < len(nodes)) & (dest_idx < len(nodes))
                print(f"Valid indices: {matching_indices.sum()}/{len(matching_indices)}")
            print("=" * 50)

        return {
            'nodes': nodes,
            'adj': adj,
            'dist': torch.exp(-distances.float().pow(2)),
            'central_mask': central_mask
        }

    def __len__(self):
        """Return the length of the dataset in batches, accounting for worker distribution"""
        if not self.split or self.split not in self.split_groups:
            return 0
            
        # Get the last group of this split which contains the total batches
        groups = self.split_groups[self.split]
        if not groups:
            return 0
            
        last_group = groups[-1]
        total_batches = last_group['end_batch']
        
        # Account for worker distribution
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # If we're in a worker process, only count batches this worker will process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            return (total_batches + num_workers - 1 - worker_id) // num_workers
        
        return total_batches

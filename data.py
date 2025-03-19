from collections import defaultdict, deque
from tqdm import tqdm
import torch
from torch_geometric.utils import to_dense_adj
from torch.utils.data import IterableDataset
from ogb.linkproppred import PygLinkPropPredDataset
import bisect


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
            self.adj_timestamps = parent_groups.get('adj_timestamps')  # Share precomputed timestamps
    
    def _build_temporal_adjacency(self):
        """
        Build time-sorted adjacency lists with deduplication and degree-based secondary sorting.
        Uses a single pass through time-ordered edges for efficiency.
        """
        positive_edges = self.graph["edges"]["positive"]
        edge_index = positive_edges['edge_index']
        edge_time = positive_edges['edge_time']
        
        # Get directionality setting
        is_bidirectional = self.config['data'].get('directionality', 'bi') == 'bi'
        degree_sort_order = self.config['training'].get('degree_sort', 'decreasing')
        
        # Create adjacency list for each node
        self.adj_list = [[] for _ in range(self.num_nodes)]
        
        # Sort edges by timestamp
        sorted_indices = torch.argsort(edge_time)
        
        # Group edges by timestamp for efficient processing
        edges_by_time = defaultdict(list)
        for idx in sorted_indices:
            t = edge_time[idx].item()
            edges_by_time[t].append(idx.item())
        
        # Track cumulative degrees for each node (running counter)
        node_degrees = defaultdict(int)
        
        # Process edges in time order
        for timestamp in sorted(edges_by_time.keys()):
            # Track unique edges at this timestamp to deduplicate
            seen_at_timestamp = set()
            # Track degree increases to apply at the end of this timestamp
            degree_updates = defaultdict(int)
            
            for edge_idx in edges_by_time[timestamp]:
                src, dst = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()
                
                # Deduplicate edges
                edge_key = (src, dst)
                if edge_key in seen_at_timestamp:
                    continue
                    
                # Mark this edge as seen
                seen_at_timestamp.add(edge_key)
                if is_bidirectional:
                    seen_at_timestamp.add((dst, src))
                    
                # Store edge with current cumulative degrees
                self.adj_list[src].append((dst, timestamp, edge_idx, node_degrees[dst]))
                
                # For undirected/bidirectional graphs
                if is_bidirectional:
                    self.adj_list[dst].append((src, timestamp, edge_idx, node_degrees[src]))
                
                # Track degree increases for this timestamp
                degree_updates[src] += 1
                if is_bidirectional:
                    degree_updates[dst] += 1
            
            # Update node degrees after processing all edges at this timestamp
            for node, degree_increase in degree_updates.items():
                node_degrees[node] += degree_increase
        
        # Sort adjacency lists by timestamp (primary) and degree (secondary)
        for i in range(self.num_nodes):
            if degree_sort_order == 'decreasing':
                # Sort by time ascending, then by degree descending
                self.adj_list[i].sort(key=lambda x: (x[1], -x[3]))
            else:
                # Sort by time ascending, then by degree ascending
                self.adj_list[i].sort(key=lambda x: (x[1], x[3]))
        
        # Create parallel arrays of just timestamps for bisect operations
        self.adj_timestamps = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            self.adj_timestamps[i] = [entry[1] for entry in self.adj_list[i]]

    def temporal_bfs(self, central_edge, max_t, k_hops, current_inclusive):
        """
        Two-phase BFS with fan-out control and local indexing:
        1. First collect all nodes (respecting fan-out)
        2. Build mapping from global to local indices based on sorted ordering
        3. Then collect all edges between these nodes with local indices (ignoring fan-out)
        
        Returns:
            nodes: Tensor of global node IDs (sorted)
            edge_list: Tensor of [src, dst] pairs using local indices
            distances: Distances from source/dest for each node
            central_mask: Boolean mask indicating central nodes
        """
        source, dest = central_edge[0].item(), central_edge[1].item()
        
        # Get fan_out parameter and directionality from config
        fan_out = self.config['training'].get('fan_out', float('inf'))
        is_bidirectional = self.config['data'].get('directionality', 'bi') == 'bi'
        
        # Add validation for source and dest nodes
        if source >= self.num_nodes or dest >= self.num_nodes:
            print(f"WARNING: Edge {source}->{dest} contains node outside graph range (0-{self.num_nodes-1})")
            # Return minimal valid result
            nodes = torch.tensor([source, dest], dtype=torch.long)
            edges = torch.tensor([], dtype=torch.long).reshape(0, 2)  # Empty edge list as [0, 2] tensor
            distances = torch.tensor([0, 0], dtype=torch.long)
            central_mask = torch.tensor([True, True])
            return nodes, edges, distances, central_mask
        
        # PHASE 1: Collect all relevant nodes with BFS (respecting fan-out)
        visited_nodes = set([source, dest])
        node_distances = {source: 0, dest: 0}
        queue = deque([(source, 0), (dest, 0)])
        hop_cnt = defaultdict(int)
        
        while queue:
            current_node, current_dist = queue.popleft()
            hop_cnt[current_dist] += 1
            
            if current_dist > k_hops:
                continue
            
            # Get node's adjacency list
            adj_nodes = self.adj_list[current_node]
            
            # Skip if no neighbors
            if not adj_nodes:
                continue
            
            # Use pre-computed timestamp list with bisect
            timestamps = self.adj_timestamps[current_node]
            
            # Get cutoff index using one bisect operation
            if current_inclusive:
                cutoff_idx = bisect.bisect_right(timestamps, max_t)
            else:
                cutoff_idx = bisect.bisect_left(timestamps, max_t)
            
            # print("Node: ", current_node, " - Cutoff index: ", cutoff_idx)
            # Count neighbors added for fan-out constraint
            neighbors_added = 0
            
            # Process edges in reverse order (most recent first, up to cutoff)
            for i in range(cutoff_idx - 1, -1, -1):
                neighbor, _, _, _ = adj_nodes[i]  # Unpack with 4-tuple format
                
                # Add neighbor to queue if not visited before
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    node_distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))
                    
                    # Increment counter and check fan-out limit
                    neighbors_added += 1
                    if neighbors_added >= fan_out:
                        break
                    
            # print("Visited nodes: ", len(visited_nodes), " - Queue size: ", len(queue))
        
        print("Hop count: ", dict(hop_cnt))
        # Create sorted node list and global-to-local mapping
        nodes_list = sorted(list(visited_nodes))
        global_to_local = {global_id: local_idx for local_idx, global_id in enumerate(nodes_list)}
        
        # PHASE 2: Collect all edges between the visited nodes using local indices
        visited_edges = set()
        edge_list = []  # Will store [src_local, dst_local] pairs
        
        for current_node in visited_nodes:
            # Get node's adjacency list
            adj_nodes = self.adj_list[current_node]
            
            # Skip if no neighbors
            if not adj_nodes:
                continue
            
            # Get current node's local index
            current_local = global_to_local[current_node]
            
            # Use pre-computed timestamp list with bisect
            timestamps = self.adj_timestamps[current_node]
            
            # Get cutoff index using one bisect operation
            if current_inclusive:
                cutoff_idx = bisect.bisect_right(timestamps, max_t)
            else:
                cutoff_idx = bisect.bisect_left(timestamps, max_t)
            
            # Process all edges up to cutoff that connect to visited nodes
            for i in range(cutoff_idx - 1, -1, -1):
                neighbor, _, edge_idx, _ = adj_nodes[i]  # Unpack with 4-tuple format
                
                # Only consider edges where both nodes are in our visited set
                if neighbor in visited_nodes:
                    # Skip if edge already visited
                    if edge_idx in visited_edges:
                        continue
                    
                    # Add the edge ID to visited set for deduplication
                    visited_edges.add(edge_idx)
                    
                    # Get neighbor's local index
                    neighbor_local = global_to_local[neighbor]
                    
                    # Add edge with local indices to edge list
                    edge_list.append([current_local, neighbor_local])
        
        # Convert to tensors efficiently
        nodes = torch.tensor(nodes_list, dtype=torch.long)
        
        # Convert edge list to tensor, or create empty tensor with correct shape
        if edge_list:
            edges = torch.tensor(edge_list, dtype=torch.long)
        else:
            edges = torch.zeros((0, 2), dtype=torch.long)
        
        # Vectorized distance calculation
        distances = torch.tensor([node_distances[n] for n in nodes_list], dtype=torch.long)
        
        # Use vectorized operations for central mask
        central_mask = torch.zeros(len(nodes), dtype=torch.bool)
        central_mask[(nodes == source) | (nodes == dest)] = True
        
        # Print nodes edges information
        print("Num nodes: ", len(nodes), " - Num edges: ", edges.size(0))
        
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
                'adj_list': self.adj_list,  # Now safely shared
                'adj_timestamps': self.adj_timestamps
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
        
        # No need to sort nodes - they're already sorted in temporal_bfs
        # No need for edge lookups - edges already have local indices
        
        # Just transpose to get shape [2, num_edges] as expected by to_dense_adj
        local_edges = edges.t()
        
        # Create adjacency matrix directly from local indices
        adj = to_dense_adj(local_edges, max_num_nodes=len(nodes))[0]
        
        # Debug for empty adjacency matrices (updated debugging logic)
        if adj.sum() == 0 and len(nodes) > 2:
            print("=" * 50)
            print("WARNING: Empty adjacency matrix detected")
            print(f"Max timestamp: {max_t}")
            print(f"Number of nodes: {len(nodes)}")
            print(f"Number of edges: {edges.shape[0]}")
            print(f"Local edges shape: {local_edges.shape}")
            if local_edges.numel() > 0:
                print(f"First 5 local edges: {local_edges[:, :min(5, local_edges.shape[1])].tolist()}")
            print(f"Original central edge: {central_edge.tolist()}")
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

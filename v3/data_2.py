from collections import defaultdict, deque
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, get_worker_info
from ogb.linkproppred import PygLinkPropPredDataset
import heapq
from embedding_2 import EmbeddingManager
import numpy as np
from typing import Literal

class TemporalDataset(IterableDataset):
    def __init__(self, config, node_dim: int = 128, k: int = None, m_d: int = None):
        super().__init__()
        
        self.dataset           = PygLinkPropPredDataset(name=config['data']['name'], root=config['data']['path'])
        self.is_bidirectional  = config['data'].get('directionality', 'bi') == 'bi'
        
        self.graph             = self._create_graph(config['data']['edge_time'], config['data']['neg_multiplier'])
        self.adjacent          = self._build_temporal_adjacency_lists(config['training'].get('degree_sort', 'descending'))
        self.temporal_groups   = self._create_temporal_groups()
		
        self.embedding_manager = EmbeddingManager(self.graph['num_nodes'], node_dim)
        self.k                 = k if k != None else config['training']['k_hops']
        self.m_d               = m_d if m_d != None else config['training']['max_degree']
        
        
    def _create_graph(self, edge_time: str, neg_multiplier: int):
        print('[GRAPH] Start building graph\n\n')
        
        edge_splits = self.dataset.get_edge_split()
        
        print('[GRAPH] Splits:\n' + '\n'.join(f"- {split[0]}: {split[1]['edge'].shape[0]} edges \
(bonus negative edges: {split[1]['edge_neg'].shape[0] if 'edge_neg' in split[1] else 0})" for split in edge_splits.items()) + '\n\n')
        
        def _process_split_with_neg_ratio(split):            
            pos_edges = split['edge']
            pos_timestamps = split[edge_time]
            
            unique_timestamps = torch.unique(pos_timestamps)
            
            actual_neg_multiplier = min(neg_multiplier, len(unique_timestamps))
            
            neg_edges = split['edge_neg'].repeat_interleave(actual_neg_multiplier, dim=1)
            
            timestamp_permutaion = torch.stack([
                torch.randperm(len(unique_timestamps))[:actual_neg_multiplier]
                for _ in range(len(split['edge_neg']))
            ])
            
            neg_timestamps = unique_timestamps[timestamp_permutaion.view(-1)]
            
            all_edges = torch.cat([pos_edges, neg_edges], dim=0)
            all_timestamps = torch.cat([pos_timestamps, neg_timestamps])
            all_labels = torch.cat([torch.ones_like(pos_timestamps), torch.zeros_like(neg_timestamps)])
            
            sort_indices = torch.argsort(all_timestamps)
            
            return all_edges[sort_indices], all_timestamps[sort_indices], all_labels[sort_indices]
        
        valid_edges, valid_timestamps, valid_labels = _process_split_with_neg_ratio(edge_splits['valid'])        
        test_edges, test_timestamps, test_labels = _process_split_with_neg_ratio(edge_splits['test'])
        
        pos_edges = torch.cat([edge_splits['train']['edge'], edge_splits['valid']['edge'], edge_splits['test']['edge']])
        pos_timestamps = torch.cat([edge_splits['train'][edge_time], edge_splits['valid'][edge_time], edge_splits['test'][edge_time]])
        pos_sort_indices = torch.argsort(pos_timestamps)
        pos_edges = pos_edges[pos_sort_indices]
        pos_timestamps = pos_timestamps[pos_sort_indices]
        
        train_sort_indices = torch.argsort(edge_splits['train'][edge_time])
        
        return {
            'splits': {
                'train': {
                    'edges': edge_splits['train']['edge'][train_sort_indices],        # shape = (M, 2)
                    'timestamps': edge_splits['train'][edge_time][train_sort_indices] # shape = (M,)
                },
                'valid': {
                    'edges': valid_edges,
                    'timestamps': valid_timestamps,
                    'labels': valid_labels
                },
                'test': {
                    'edges': test_edges,
                    'timestamps': test_timestamps,
                    'labels': test_labels
                },
                'positive': {
                    'edges': pos_edges,
                    'timestamps': pos_timestamps
                }
            },
            'num_nodes': self.dataset[0].num_nodes,
            'num_edges': self.dataset.edge_index.size()
        }
    
    
    def _build_temporal_adjacency_lists(self, degree_sort_order: Literal['descending', 'ascending']  = 'descending'):
        print('[ADJAC] Start building temporal adjacency list\n')
        
        pos_edges = self.graph['splits']['positive']
        
        adjacent = [[] for _ in range(self.graph['num_nodes'])]
        
        edges_by_timestamp = defaultdict(list)
        for idx, timestamp in enumerate(pos_edges['timestamps']):
            edges_by_timestamp[timestamp].append(idx)
            
        node_degrees = defaultdict(int)
        
        for timestamp in tqdm(sorted(edges_by_timestamp.keys())):
            existed_edges = set()
            
            degree_updates = defaultdict(int)
            
            for idx in edges_by_timestamp[timestamp]:
                src = pos_edges['edges'][idx, 0].item()
                dst = pos_edges['edges'][idx, 1].item()
                
                if (src, dst) in existed_edges:
                    continue
                
                existed_edges.add((src, dst))
                adjacent[src].append((dst, timestamp, idx, node_degrees[dst]))
                degree_updates[src] += 1
                
                if self.is_bidirectional:
                    existed_edges.add((dst, src))
                    adjacent[dst].append((src, timestamp, idx, node_degrees[src]))
                    degree_updates[dst] += 1
                    
            for node, degree_increase in degree_updates.items():
                node_degrees[node] += degree_increase
        
        print('\n')
        
        print('[ADJAC] Sort adjacent nodes by timestamp then by degree\n')
        
        for i in tqdm(range(self.graph['num_nodes'])):
            if degree_sort_order == 'descending':
                adjacent[i].sort(key=lambda v: (v[1], -v[3]))
            else:
                adjacent[i].sort(key=lambda v: (v[1], v[3]))
        
        print('\n')
        
        return adjacent
    
    
    def _create_temporal_groups(self):
        print('[TEMPG] Start creating temporal groups\n\n')
        
        temporal_groups = {}
        
        for split_name, split in self.graph['splits'].items():
            if split_name == 'positive':
                continue

            print(f'[TEMPG] Process split {split_name}\n\n')
            
            groups = defaultdict(list)
            
            for idx, timestamp in enumerate(split['timestamps']):
                key = int(timestamp.item())
                groups[key].append(idx)
                
            print(f'[TEMPG] Found {len(groups.keys())} timestamps\n')
                
            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
            temporal_group = []
            
            for timestamp, edge_indices in tqdm(sorted_groups):
                temporal_group.append({
                    'timestamp': timestamp,
                    'edges': split['edges'][edge_indices],
                    'labels': split['labels'][edge_indices] if split_name != 'train' else None
                })
                
            temporal_groups[split_name] = temporal_group
            
            print('\n')

        return temporal_groups
    
    
    def __iter__(self):
        if not self.split:
            raise ValueError('Split must be specified for cloned datasets')
        
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        current_worker_id = worker_info.id if worker_info else 0

        counter = 0
        for group in self.temporal_groups[self.split][1:]:
            timestamp = group['timestamp']
            
            for idx, edge in enumerate(group['edges']):
                if (counter + idx) % num_workers != current_worker_id:
                    continue
                
                paths  = torch.empty((0, self.k+1), dtype=torch.int32)
                masks  = torch.empty((0, self.k+1), dtype=torch.int8)
                labels = torch.empty((0, 1), dtype=torch.int8)
                tokens = torch.empty((0, self.k+1, self.embedding_manager.node_dim))
                
                path_list = self._temporal_bfs(edge[0], timestamp) + self._temporal_bfs(edge[1], timestamp)
                
                if len(path_list) == 0:
                    continue
                
                for path in path_list:                    
                    paths  = torch.cat((paths, path['path'].unsqueeze(dim=0)))
                    masks  = torch.cat((masks, path['mask'].unsqueeze(dim=0)))
                    labels = torch.cat((labels, path['label'].unsqueeze(dim=0)))
                    tokens = torch.cat((tokens, path['tokens'].unsqueeze(dim=0)))
                
                yield {
                    'paths': paths,
                    'masks': masks,
                    'labels': labels,
                    'tokens': tokens
                }
                
            counter = (counter + len(group['edges'])) % num_workers
      
            
    def _temporal_bfs(self, src, timestamp, current_inclusive: bool = False):
        n = self.graph['num_nodes']
        
        dist            = torch.zeros(n, dtype=torch.int32) - 1
        prev            = torch.zeros(n, dtype=torch.int32) - 1
        
        path_embeddings = torch.zeros((n, self.embedding_manager.node_dim))
        path_scores     = torch.zeros(n)
        
        dist[src] = 0
        path_embeddings[src] = self.embedding_manager.get_embedding(src)       
        
        queue = deque([src])
        
        time_check = lambda t: t < timestamp or (current_inclusive and t == timestamp)
        
        best_paths = []
        
        while queue:
            current_node = queue.popleft()
            
            if dist[current_node] == self.k:
                break
            
            for neighbour in self.adjacent[current_node]:
                if dist[neighbour[0]] == -1 and time_check(neighbour[1]):
                    queue.append(neighbour[0])
                    
                    dist[neighbour[0]] = dist[current_node] + 1
                    prev[neighbour[0]] = current_node
                    
                    path_embeddings[neighbour[0]] = self.embedding_manager.update_path_embedding(
                        path_embeddings[current_node], self.embedding_manager.get_embedding(neighbour[0])
                    )
                    path_scores[neighbour[0]] = self.embedding_manager.score(
                        path_embeddings[current_node], self.embedding_manager.get_embedding(neighbour[0])
                    )
                    
                    heapq.heappush(best_paths, (path_scores[neighbour[0]], neighbour[0]))
                    
        ret = []
        
        for (score, dst) in best_paths[:self.m_d]:            
            (path, mask) = self._trace_path(dst, prev)
            tokens = self.embedding_manager.get_path_tokens(path, mask)
            label = self._is_edge_formed_after_timestamp([path[0], path[-1]], timestamp)
            
            ret.append({
                'path': path,
                'mask': mask,
                'tokens': tokens,
                'label': torch.tensor([label], dtype=torch.int8)
            })
            
        return ret
    
    
    def _trace_path(self, dst, prev):
        path = [dst]
        
        while prev[dst].item() != -1:
            dst = prev[dst].item()
            path.append(dst)
        
        path = torch.tensor(path[::-1], dtype=torch.int32)
        mask = torch.ones(path.shape[0], dtype=torch.int8)
        
        while path.shape[0] < self.k + 1:
            path = torch.cat((path, torch.tensor([-1])))
            mask = torch.cat((mask, torch.tensor([0])))
            
        return (path, mask)
    
    
    def _is_edge_formed_after_timestamp(self, edge, timestamp, current_inclusive: bool = True) -> 0 | 1:
        for adj in self.adjacent[edge[0]]:
            if adj[0] == edge[1] and (adj[1] > timestamp or (adj[1] == timestamp and current_inclusive)):
                return 1
            
        return 0
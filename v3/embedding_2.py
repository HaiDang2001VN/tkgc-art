from collections import defaultdict
import math
from torch import nn
import torch

class EmbeddingManager(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int = 128):
        super().__init__()
        self.num_nodes    = num_nodes
        self.node_dim     = node_dim
        self.embeddings   = nn.Embedding(num_nodes, node_dim)
        self.padding      = torch.zeros(node_dim)
        self.update_cache = defaultdict(lambda: { 'sum': 0.0, 'count': 0 })
        self.reset()
        
    def reset(self):
        nn.init.normal_(
            self.embeddings.weight,
            mean=0.0,
            std=1/math.sqrt(self.node_dim)
        )
        
        self.update_cache.clear()
        
    def get_embedding(self, node):
        return self.embeddings(torch.tensor([node])).view(-1).detach()
    
    def update_embeddings(self, node, update):
        node = node.detach()
        self.update_cache[node.item()]['sum'] += update.detach()
        self.update_cache[node.item()]['count'] += 1
        
    def transition_timestamp(self):
        with torch.no_grad():
            for node, data in self.update_cache.items():
                if data['count'] > 0:
                    self.embeddings.weight.data[node] = data['sum']
            self.update_cache.clear()
            
    def score(self, embedding1, embedding2):         
        return torch.cosine_similarity(embedding1, embedding2, dim=0).item()
        
    def update_path_embedding(self, path_embedding, node_embedding):
        return (path_embedding + node_embedding) / 2
      
    def get_path_tokens(self, path, mask):
        embedding = torch.empty((0, self.node_dim))
        
        for (node, msk) in zip(path, mask):            
            if msk == 0:
                embedding = torch.cat((embedding, self.padding.unsqueeze(0)))
            else:
                embedding = torch.cat((embedding, self.get_embedding(node).unsqueeze(0)))
                
        return embedding
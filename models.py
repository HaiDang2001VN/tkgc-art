import math
import torch
import torch.nn as nn
from collections import defaultdict

class DGT(nn.Module):
    """Descriptive Graph Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
        # Read and store intermediate layers
        intermediate_layers = config['intermediate_layers']
        self.intermediate_layers = {
            int(layer): float(weight) for layer, weight in intermediate_layers.items() if weight > 0
        }
        
    def forward(self, x):
        intermediate = {}
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if i in self.intermediate_layers:
                intermediate[i] = x
        return intermediate

class PGT(nn.Module):
    """Predictive Graph Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_layers']
        )
        
    def forward(self, x):
        return self.transformer(x)

class TemporalEmbeddingManager(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super().__init__()
        self.num_nodes = num_nodes  # Track total nodes
        self.node_dim = node_dim    # Track embedding dimension
        self.embeddings = nn.Embedding(num_nodes, node_dim)
        self.update_cache = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        self.reset()  # Centralized initialization

    def reset(self):
        """Reinitialize embeddings with scaled normal distribution"""
        nn.init.normal_(self.embeddings.weight, 
                       mean=0.0, 
                       std=1/math.sqrt(self.node_dim))
        self.update_cache.clear()

    def get_embedding(self, node):
        return self.embeddings(node).detach()

    def update_embeddings(self, node, update):
        node = node.detach()
        self.update_cache[node.item()]['sum'] += update.detach()
        self.update_cache[node.item()]['count'] += 1

    def transition_timestamp(self):
        with torch.no_grad():
            for node, data in self.update_cache.items():
                if data['count'] > 0:
                    self.embeddings.weight.data[node] = data['sum'] / data['count']
            self.update_cache.clear()

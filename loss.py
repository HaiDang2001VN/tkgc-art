import torch
import torch.nn.functional as F
import math

def compute_dgt_loss(intermediate_outputs, adj_matrix, layer_weights):
    total_loss = 0.0
    total_weight = sum(layer_weights.values())
    
    for layer_idx, weight in layer_weights.items():
        embeddings = intermediate_outputs[layer_idx]
        attn = F.softmax(
            torch.matmul(embeddings, embeddings.transpose(-2, -1)) / 
            math.sqrt(embeddings.size(-1)), 
            dim=-1
        )
        masked = attn * adj_matrix
        layer_loss = -masked.sum() / adj_matrix.sum()
        total_loss += (weight / total_weight) * layer_loss
        
    return total_loss

def compute_pgt_loss(final_embeddings, central_masks, d_model):
    batch_likelihood = 0.0
    for emb, mask in zip(final_embeddings, central_masks):
        # Compute attention matrix
        attn_scores = torch.matmul(emb, emb.T) / math.sqrt(d_model)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Get central node indices
        central_indices = torch.where(mask)[0]
        if len(central_indices) != 2:
            raise ValueError(f"Expected exactly 2 central nodes, got {len(central_indices)}")
        
        i, j = central_indices
        # Sum both directions for undirected graph
        edge_likelihood = attn_weights[i, j] + attn_weights[j, i]
        batch_likelihood += edge_likelihood
    
    avg_likelihood = batch_likelihood / len(final_embeddings)
    return -avg_likelihood

def adaptive_update(old_emb, new_emb, distance):
    weights = distance.unsqueeze(-1)
    return (1 - weights) * old_emb + weights * new_emb.detach()
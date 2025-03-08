import torch
import torch.nn.functional as F
import math

def compute_dgt_loss(intermediate_outputs, adj_matrix, layer_weights):
    total_loss = 0.0
    total_weight = sum(layer_weights.values())
    
    for layer_idx, weight in layer_weights.items():
        embeddings = intermediate_outputs[layer_idx]
        
        temp = torch.matmul(embeddings, embeddings.transpose(-2, -1))
        
        attn = F.softmax(
            temp / math.sqrt(embeddings.size(-1)), 
            dim=-1
        )
        
        masked = attn * adj_matrix
        
        # Handle division by zero (no edges)
        if adj_matrix.sum() == 0:
            layer_loss = torch.tensor(0.0)
            # Error when more than 2 nodes
            if adj_matrix.size(0) > 2:
                print("embeddings", embeddings.shape)
                print("temp", temp.shape)
                print("attn", attn.shape)
                print("adj_matrix", adj_matrix.shape)
                print("masked", masked.shape)
                print("layer_loss", layer_loss)
                print("total_loss", total_loss)
                print("total_weight", total_weight)
                print("weight", weight)
                print("masked.sum()", masked.sum())
                print("adj_matrix.sum()", adj_matrix.sum())
                print("masked.sum() / adj_matrix.sum()",
                    masked.sum() / adj_matrix.sum())
                raise ValueError("Division by zero")
        else:
            # Modified to use log scaling
            normalized_attn = masked.sum() / adj_matrix.sum()
            layer_loss = -torch.log(normalized_attn + 1e-10)  # Add epsilon for numerical stability
            
        total_loss += (weight / total_weight) * layer_loss
        
    return total_loss

def compute_pgt_loss(final_embeddings, central_masks, d_model):
    batch_log_likelihood = 0.0  # Changed to track log likelihood
    edge_scores = []
    
    for emb, mask in zip(final_embeddings, central_masks):
        # Compute attention matrix
        attn_scores = torch.matmul(emb, emb.T) / math.sqrt(d_model)
        # print(attn_scores.shape)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # print(attn_weights.shape)
        
        # Get central node indices
        central_indices = torch.where(mask)[0]
        if len(central_indices) != 2:
            raise ValueError(f"Expected exactly 2 central nodes, got {len(central_indices)}")
        
        i, j = central_indices
        
        # Sum both directions for undirected graph
        edge_likelihood = attn_weights[i, j] + attn_weights[j, i]
        
        # Apply log transformation to edge likelihood
        log_edge_likelihood = torch.log(edge_likelihood + 1e-10)
        batch_log_likelihood += log_edge_likelihood
        
        # Calculate quantile-based edge score (unchanged)
        n = attn_weights.size(0)
        i_to_all = attn_weights[i]
        j_quantile_in_i = torch.sum(i_to_all <= i_to_all[j]).float() / n
        j_to_all = attn_weights[j]
        i_quantile_in_j = torch.sum(j_to_all <= j_to_all[i]).float() / n
        edge_quantile_score = (j_quantile_in_i + i_quantile_in_j) / 2.0
        edge_scores.append(edge_quantile_score)
    
    avg_log_likelihood = batch_log_likelihood / len(final_embeddings)
    return -avg_log_likelihood, edge_scores

def adaptive_update(old_emb, new_emb, distance):
    weights = distance.unsqueeze(-1)
    return (1 - weights) * old_emb + weights * new_emb.detach()
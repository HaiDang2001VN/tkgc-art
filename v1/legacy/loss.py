import torch
import torch.nn.functional as F
import math

def compute_dgt_loss(intermediate_outputs, adj_matrix, layer_weights):
    total_loss = 0.0
    total_weight = sum(layer_weights.values())
    
    for layer_idx, weight in layer_weights.items():
        embeddings = intermediate_outputs[layer_idx]
        
        # Compute similarity matrix (raw dot products)
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(-2, -1))
        
        # Handle empty graph
        if adj_matrix.sum() == 0:
            layer_loss = torch.tensor(0.0, device=embeddings.device)
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
                raise ValueError("Division by zero")
        else:
            num_nodes = adj_matrix.size(0)
            
            # Create masks for connected and non-connected nodes (excluding self-connections)
            self_mask = torch.eye(num_nodes, device=adj_matrix.device, dtype=torch.bool)
            connected_mask = adj_matrix.bool() & ~self_mask
            non_connected_mask = ~adj_matrix.bool() & ~self_mask
            
            # Count valid entries for each node
            connected_counts = connected_mask.sum(dim=1)
            non_connected_counts = non_connected_mask.sum(dim=1)
            
            # Create masks for valid comparisons (nodes with both connected and non-connected neighbors)
            valid_nodes = (connected_counts > 0) & (non_connected_counts > 0)
            
            if valid_nodes.sum() == 0:
                layer_loss = torch.tensor(0.0, device=embeddings.device)
            else:
                # Calculate means for connected nodes
                connected_sums = torch.sum(similarity_matrix * connected_mask.float(), dim=1)
                connected_means = torch.zeros_like(connected_sums)
                valid_conn = connected_counts > 0
                connected_means[valid_conn] = connected_sums[valid_conn] / connected_counts[valid_conn].float()
                
                # Calculate means for non-connected nodes
                non_connected_sums = torch.sum(similarity_matrix * non_connected_mask.float(), dim=1)
                non_connected_means = torch.zeros_like(non_connected_sums)
                valid_non_conn = non_connected_counts > 0
                non_connected_means[valid_non_conn] = non_connected_sums[valid_non_conn] / non_connected_counts[valid_non_conn].float()
                
                # Calculate variances
                connected_means_expanded = connected_means.unsqueeze(1).expand_as(similarity_matrix)
                non_connected_means_expanded = non_connected_means.unsqueeze(1).expand_as(similarity_matrix)
                
                # Squared deviations
                connected_sq_dev = ((similarity_matrix - connected_means_expanded) * connected_mask.float()) ** 2
                non_connected_sq_dev = ((similarity_matrix - non_connected_means_expanded) * non_connected_mask.float()) ** 2
                
                # Sum squared deviations
                connected_sum_sq_dev = torch.sum(connected_sq_dev, dim=1)
                non_connected_sum_sq_dev = torch.sum(non_connected_sq_dev, dim=1)
                
                # Calculate variances with proper degrees of freedom
                connected_var = torch.zeros_like(connected_sums)
                non_connected_var = torch.zeros_like(non_connected_sums)
                
                # Ensure we have enough samples for variance (n > 1)
                valid_conn_var = connected_counts > 1
                valid_non_conn_var = non_connected_counts > 1
                
                connected_var[valid_conn_var] = connected_sum_sq_dev[valid_conn_var] / (connected_counts[valid_conn_var] - 1).float()
                non_connected_var[valid_non_conn_var] = non_connected_sum_sq_dev[valid_non_conn_var] / (non_connected_counts[valid_non_conn_var] - 1).float()
                
                # Handle cases with only one sample by setting variance to a small constant
                connected_var[connected_counts == 1] = 1e-5
                non_connected_var[non_connected_counts == 1] = 1e-5
                
                # Calculate Welch's standard error (sqrt of sum of weighted variances)
                # SE = sqrt(s₁²/n₁ + s₂²/n₂)
                pooled_std_sq = (connected_var / connected_counts.float()) + (non_connected_var / non_connected_counts.float())
                pooled_std = torch.sqrt(pooled_std_sq + 1e-10)  # Adding small epsilon for numerical stability
                
                # Calculate mean difference (unmasked - masked)
                # Note: We're using (non_connected - connected) since we want to maximize this difference
                mean_diff = non_connected_means - connected_means
                
                # Weight mean differences by pooled std
                weighted_diffs = pooled_std * mean_diff
                
                # Calculate final loss as weighted average across valid nodes
                sum_pooled_std = pooled_std[valid_nodes].sum()
                if sum_pooled_std > 0:
                    layer_loss = weighted_diffs[valid_nodes].sum() / sum_pooled_std
                else:
                    layer_loss = torch.tensor(0.0, device=embeddings.device)
                
        total_loss += (weight / total_weight) * layer_loss
        
    return total_loss

def compute_pgt_loss(final_embeddings, central_masks, d_model):
    batch_z_score = 0.0
    edge_scores = []
    
    for emb, mask in zip(final_embeddings, central_masks):
        # Compute similarity matrix (raw dot products normalized by sqrt(d_model))
        similarity_matrix = torch.matmul(emb, emb.T) / math.sqrt(d_model)
        
        # Get central node indices
        central_indices = torch.where(mask)[0]
        if len(central_indices) != 2:
            raise ValueError(f"Expected exactly 2 central nodes, got {len(central_indices)}")
        
        i, j = central_indices
        
        # Get the dot product between central nodes
        central_edge_sim = similarity_matrix[i, j]
        
        # For first central node (i):
        i_sims = similarity_matrix[i]  # All similarities from node i
        
        # Create mask to exclude the central edge
        i_mask = torch.ones_like(i_sims, dtype=torch.bool)
        i_mask[j] = False  # Exclude connection to other central node
        
        # Calculate statistics excluding the central edge
        i_sims_no_central = i_sims[i_mask]
        i_mean = i_sims_no_central.mean()
        i_std = i_sims_no_central.std(unbiased=True)
        i_std = torch.clamp(i_std, min=1e-10)  # Prevent division by zero
        i_z_score = (central_edge_sim - i_mean) / i_std
        
        # For second central node (j):
        j_sims = similarity_matrix[j]  # All similarities from node j
        
        # Create mask to exclude the central edge
        j_mask = torch.ones_like(j_sims, dtype=torch.bool)
        j_mask[i] = False  # Exclude connection to other central node
        
        # Calculate statistics excluding the central edge
        j_sims_no_central = j_sims[j_mask]
        j_mean = j_sims_no_central.mean()
        j_std = j_sims_no_central.std(unbiased=True)
        j_std = torch.clamp(j_std, min=1e-10)  # Prevent division by zero
        j_z_score = (central_edge_sim - j_mean) / j_std
        
        # Average z-score for this edge
        edge_z_score = (i_z_score + j_z_score) / 2
        batch_z_score += edge_z_score
        
        # Calculate quantile-based edge score for compatibility
        n = similarity_matrix.size(0)
        j_quantile_in_i = torch.sum(i_sims <= central_edge_sim).float() / n
        i_quantile_in_j = torch.sum(j_sims <= central_edge_sim).float() / n
        edge_quantile_score = (j_quantile_in_i + i_quantile_in_j) / 2.0
        edge_scores.append(edge_quantile_score)
    
    avg_z_score = batch_z_score / len(final_embeddings)
    # Return negative z-score as loss (to maximize z-score)
    return -avg_z_score, edge_scores

def adaptive_update(old_emb, new_emb, distance):
    weights = distance.unsqueeze(-1)
    return (1 - weights) * old_emb + weights * new_emb.detach()
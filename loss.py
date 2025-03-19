import torch
import torch.nn.functional as F
import math

def adaptive_update(old_emb, new_emb, distance):
    """
    Apply adaptive update to embeddings
    
    Args:
        old_emb: Base embeddings [num_nodes, dim]
        new_emb: New embeddings [num_nodes, dim]
        distance: Distance weights [num_nodes]
        
    Returns:
        Updated embeddings [num_nodes, dim]
    """
    weights = distance.unsqueeze(-1)  # [num_nodes, 1]
    return (1 - weights) * old_emb + weights * new_emb.detach()

def adaptive_update_multi_layer(old_emb, intermediate_outputs, distance, layer_weights):
    """
    Apply adaptive update to multiple layers of embeddings and stack into tensor
    
    Args:
        old_emb: Base embeddings [num_nodes, dim]
        intermediate_outputs: Dictionary of intermediate layer outputs 
        distance: Distance weights for adaptive update [num_nodes]
        layer_weights: Dictionary of layer weights {layer_idx: weight}
        
    Returns:
        weighted_embs: Tensor of weighted embeddings [num_layers, num_nodes, dim]
        layer_weight_tensor: Tensor of normalized layer weights [num_layers]
        layer_indices: List of layer indices in same order as tensor dimensions
    """
    # Sort layer indices to ensure consistent ordering
    layer_indices = sorted(layer_weights.keys())
    
    # Get dimensions for pre-allocation
    device = old_emb.device
    num_layers = len(layer_indices)
    num_nodes = old_emb.size(0)
    emb_dim = old_emb.size(1)
    
    # Create stacked tensor of all weighted embeddings
    weighted_embs = torch.zeros(num_layers, num_nodes, emb_dim, device=device)
    
    # Fill tensor with adaptive updates for each layer
    for i, layer_idx in enumerate(layer_indices):
        layer_output = intermediate_outputs[layer_idx]
        if layer_output.dim() > 2:  # Handle batch dimension if present
            layer_output = layer_output.squeeze(0)  # [num_nodes, dim]
        weighted_embs[i] = adaptive_update(old_emb, layer_output, distance)
    
    # Convert layer weights to tensor in same order
    weights = [layer_weights[idx] for idx in layer_indices]
    layer_weight_tensor = torch.tensor(weights, device=device)  # [num_layers]
    layer_weight_tensor = layer_weight_tensor / layer_weight_tensor.sum()
    
    return weighted_embs, layer_weight_tensor, layer_indices

def compute_dgt_loss(weighted_embs, adj_matrix, layer_weight_tensor=None):
    """
    Compute DGT loss based on adaptively weighted embeddings
    
    Args:
        weighted_embs: Tensor of adaptively weighted embeddings 
                      Either [num_layers, num_nodes, dim] or [num_nodes, dim]
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        layer_weight_tensor: Layer weights tensor [num_layers] or None
    
    Returns:
        Total loss value [scalar]
    """
    # Handle single layer case by expanding dimension
    if weighted_embs.dim() == 2:
        # [num_nodes, dim] -> [1, num_nodes, dim]
        weighted_embs = weighted_embs.unsqueeze(0)
        if layer_weight_tensor is None:
            # Default to weight 1.0 for single layer
            layer_weight_tensor = torch.ones(1, device=weighted_embs.device)
    
    # Get dimensions
    num_layers, num_nodes, _ = weighted_embs.shape
    device = weighted_embs.device
    
    # Handle empty graph
    if adj_matrix.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # Create masks for connected and non-connected nodes (excluding self-connections)
    self_mask = torch.eye(num_nodes, device=device, dtype=torch.bool)  # [num_nodes, num_nodes]
    connected_mask = adj_matrix.bool() & ~self_mask  # [num_nodes, num_nodes]
    non_connected_mask = ~adj_matrix.bool() & ~self_mask  # [num_nodes, num_nodes]
    
    # Count valid entries for each node
    connected_counts = connected_mask.sum(dim=1)  # [num_nodes]
    non_connected_counts = non_connected_mask.sum(dim=1)  # [num_nodes]
    
    # Create masks for valid comparisons (nodes with both connected and non-connected neighbors)
    valid_nodes = (connected_counts > 0) & (non_connected_counts > 0)  # [num_nodes]
    
    if valid_nodes.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # Compute all similarity matrices at once
    similarity_matrices = torch.matmul(weighted_embs, weighted_embs.transpose(-2, -1))  # [num_layers, num_nodes, num_nodes]
    
    # Calculate means for connected nodes
    # First expand masks for broadcasting with layers
    connected_mask_expanded = connected_mask.unsqueeze(0).expand(num_layers, -1, -1)  # [num_layers, num_nodes, num_nodes]
    
    # Sum similarities for connected nodes
    connected_sums = torch.sum(similarity_matrices * connected_mask_expanded.float(), dim=2)  # [num_layers, num_nodes]
    
    # Create counts tensor expanded for layers
    conn_counts_expanded = connected_counts.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Compute means safely with division mask
    connected_means = torch.zeros_like(connected_sums)  # [num_layers, num_nodes]
    valid_conn = connected_counts > 0  # [num_nodes]
    valid_conn_expanded = valid_conn.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Only divide where counts > 0
    connected_means = torch.where(
        valid_conn_expanded,
        connected_sums / torch.clamp(conn_counts_expanded.float(), min=1.0),
        torch.zeros_like(connected_sums)
    )  # [num_layers, num_nodes]
    
    # Calculate means for non-connected nodes
    non_connected_mask_expanded = non_connected_mask.unsqueeze(0).expand(num_layers, -1, -1)  # [num_layers, num_nodes, num_nodes]
    
    # Sum similarities for non-connected nodes
    non_connected_sums = torch.sum(similarity_matrices * non_connected_mask_expanded.float(), dim=2)  # [num_layers, num_nodes]
    
    # Create counts tensor expanded for layers
    non_conn_counts_expanded = non_connected_counts.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Compute means safely with division mask
    non_connected_means = torch.zeros_like(non_connected_sums)  # [num_layers, num_nodes]
    valid_non_conn = non_connected_counts > 0  # [num_nodes]
    valid_non_conn_expanded = valid_non_conn.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Only divide where counts > 0
    non_connected_means = torch.where(
        valid_non_conn_expanded,
        non_connected_sums / torch.clamp(non_conn_counts_expanded.float(), min=1.0),
        torch.zeros_like(non_connected_sums)
    )  # [num_layers, num_nodes]
    
    # Expand means for variance calculation
    conn_means_expanded = connected_means.unsqueeze(-1).expand(-1, -1, num_nodes)  # [num_layers, num_nodes, num_nodes]
    non_conn_means_expanded = non_connected_means.unsqueeze(-1).expand(-1, -1, num_nodes)  # [num_layers, num_nodes, num_nodes]
    
    # Squared deviations
    conn_sq_dev = ((similarity_matrices - conn_means_expanded) * connected_mask_expanded.float()) ** 2  # [num_layers, num_nodes, num_nodes]
    non_conn_sq_dev = ((similarity_matrices - non_conn_means_expanded) * non_connected_mask_expanded.float()) ** 2  # [num_layers, num_nodes, num_nodes]
    
    # Sum squared deviations
    conn_sum_sq_dev = torch.sum(conn_sq_dev, dim=2)  # [num_layers, num_nodes]
    non_conn_sum_sq_dev = torch.sum(non_conn_sq_dev, dim=2)  # [num_layers, num_nodes]
    
    # Calculate variances with proper degrees of freedom
    valid_conn_var = connected_counts > 1  # [num_nodes]
    valid_conn_var_expanded = valid_conn_var.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    valid_non_conn_var = non_connected_counts > 1  # [num_nodes]
    valid_non_conn_var_expanded = valid_non_conn_var.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Calculate degrees of freedom
    conn_dof = torch.clamp(conn_counts_expanded - 1, min=1).float()  # [num_layers, num_nodes]
    non_conn_dof = torch.clamp(non_conn_counts_expanded - 1, min=1).float()  # [num_layers, num_nodes]
    
    # Calculate variances with stability handling
    conn_var = torch.where(
        valid_conn_var_expanded,
        conn_sum_sq_dev / conn_dof,
        torch.ones_like(conn_sum_sq_dev) * 1e-5
    )  # [num_layers, num_nodes]
    
    non_conn_var = torch.where(
        valid_non_conn_var_expanded,
        non_conn_sum_sq_dev / non_conn_dof,
        torch.ones_like(non_conn_sum_sq_dev) * 1e-5
    )  # [num_layers, num_nodes]
    
    # Calculate Welch's standard error (sqrt of sum of weighted variances)
    # SE = sqrt(s₁²/n₁ + s₂²/n₂)
    pooled_std_sq = (conn_var / conn_counts_expanded.float()) + (non_conn_var / non_conn_counts_expanded.float())  # [num_layers, num_nodes]
    pooled_std = torch.sqrt(pooled_std_sq + 1e-10)  # [num_layers, num_nodes]
    
    # Calculate mean difference (unmasked - masked)
    # We want to maximize the difference between non-connected and connected similarities
    mean_diff = non_connected_means - connected_means  # [num_layers, num_nodes]
    
    # Weight mean differences by pooled std
    weighted_diffs = pooled_std * mean_diff  # [num_layers, num_nodes]
    
    # Expand valid nodes mask to all layers
    valid_nodes_expanded = valid_nodes.unsqueeze(0).expand(num_layers, -1)  # [num_layers, num_nodes]
    
    # Calculate weighted loss contributions for each node in each layer
    weighted_node_losses = weighted_diffs * valid_nodes_expanded.float()  # [num_layers, num_nodes]
    
    # Calculate sum of pooled std across valid nodes for each layer
    pooled_std_sums = torch.sum(pooled_std * valid_nodes_expanded.float(), dim=1)  # [num_layers]
    
    # Calculate layer losses with safe division
    layer_losses = torch.zeros(num_layers, device=device)  # [num_layers]
    valid_layers = pooled_std_sums > 0  # [num_layers]
    
    # Only compute loss for layers with valid nodes
    layer_losses[valid_layers] = torch.sum(
        weighted_node_losses[valid_layers], dim=1
    ) / pooled_std_sums[valid_layers]  # [num_layers]
    
    # Weight layers by importance and sum
    if layer_weight_tensor is not None:
        # Ensure layer_weight_tensor sums to 1
        norm_layer_weights = layer_weight_tensor / layer_weight_tensor.sum()  # [num_layers]
        total_loss = torch.sum(layer_losses * norm_layer_weights)  # scalar
    else:
        total_loss = torch.mean(layer_losses)  # scalar
    
    # Compute average unnormalized mean difference
    avg_mean_diff = torch.mean(mean_diff)  # scalar
    
    return total_loss, avg_mean_diff

def compute_pgt_loss(final_embeddings, central_masks, d_model):
    """
    Compute PGT loss using z-scores of central edge similarities
    
    Args:
        final_embeddings: List of embeddings tensors, each [num_nodes, dim]
        central_masks: List of boolean masks indicating central nodes
        d_model: Model dimension for normalization
        
    Returns:
        loss: Negative average z-score (scalar)
        edge_scores: List of edge scores for compatibility
    """
    batch_z_score = 0.0
    edge_scores = []
    
    for emb, mask in zip(final_embeddings, central_masks):
        # Compute similarity matrix (raw dot products normalized by sqrt(d_model))
        similarity_matrix = torch.matmul(emb, emb.T) / math.sqrt(d_model)  # [num_nodes, num_nodes]
        
        # Get central node indices
        central_indices = torch.where(mask)[0]  # [2]
        if len(central_indices) != 2:
            raise ValueError(f"Expected exactly 2 central nodes, got {len(central_indices)}")
        
        i, j = central_indices  # i, j are scalars
        
        # Get the dot product between central nodes
        central_edge_sim = similarity_matrix[i, j]  # scalar
        
        # For first central node (i):
        i_sims = similarity_matrix[i]  # [num_nodes]
        
        # Create mask to exclude the central edge
        i_mask = torch.ones_like(i_sims, dtype=torch.bool)  # [num_nodes]
        i_mask[j] = False  # Exclude connection to other central node
        
        # Calculate statistics excluding the central edge
        i_sims_no_central = i_sims[i_mask]  # [num_nodes-1]
        i_mean = i_sims_no_central.mean()  # scalar
        i_std = i_sims_no_central.std(unbiased=True)  # scalar
        i_std = torch.clamp(i_std, min=1e-10)  # Prevent division by zero
        i_z_score = (central_edge_sim - i_mean) / i_std  # scalar
        
        # For second central node (j):
        j_sims = similarity_matrix[j]  # [num_nodes]
        
        # Create mask to exclude the central edge
        j_mask = torch.ones_like(j_sims, dtype=torch.bool)  # [num_nodes]
        j_mask[i] = False  # Exclude connection to other central node
        
        # Calculate statistics excluding the central edge
        j_sims_no_central = j_sims[j_mask]  # [num_nodes-1]
        j_mean = j_sims_no_central.mean()  # scalar
        j_std = j_sims_no_central.std(unbiased=True)  # scalar
        j_std = torch.clamp(j_std, min=1e-10)  # Prevent division by zero
        j_z_score = (central_edge_sim - j_mean) / j_std  # scalar
        
        # Average z-score for this edge
        edge_z_score = (i_z_score + j_z_score) / 2  # scalar
        batch_z_score += edge_z_score  # scalar
        
        # Calculate quantile-based edge score for compatibility
        n = similarity_matrix.size(0)  # scalar
        j_quantile_in_i = torch.sum(i_sims <= central_edge_sim).float() / n  # scalar
        i_quantile_in_j = torch.sum(j_sims <= central_edge_sim).float() / n  # scalar
        edge_quantile_score = (j_quantile_in_i + i_quantile_in_j) / 2.0  # scalar
        edge_scores.append(edge_quantile_score)
    
    avg_z_score = batch_z_score / len(final_embeddings)  # scalar
    # Return negative z-score as loss (to maximize z-score)
    return -avg_z_score, edge_scores, avg_z_score
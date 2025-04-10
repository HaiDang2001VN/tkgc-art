import torch
import torch.nn.functional as F


def l2_distance(predicted, target, mask):
    """
    Computes the L2 (Euclidean) distance between predicted and target embeddings.
    
    Args:
        predicted: Tensor of shape [batch, seq_len, embed_dim]
        target: Tensor of shape [batch, seq_len, embed_dim]
        mask: Tensor of shape [batch, seq_len] with 1 for valid tokens, 0 for padding
        
    Returns:
        Tensor of shape [batch] with average L2 distance per sequence, 
        considering only valid (non-padded) tokens.
    """
    # Compute L2 norm along embedding dimension per token
    token_l2 = torch.norm(predicted - target, p=2, dim=-1)  # shape: [batch, seq_len]
    
    # Apply mask to zero out padding tokens
    masked_l2 = token_l2 * mask  # shape: [batch, seq_len]
    
    # Sum distances and divide by the number of valid tokens per sequence
    # Add small epsilon to avoid division by zero for empty sequences
    seq_lengths = mask.sum(dim=1) + 1e-10  # shape: [batch]
    seq_l2 = masked_l2.sum(dim=1) / seq_lengths  # shape: [batch]
    
    return seq_l2


def cosine_distance(predicted, target, mask):
    """
    Computes a distance measure based on cosine similarity.
    Returns 1 - cosine_similarity so that lower values are better.
    
    Args:
        predicted: Tensor of shape [batch, seq_len, embed_dim]
        target: Tensor of shape [batch, seq_len, embed_dim]
        mask: Tensor of shape [batch, seq_len] with 1 for valid tokens, 0 for padding
        
    Returns:
        Tensor of shape [batch] with average cosine distance per sequence,
        considering only valid (non-padded) tokens.
    """
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(
        predicted, target, dim=-1)  # shape: [batch, seq_len]
    
    # Convert similarity to distance
    token_cos_dist = 1.0 - cos_sim  # shape: [batch, seq_len]
    
    # Apply mask to zero out padding tokens
    masked_cos_dist = token_cos_dist * mask  # shape: [batch, seq_len]
    
    # Sum distances and divide by the number of valid tokens per sequence
    # Add small epsilon to avoid division by zero for empty sequences
    seq_lengths = mask.sum(dim=1) + 1e-10  # shape: [batch]
    seq_cos_dist = masked_cos_dist.sum(dim=1) / seq_lengths  # shape: [batch]
    
    return seq_cos_dist


def compute_distance(predicted, target, mask, distance_metric="l2"):
    """
    Computes the distance between predicted and target embeddings.
    
    Args:
        predicted: Tensor of shape [batch, seq_len, embed_dim]
        target: Tensor of shape [batch, seq_len, embed_dim]
        mask: Tensor of shape [batch, seq_len] with 1 for valid tokens, 0 for padding
        distance_metric: String specifying the distance metric to use ('l2' or 'cosine')
        
    Returns:
        Tensor of shape [batch] with average distance per sequence
    """
    distance_metric = distance_metric.lower()
    
    if distance_metric == "l2":
        return l2_distance(predicted, target, mask)
    elif distance_metric == "cosine":
        return cosine_distance(predicted, target, mask)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # Return per-sample losses
    return dist

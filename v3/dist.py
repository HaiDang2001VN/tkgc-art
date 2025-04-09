import torch
import torch.nn.functional as F


def l2_distance(predicted, target):
    """
    Computes the L2 (Euclidean) distance between predicted and target embeddings.
    predicted, target: Tensors of shape [batch, seq_len, embed_dim]
    Returns:
        Tensor of shape [batch] with average L2 distance per sequence.
    """
    # Compute L2 norm along embedding dimension per token
    token_l2 = torch.norm(predicted - target, p=2, dim=-
                          1)  # shape: [batch, seq_len]
    # Average distance over sequence length
    seq_l2 = token_l2.mean(dim=1)
    return seq_l2


def cosine_distance(predicted, target):
    """
    Computes a distance measure based on cosine similarity.
    Returns 1 - cosine_similarity so that lower values are better.
    predicted, target: Tensors of shape [batch, seq_len, embed_dim]
    """
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(
        predicted, target, dim=-1)  # shape: [batch, seq_len]
    # Convert similarity to distance
    token_cos_dist = 1.0 - cos_sim
    seq_cos_dist = token_cos_dist.mean(dim=1)
    return seq_cos_dist


class NTPLoss(torch.nn.Module):
    """
    Custom Next-Token Prediction (NTP) Loss that uses a binary label to condition
    the loss. For positive examples (label=1), it minimizes the distance between
    predicted and target embeddings; for negative examples (label=0), it applies
    a hinge loss (with margin) to maximize the distance.
    """

    def __init__(self, margin=1.0, distance_metric="l2"):
        super(NTPLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric.lower()

    def forward(self, predicted, target, binary_labels):
        """
        predicted: Tensor of shape [batch, seq_len, embed_dim]
        target: Tensor of shape [batch, seq_len, embed_dim]
        binary_labels: Tensor of shape [batch] with values 0 or 1 (float type)
        """
        if self.distance_metric == "l2":
            dist = l2_distance(predicted, target)  # [batch]
        elif self.distance_metric == "cosine":
            dist = cosine_distance(predicted, target)  # [batch]
        else:
            raise ValueError("Unsupported distance metric")

        # For positive examples (label==1): minimize distance (loss = dist)
        # For negative examples (label==0): maximize distance via hinge loss: loss = relu(margin - dist)
        pos_loss = dist
        neg_loss = F.relu(self.margin - dist)

        # binary_labels assumed to be float tensor of shape [batch] (1 for positive, 0 for negative)
        loss = binary_labels * pos_loss + (1.0 - binary_labels) * neg_loss
        return loss.mean()

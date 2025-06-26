import json
import torch
import torch.nn.functional as F


def load_configuration(config_path):
    with open(config_path) as file:
        return json.load(file)


def norm(tensor: torch.Tensor, model=None, model_name=None, dim: int = -1) -> torch.Tensor:
    """
    Calculate the norm of a tensor based on the model type.
    
    Args:
        tensor: The tensor to calculate norm for
        model: The KGE model object (optional)
        model_name: Name of the model (optional, used if model doesn't have p_norm attribute)
        dim: Dimension along which to calculate norm (default: -1)
        
    Returns:
        Normalized tensor
    """
    if model and hasattr(model, 'p_norm'):
        return F.normalize(tensor, p=model.p_norm, dim=dim)
    elif model_name == 'rotate':
        return torch.linalg.vector_norm(tensor, dim=dim)
    else:
        return tensor.norm(p=2, dim=dim)

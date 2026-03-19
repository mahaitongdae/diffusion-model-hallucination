import torch
import logging
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class WeightsTransformFuncsRegistry:

    weights_transform_funcs = {}

    @classmethod
    def register(cls, func):
        cls.weights_transform_funcs[func.__name__] = func
        return func

    @classmethod
    def get(cls, name):
        return cls.weights_transform_funcs[name]

@WeightsTransformFuncsRegistry.register
def identity(weights, temperature=1.0, **kwargs):
    return weights / temperature

@WeightsTransformFuncsRegistry.register
def standarize(weights, temperature=1.0, **kwargs):
    return (weights - weights.mean()) / (weights.std() + 1e-8)

@WeightsTransformFuncsRegistry.register
def substract_min(weights, temperature=1.0, **kwargs):
    return weights - weights.min()

@WeightsTransformFuncsRegistry.register
def relu(weights, temperature=1.0, **kwargs):
    return torch.relu(weights / temperature)

@WeightsTransformFuncsRegistry.register
def softmax(weights, temperature=1.0, **kwargs):
    # assert weights.ndim == 1, "Softmax only supports 1D weights, but got shape: " + str(weights.shape)
    if weights.ndim != 1:
        logger.warning("Softmax only supports 1D weights, but got shape: " + str(weights.shape))
        weights = weights.view(-1)
    return torch.softmax(weights / temperature, dim=-1) * weights.shape[0]

@WeightsTransformFuncsRegistry.register
def softmax_negative(weights, temperature=1.0, min_value=-0.1, **kwargs):
    # assert weights.ndim == 1, "Softmax only supports 1D weights, but got shape: " + str(weights.shape)
    weights = weights.squeeze()
    if weights.ndim != 1:
        logger.warning("Softmax_negative only supports 1D weights, but got shape: " + str(weights.shape))
        weights = weights.view(-1)
    return torch.softmax(weights / temperature, dim=-1) * weights.shape[0] + min_value

@WeightsTransformFuncsRegistry.register
def quadratic(weights, temperature=1.0, **kwargs):
    return (weights / temperature) ** 2

@WeightsTransformFuncsRegistry.register
def quad_relu_normalize(weights, temperature=1.0, **kwargs):
    """
    Solves for v such that mean((ReLU(weights - v) / temperature)^2) = 1.
    Supports both 1D and 2D (batch) inputs.
    """
    v = solve_v_squared_batch(weights, temperature)
    weights_transformed = (torch.relu(weights - v) / temperature) ** 2
    return weights_transformed

@WeightsTransformFuncsRegistry.register
def relu_normalize(weights, temperature=1.0, **kwargs):
    """
    Solves for v such that mean(ReLU(weights - v) / temperature) = 1.
    Supports both 1D and 2D (batch) inputs.
    """
    v = solve_v_batch(weights, temperature)
    weights_transformed = torch.relu(weights - v) / temperature
    return weights_transformed

@WeightsTransformFuncsRegistry.register
def group_relative_relu_linear(weights, temperature=1.0, **kwargs):
    """
    Transforms weights to be relative to the group mean.
    """
    weights = weights.squeeze()
    assert weights.ndim == 1, "Group relative linear only supports 1D weights, but got shape: " + str(weights.shape)
    group_mean, group_std = weights.mean(), weights.std()
    weights_transformed = (weights - group_mean) / (group_std + 1e-6)
    weights_transformed = F.relu(weights_transformed)
    return weights_transformed

@WeightsTransformFuncsRegistry.register
def group_relative_relu_quadratic(weights, temperature=1.0, **kwargs):
    """
    Transforms weights to be relative to the group mean.
    """
    weights = weights.squeeze()
    assert weights.ndim == 1, "Group relative linear only supports 1D weights, but got shape: " + str(weights.shape)
    group_mean, group_std = weights.mean(), weights.std()
    weights_transformed = (weights - group_mean) / (group_std + 1e-6)
    weights_transformed = F.relu(weights_transformed) ** 2
    return weights_transformed

@WeightsTransformFuncsRegistry.register
def group_relative_negative_quadratic(weights, temperature=1.0, **kwargs):
    """
    Transforms weights to be relative to the group mean.
    """
    weights = weights.squeeze()
    assert weights.ndim == 1, "Group relative linear only supports 1D weights, but got shape: " + str(weights.shape)
    group_mean, group_std = weights.mean(), weights.std()
    weights_transformed = (weights - group_mean) / (group_std + 1e-6)
    weights_transformed = torch.where(weights_transformed > 0, weights_transformed ** 2, weights_transformed.clamp(min=-0.1))
    return weights_transformed

@WeightsTransformFuncsRegistry.register
def group_relative_negative_linear(weights, temperature=1.0, min_value=-0.1, **kwargs):
    """
    Transforms weights to be relative to the group mean.
    """
    weights = weights.squeeze()
    assert weights.ndim == 1, "Group relative linear only supports 1D weights, but got shape: " + str(weights.shape)
    group_mean, group_std = weights.mean(), weights.std()
    weights_transformed = (weights - group_mean) / (group_std + 1e-6)
    weights_transformed = torch.where(weights_transformed > 0, weights_transformed, weights_transformed.clamp(min=min_value))
    return weights_transformed


def solve_v_batch(x, l):
    """
    Solves for v such that mean(ReLU(x - v) / l) = 1 for each batch.
    
    Args:
        x: Input data of shape (batch_size, num_samples) or (num_samples,)
        l: Scale parameter of shape (batch_size, 1), (batch_size,), or scalar.
    
    Returns:
        v: Solution of shape (batch_size, 1) or scalar.
    """
    device = x.device
    if x.ndim == 1:
        x = x.unsqueeze(0)
        is_1d = True
    else:
        is_1d = False

    batch_size, N = x.shape
    # Target sum: sum(ReLU(...)) = N * l
    if isinstance(l, torch.Tensor):
        if l.ndim == 1:
            l = l.unsqueeze(-1)
    target = N * l
    
    # 1. Sort descending
    x_sorted, _ = torch.sort(x, dim=-1, descending=True)
    
    # 2. Cumulative sums
    cumsum_x = torch.cumsum(x_sorted, dim=-1)
    
    # 3. Indices k = 1..N
    k_indices = torch.arange(1, N + 1, device=device).view(1, -1)
    
    # 4. Determine active set size k*
    # term_k = sum(x_top_k) - k * x_k
    term_k = cumsum_x - k_indices * x_sorted
    
    mask = term_k < target
    k_star = torch.sum(mask.to(torch.int32), dim=-1, keepdim=True)
    k_star = torch.clamp(k_star, min=1)
    
    # 5. Gather sum of top k* elements
    sum_active = torch.take_along_dim(cumsum_x, k_star - 1, dim=-1)
    
    # 6. Solve for v: sum_active - k*v = target  => v = (sum_active - target) / k*
    v = (sum_active - target) / k_star
    
    if is_1d:
        return v.squeeze()
    return v

def solve_v_squared_batch(x, l):
    """
    Solves for v such that mean((ReLU(x - v) / l)^2) = 1.
    
    Args:
        x: Input data of shape (batch_size, num_samples) or (num_samples,)
        l: Scale parameter of shape (batch_size, 1), (batch_size,), or scalar.
    
    Returns:
        v: Solution of shape (batch_size, 1) or scalar.
    """
    device = x.device
    if x.ndim == 1:
        x = x.unsqueeze(0)
        is_1d = True
    else:
        is_1d = False

    batch_size, N = x.shape
    # Target sum of squares: sum(ReLU(...)^2) = N * l^2
    if isinstance(l, torch.Tensor):
        if l.ndim == 1:
            l = l.unsqueeze(-1)
    C = N * (l ** 2)
    
    # 1. Sort descending
    x_sorted, _ = torch.sort(x, dim=-1, descending=True)
    
    # 2. Cumulative sums for moments
    cumsum_x = torch.cumsum(x_sorted, dim=-1)
    cumsum_x2 = torch.cumsum(x_sorted ** 2, dim=-1)
    
    # 3. Indices k = 1..N
    k_indices = torch.arange(1, N + 1, device=device).view(1, -1)
    
    # 4. Calculate "Energy at Boundary" (v = x_k)
    # Expansion: sum(x^2) - 2*x_k*sum(x) + k*x_k^2
    energy_at_boundary = (
        cumsum_x2 
        - 2 * x_sorted * cumsum_x 
        + k_indices * (x_sorted ** 2)
    )
    
    # 5. Determine active set size k*
    mask = energy_at_boundary < C
    k_star = torch.sum(mask.to(torch.int32), dim=-1, keepdim=True)
    k_star = torch.clamp(k_star, min=1)
    
    # 6. Gather sufficient statistics for the valid k*
    S1_active = torch.take_along_dim(cumsum_x, k_star - 1, dim=-1)
    S2_active = torch.take_along_dim(cumsum_x2, k_star - 1, dim=-1)
    
    # 7. Solve Quadratic: k*v^2 - 2*S1*v + (S2 - C) = 0
    # v = (S1 - sqrt(S1^2 - k*(S2 - C))) / k
    term_inside_sqrt = S1_active**2 - k_star * (S2_active - C)
    term_inside_sqrt = torch.clamp(term_inside_sqrt, min=0.0)
    
    v = (S1_active - torch.sqrt(term_inside_sqrt)) / k_star
    
    if is_1d:
        return v.squeeze()
    return v

if __name__ == "__main__":
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights_transformed = quad_relu_normalize(weights, temperature=1.0)
    print(weights_transformed.mean())

from typing import Iterable
import torch


def GELU(x: torch.Tensor) -> torch.Tensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    Args:
        in_features: torch.FloatTensor
            Input features to run GELU on. Shape is arbitrary.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of applying
        GELU to each element.
    """
    return 0.5 * x * (1 + torch.erf(x / (torch.Tensor([2], device=x.device) ** 0.5)))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features: torch.FloatTensor
            Input features to softmax. Shape is arbitrary.
        dim: int
            Dimension of the `in_features` to apply softmax to.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / exp_x.sum(dim=dim, keepdim=True) 


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    # rescale for numerical stability by subtracting maximum
    # torch.logsumexp already does this so maybe not necessary
    logits -= logits.max(dim=-1, keepdim=True).values
    ip = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
    lse = logits.logsumexp(dim=-1)
    return (lse - ip).mean()


def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float,
        eps: float = 1e-6,
    ):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.
        eps: a small positive value to avoid division by zero when rescaling gradients.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    squared_grads = [torch.sum(p.grad ** 2) for p in parameters if p.grad is not None]
    l2_norm = torch.sqrt(torch.sum(torch.stack(squared_grads)))
    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / (l2_norm + eps)



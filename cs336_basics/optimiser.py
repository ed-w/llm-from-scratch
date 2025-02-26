import math
from typing import Callable, Optional, Tuple
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """This class implements the AdamW optimisation algorithm for use with PyTorch models.
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.01.
    """
    def __init__(self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr" : lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:

            # get hyperparameters
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # update step count
                state = self.state[p]
                t = state.get("t", 1)

                # get gradients and moment estimates
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(grad, device=grad.device))
                v = state.get("v", torch.zeros_like(grad, device=grad.device))

                # update moment estimates
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                # compute adjusted alpha and update parameters 
                alpha = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + \
            0.5 * (1 + math.cos(
                math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
            )) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
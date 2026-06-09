import torch
from collections.abc import Iterable

def gradient_clipping(
        params: Iterable[torch.Tensor],
        max_norm: float,
        eps: float = 1e-6
    ) -> float:
    total_norm = 0.0

    for p in params:
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)
            total_norm += grad_norm.item() ** 2

    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
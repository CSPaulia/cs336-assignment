import torch

def cross_entropy(
        pred: torch.Tensor, 
        target: torch.Tensor
    ):
    loss = (-pred[torch.arange(pred.shape[0]), target] + torch.logsumexp(pred, dim=-1)).mean()
    return loss
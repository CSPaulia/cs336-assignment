from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss
    

class AdamW(SGD):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._init_moment_vectors()

    def _init_moment_vectors(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:  # If the state is uninitialized.
                    state["m"] = torch.zeros_like(p.data)  # Initialize first moment vector.
                    state["v"] = torch.zeros_like(p.data)  # Initialize second moment vector.

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                lr_t = lr * math.sqrt(1 - self.betas[1] ** (t + 1)) / (1 - self.betas[0] ** (t + 1))
                p.data -= lr * self.weight_decay * p.data  # Apply weight decay.
    
                state["m"] = self.betas[0] * state["m"] + (1 - self.betas[0]) * grad  # Update biased first moment estimate.
                state["v"] = self.betas[1] * state["v"] + (1 - self.betas[1]) * grad * grad  # Update biased second moment estimate.

                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + self.eps)  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())

        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
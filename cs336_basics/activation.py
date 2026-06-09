import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Float

class Swish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x * self.beta)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.swish = Swish()

        self.w1_weight = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w2_weight = nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )
        self.w3_weight = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.__init_weight__()

    def __init_weight__(self):
        # nn.init.trunc_normal_(self.w1_weight)
        # nn.init.trunc_normal_(self.w2_weight)
        # nn.init.trunc_normal_(self.w3_weight)
        nn.init.xavier_uniform_(self.w1_weight)
        nn.init.xavier_uniform_(self.w2_weight)
        nn.init.xavier_uniform_(self.w3_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.w1_weight.T
        gate = self.swish(gate)
        up = x @ self.w3_weight.T
        gated_up = gate * up
        out = gated_up @ self.w2_weight.T
        return out
    

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, 
            in_features: Float[Tensor, " ..."], 
            dim: int
        ) -> torch.Tensor:
        exp_x = torch.exp(in_features - in_features.max(dim=dim, keepdim=True).values)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)
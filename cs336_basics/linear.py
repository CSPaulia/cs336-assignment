import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.__init_weight__()

        self.device = device
        self.dtype = dtype

    def __init_weight__(self):
        # nn.init.trunc_normal_(self.weight)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        return x
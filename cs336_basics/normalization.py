import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty((d_model,), device=device, dtype=dtype)
        )
        self.__init_weight__()

        self.device = device
        self.dtype = dtype

    def __init_weight__(self):
        # nn.init.trunc_normal_(self.weight)
        nn.init.ones_(self.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        x = x.to(self.dtype)
        return x
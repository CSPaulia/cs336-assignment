import torch
import torch.nn as nn
import einops

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device=None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=x.device) / self.d_k))
        angles = token_positions[..., None] * freqs
        x = einops.rearrange(
            x, 
            "... (d q) -> ... d q", 
            d = self.d_k // 2
        )
        x1 = x[..., 0] * torch.cos(angles) - x[..., 1] * torch.sin(angles)
        x2 = x[..., 0] * torch.sin(angles) + x[..., 1] * torch.cos(angles)
        x = torch.stack([x1, x2], dim=-1)
        x = einops.rearrange(
            x, 
            "... d q -> ... (d q)", 
            d = self.d_k // 2
        )
        return x
    
class NoPositionalEmbedding(nn.Module):
    def __init__(
            self,
            max_seq_len: int,
            device=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return x
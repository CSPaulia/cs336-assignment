import torch
import torch.nn as nn

from .attention import CausalMultiHeadSelfAttention
from .normalization import RMSNorm
from .activation import SwiGLU, Softmax, Swish
from .embedding import Embedding
from .linear import Linear

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int = None,
            theta: float = None,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.mha = CausalMultiHeadSelfAttention(
            d_model, 
            num_heads, 
            max_seq_len, 
            theta=theta,
            device=device,
            dtype=dtype,
        )

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.swiglu = SwiGLU(
            d_model, 
            d_ff, 
            device=device, 
            dtype=dtype,
        )

        # self.swiglu = Swish()
    
    # Pre-norm architecture
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        normed_x = self.norm1(x)
        if token_positions is None and self.theta is not None:
            token_positions = torch.arange(normed_x.shape[-2], device=normed_x.device)
        attn_out = self.mha(normed_x, token_positions)
        x = x + attn_out
        normed_x = self.norm2(x)
        ff_out = self.swiglu(normed_x)
        x = x + ff_out
        return x

    # Post-norm architecture
    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
    #     if token_positions is None and self.theta is not None:
    #         token_positions = torch.arange(x.shape[-2], device=x.device)
    #     attn_out = self.mha(x, token_positions)
    #     x = x + attn_out
    #     x = self.norm1(x)
    #     ff_out = self.swiglu(x)
    #     x = x + ff_out
    #     x = self.norm2(x)
    #     return x
    

class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = None,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, 
                num_heads, 
                d_ff, 
                context_length, 
                theta=rope_theta, 
                device=device, 
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # self.output_softmax = Softmax()

        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_embedding(x)
        # x = self.output_softmax(x, dim=-1)
        return x
import torch
import torch.nn as nn

import einops

from .activation import Softmax, SwiGLU
from .position_embedding import RotaryPositionalEmbedding, NoPositionalEmbedding
from .normalization import RMSNorm

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None
        ) -> torch.Tensor:
        d_k = query.shape[-1]
        qk = query @ key.transpose(-2, -1) / (d_k ** 0.5)
        if mask is not None:
            qk[mask == 0] = float("-inf")
        attn_weights = self.softmax(qk, dim=-1)
        output = attn_weights @ value
        return output
    

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = None,
            theta: float = None,
            device=None,
            dtype=None,
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.k_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.v_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.out_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.__init_weight__()

        self.sdpa = ScaledDotProductAttention()

        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta, 
                self.d_k, 
                max_seq_len
            )

        # self.rope = NoPositionalEmbedding(
        #     max_seq_len,
        # )

    def __init_weight__(self):
        # nn.init.trunc_normal_(self.q_weight)
        # nn.init.trunc_normal_(self.k_weight)
        # nn.init.trunc_normal_(self.v_weight)
        # nn.init.trunc_normal_(self.out_weight)
        nn.init.xavier_uniform_(self.q_weight)
        nn.init.xavier_uniform_(self.k_weight)
        nn.init.xavier_uniform_(self.v_weight)
        nn.init.xavier_uniform_(self.out_weight)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = x @ self.q_weight.T
        K = x @ self.k_weight.T
        V = x @ self.v_weight.T

        Q_h = einops.rearrange(
            Q,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads,
        )
        K_h = einops.rearrange(
            K,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads,
        )
        V_h = einops.rearrange(
            V,
            "... seq_len (h d_k) -> ... h seq_len d_k",
            h = self.num_heads,
        )

        if token_positions is not None and self.rope is not None:
            Q_h = self.rope(Q_h, token_positions)
            K_h = self.rope(K_h, token_positions)

        mask = torch.ones(
            (Q.shape[-2], K.shape[-2]), 
            device=x.device, 
            dtype=torch.bool
        ).tril()
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(Q_h.shape[0], Q_h.shape[1], 1, 1)

        hidden_h = self.sdpa(Q_h, K_h, V_h, mask)
        hidden = einops.rearrange(
            hidden_h,
            "... h seq_len d_k -> ... seq_len (h d_k)",
            h = self.num_heads,
        )
        out = hidden @ self.out_weight.T
        return out
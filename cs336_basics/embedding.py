import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.__init_weight__()

        self.device = device
        self.dtype = dtype

    def __init_weight__(self):
        # nn.init.trunc_normal_(self.weight)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.weight[token_ids]
        return x
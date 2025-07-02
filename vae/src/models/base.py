import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import List, Tuple, Iterable, Callable

from scvi.nn import FCLayers


def _identity(x):
    return x

# Build dynamic fc layers with integrated multi-head attention
class AttentionFCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,                     # number of genes
        n_out: int,
        n_hidden: int = 128,         # gene embedding dimension
        attention_heads: int = 4,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden

        if n_hidden % attention_heads != 0:
            raise ValueError(f'n_hidden {n_hidden} must be divisible by attention_heads {attention_heads}.')

        # Embedding per gene (maps each gene to n_hidden)
        self.input_proj = nn.Linear(1, n_hidden)

        # Transformer encoder layer (multi-head self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Optional layer norm
        self.layer_norm = nn.LayerNorm(n_hidden) if use_layer_norm else nn.Identity()

        # Output FC layer
        self.output_fc = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, n_genes)
        x = x.unsqueeze(-1)                        # (B, G, 1)
        x = self.input_proj(x)                     # (B, G, E)
        x = self.attn_encoder(x)                   # (B, G, E)
        x = self.layer_norm(x)                     # (B, G, E)
        x = x.mean(dim=1)                          # (B, E) - mean pooling over genes
        return self.output_fc(x)                   # (B, n_out)


# Attention Encoder
class AttentionEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        attention_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
    ):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.return_dist = return_dist

        self.encoder = AttentionFCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_hidden=n_hidden,
            attention_heads=attention_heads,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        return (dist, latent) if self.return_dist else (q_m, q_v, latent)

import torch
import torch.nn as nn

from src.modules._attention import MultiHeadAttention


class ProjectionBlock(nn.Module):
    """Simple feedforward block for VAEs."""

    def __init__(
        self,
        n_input: int,
        n_output: int | None = None,
        activation_fn: nn.Module = nn.GELU,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        # Initialize with basic linear layer
        layers = [nn.Linear(n_input, n_output)]
        
        # Normalization (choose one)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(n_output))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(n_output))

        # Nonlinearity
        layers.append(activation_fn())

        # Optional dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # Create full block
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.block(x)

    
class FcBlock(nn.Module):
    """Fully connected bottleneck block with optional residuals."""

    def __init__(
        self,
        n_input: int,
        ff_mult: int = 4,
        n_hidden: int | None = None,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        activation_fn: type[nn.Module] = nn.GELU,
        bias: bool = False,
        noise_std: float = 0.0,
        use_residuals: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.use_residuals = use_residuals
        self.noise_std = noise_std

        n_hidden = int(n_input * ff_mult) if n_hidden is None else n_hidden

        self.ff = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=bias),
            nn.LayerNorm(n_hidden) if use_layer_norm else nn.Identity(),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, n_input, bias=bias),
        )

        self.act = activation_fn()

    def forward(self, x: torch.Tensor, **kwargs):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        h = self.ff(x)

        if self.use_residuals:
            return self.act(x + h)
        else:
            return h

class TransformerBlock(nn.Module):
    """Transformer-style block with multi-head attention + feedforward + residuals."""

    def __init__(
        self, 
        n_input: int,
        n_head: int,
        head_size: int | None = None,
        ff_mult: int = 4,
        use_attention: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        activation_fn: nn.Module = nn.GELU,
        bias: bool = True,
        noise_std: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.noise_std = noise_std
        self.use_attention = use_attention

        # Attention over features
        if use_attention:
            self.attn = MultiHeadAttention(
                num_heads=n_head, 
                n_input=n_input, 
                head_size=head_size, 
                dropout_rate=dropout_rate,
                **kwargs
            )

        # Norm layers
        self.norm1 = nn.LayerNorm(n_input) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(n_input) if use_layer_norm else nn.Identity()

        # Feedforward MLP (expand → contract back to n_input)
        hidden_dim = ff_mult * n_input
        self.ff = nn.Sequential(
            nn.Linear(n_input, hidden_dim, bias=bias),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_input, bias=bias)
        )

        # Optional BatchNorm
        self.batch_norm = nn.BatchNorm1d(n_input) if use_batch_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, n_input)
        Returns:
            out: (B, n_input)
        """
        # --- Attention + residual ---
        if self.use_attention:
            attn_out = self.attn(self.norm1(x), **kwargs)              # (B, n_input)
            x = x + self.dropout(attn_out)

        # --- Feedforward + residual ---
        r = self.ff(self.norm2(x))                       # (B, n_input)
        if self.use_batch_norm and not isinstance(self.batch_norm, nn.Identity):
            r = self.batch_norm(r)
        out = x + self.dropout(r)

        # Add Gaussian noise if training
        if self.training and self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        return out
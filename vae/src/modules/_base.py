import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from collections.abc import Callable, Iterable
from typing import Literal, Iterable, Optional

from torch.distributions import Normal

from src.utils.constants import REGISTRY_KEYS
from src.utils.common import to_tensor


class ExternalClassEmbedding(nn.Module):
    """External Class embedding. Can be a mix of static pre-trained embedding and learnable control embedding."""
    def __init__(
        self, 
        cls_emb: torch.Tensor,
        cls_sim: torch.Tensor | None,
        ctrl_class_idx: int | None,
        use_control: bool = True,
        **kwargs
    ):
        super().__init__()
        # Save class embedding
        self.cls_emb = F.normalize(to_tensor(cls_emb), dim=-1)
        # Calculate class similarities
        self.cls_sim = to_tensor(cls_sim) if cls_sim is not None else self.cls_emb @ self.cls_emb.T
        # Get number of classes and set control class index
        self.cls_emb_dim = self.cls_emb.shape[-1]
        self.ctrl_class_idx = ctrl_class_idx
        # Initialize class embedding shape
        self._ncls = self.cls_emb.shape[0]
        self._ndims = self.cls_emb.shape[-1]
        # Initialize learnable control parameter
        if self.ctrl_class_idx is not None:
            cls_range = torch.arange(self._ncls)
            if self.ctrl_class_idx not in cls_range:
                raise IndexError(f'Control class index "{self.ctrl_class_idx}" is outside of class embedding range "0:{self.cls_emb_dim}".')
            # Remove control class from embedding
            if not use_control:
                no_ctrl_mask = self.ctrl_class_idx != cls_range
                self.cls_emb = self.cls_emb[no_ctrl_mask]
                self.cls_sim = self.cls_sim[no_ctrl_mask][:,no_ctrl_mask]
                # Update self._shape
                self._ncls = self.cls_emb.shape[0]
                self.use_control_emb = False
            else:
                # Learnable control embedding
                self.control_emb = torch.nn.Parameter(torch.randn(1, self.cls_emb_dim) * 0.02)
                self.use_control_emb = True
        else:
            # Disable control embedding
            self.use_control_emb = False

    def forward(self, device: int | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine output device
        _device = device if device is not None else self.cls_emb.device
        # Update control embedding and class similarities
        if self.use_control_emb:
            # Return static embedding + learnable control embedding
            self.cls_emb[self.ctrl_class_idx] = self.control_emb.squeeze(0)
            # Recalculate class similarities every time since control is changing
            self.cls_sim = self.cls_emb @ self.cls_emb.T
        # Move to device and return
        return self.cls_emb.to(device=_device), self.cls_sim.to(device=_device)
        
    @property
    def shape(self) -> tuple[int, int]:
        return (self._ncls, self._ndims)

def _identity(x):
    return x

class Block(nn.Module):
    """Simple feedforward block for VAEs."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        activation_fn: nn.Module = nn.LeakyReLU,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        **kwargs
    ):
        super().__init__()

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

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerBlock(nn.Module):
    """Transformer-style block with multi-head attention + feedforward + residuals."""

    def __init__(
        self, 
        n_input: int,
        n_head: int,
        head_size: int | None = None,
        ff_mult: int = 4,                 # expansion factor in FFN (like in Vaswani et al.)
        use_attention: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        activation_fn: nn.Module = nn.LeakyReLU,
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


class FunnelFCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_cat_list: Optional[Iterable[int]] = None,
        n_layers: int = 2,
        min_attn_dim: int = 64,
        max_attn_dim: int = 256,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        use_attention: bool = False,
        n_head: int = 4,
        bias: bool = True,
        inverted: bool = False,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.LeakyReLU,
        noise_std: float = 0.0,
        skip_first: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.inject_covariates = inject_covariates
        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.min_attn_dim = min_attn_dim
        self.kwargs = kwargs

        # Init modules
        self.layer_norm = nn.LayerNorm(n_out)
        self.block = Block

        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in (n_cat_list or [])]
        self.cat_dim = sum(self.n_cat_list)

        # Geometric decay over n hidden to output layer
        hidden_dims = np.geomspace(n_hidden, n_out, num=n_layers + 1).astype(int)
        # Set n_hidden as bottleneck first dimension if its not the same as n_in
        e = 0
        if n_hidden != n_in:
            e = 1
            hidden_dims = np.r_[n_in, hidden_dims]
        
        # Ensure its divisible by n_heads if attention is used
        if use_attention:
            if inverted:
                hidden_dims -= np.concatenate(((hidden_dims % n_head)[:-1], [0]))
            else:
                hidden_dims -= np.concatenate(([0], (hidden_dims % n_head)[1:]))
        # Init layers
        self.layers = nn.ModuleList()

        for i in range(n_layers + e):
            in_dim = hidden_dims[i] + self.cat_dim * self.inject_into_layer(i)
            out_dim = hidden_dims[i + 1]
            # Determine whether to to use attention for block, never use on the first layer
            is_attention_block = False if (i == 0 and skip_first) else (min_attn_dim <= in_dim <= max_attn_dim) and use_attention
            # Create block for each layer
            block = self.block(
                n_input=in_dim, 
                n_output=out_dim, 
                n_head=n_head,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=use_activation,
                use_attention=is_attention_block,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
                bias=bias,
                noise_std=noise_std
            )
            self.layers.append(block)
    
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        return layer_num==0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor) -> torch.Tensor:
        # One-hot encode categorical covariates
        one_hot_cat_list = []
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat > 1:
                cat = cat.squeeze(-1) if cat.dim() == 2 and cat.size(-1) == 1 else cat
                one_hot = F.one_hot(cat, num_classes=n_cat).float()
                one_hot_cat_list.append(one_hot)

        for i, layer in enumerate(self.layers):
            if self.inject_into_layer(i):
                if one_hot_cat_list:
                    cat_input = torch.cat(one_hot_cat_list, dim=-1)
                    cat_input = cat_input.expand(x.shape[0], cat_input.shape[-1])
                    x = torch.cat([x, cat_input], dim=-1)
            x = layer(x)
        # Apply layer norm
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x


class AttentionLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_cat_list: Optional[Iterable[int]] = None,
        n_layers: int = 2,
        n_attn_layers: int = 3,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        use_attention: bool = True,
        gene_emb_dim: int | None = None,
        n_head: int = 4,
        bias: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.LeakyReLU,
        noise_std: float = 0.0,
        linear_encoder: bool = True,
        compress_emb_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.inject_covariates = inject_covariates
        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.n_hidden = n_hidden
        self.compress_emb = compress_emb_dim is not None and compress_emb_dim < 2048
        self.kwargs = kwargs

        # Init modules
        self.layer_norm = nn.LayerNorm(n_out)

        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in (n_cat_list or [])]
        self.cat_dim = sum(self.n_cat_list)

        # Encode down to bottleneck latent space
        if linear_encoder:
            self.encoder = nn.Linear(n_in, n_out)
        else:
            self.encoder = FunnelFCLayers(
                n_in=n_in, 
                n_out=n_out, 
                n_hidden=n_hidden, 
                n_layers=n_layers,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
                dropout_rate=dropout_rate,
            )
        # Add encoder for embedding dimensions
        emb_dim = None
        if gene_emb_dim is not None:
            if self.compress_emb:
                # Compress embedding using a linear layer
                self.embedding_compressor = nn.Linear(gene_emb_dim, compress_emb_dim)
                emb_dim = compress_emb_dim
            else:
                # Fall back to full embedding (likely blows memory limits unless its already compressed)
                emb_dim = gene_emb_dim
        # Introduce multiple attention layers of same dimension
        self.layers = nn.ModuleList()
        for _ in np.arange(n_attn_layers):
            # Create block for each layer
            block = TransformerBlock(
                n_input=n_out,
                n_head=n_head,
                use_attention=use_attention,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
                bias=bias,
                noise_std=noise_std,
                emb_dim=emb_dim,
                **kwargs
            )
            self.layers.append(block)
    
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor, gene_embedding: torch.Tensor | None = None) -> torch.Tensor:
        # First linear projection
        x = self.encoder(x)
        # Project gene embedding if external is given
        if gene_embedding is not None:
            g = gene_embedding.to(x.device)
            # Compress embedding dimensions
            if self.compress_emb:
                g = self.embedding_compressor(g.T).T
            # Compress features with gene encoder (no grad)
            with torch.no_grad():
                feature_emb = self.encoder(g)
        else:
            feature_emb = None
        # One-hot encode categorical covariates
        one_hot_cat_list = []
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat > 1:
                cat = cat.squeeze(-1) if cat.dim() == 2 and cat.size(-1) == 1 else cat
                one_hot = F.one_hot(cat, num_classes=n_cat).float()
                one_hot_cat_list.append(one_hot)

        for i, layer in enumerate(self.layers):
            if self.inject_into_layer(i):
                if one_hot_cat_list:
                    cat_input = torch.cat(one_hot_cat_list, dim=-1)
                    cat_input = cat_input.expand(x.shape[0], cat_input.shape[-1])
                    x = torch.cat([x, cat_input], dim=-1)
            x = layer(x, feature_emb=feature_emb)
        # Apply layer norm
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x
    

class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.LeakyReLU,
        linear: bool = False,
        **kwargs
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        self.linear = linear
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        cat_dim = sum(self.n_cat_list)
        # Add a single linear layer
        if linear:
            self.fc_layers = nn.Linear(n_in + cat_dim, n_out)
        else:
            self.fc_layers = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            f"Layer {i}",
                            nn.Sequential(
                                nn.Linear(
                                    n_in + cat_dim * self.inject_into_layer(i),
                                    n_out,
                                    bias=bias,
                                ),
                                # non-default params come from defaults in original Tensorflow
                                # implementation
                                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                                if use_batch_norm
                                else None,
                                nn.LayerNorm(n_out, elementwise_affine=False)
                                if use_layer_norm
                                else None,
                                activation_fn() if use_activation else None,
                                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            ),
                        )
                        for i, (n_in, n_out) in enumerate(
                            zip(layers_dim[:-1], layers_dim[1:], strict=True)
                        )
                    ]
                )
            )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        # Apply linear mapping
        if self.linear:
            if x.dim() == 3:
                one_hot_cat_list_layer = [
                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                    for o in one_hot_cat_list
                ]
            else:
                one_hot_cat_list_layer = one_hot_cat_list
            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
            return self.fc_layers(x)
        # Forward through layers
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            if (
                                x.device.type == "mps"
                            ):  # TODO: remove this when MPS supports for loop.
                                x = torch.cat(
                                    [(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0
                                )
                            else:
                                x = torch.cat(
                                    [layer(slice_x).unsqueeze(0) for slice_x in x], dim=0
                                )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x
    

class FeatureAttention(nn.Module):
    """
    Per-sample self-attention mechanism over input features.
    For each sample in the batch, computes attention between its features.
    """
    def __init__(self, n_input: int, head_size: int | None = None, dropout_rate: float = 0.1):
        super().__init__()
        self.head_size = n_input if head_size is None else head_size 
        self.query = nn.Linear(n_input, self.head_size, bias=False)
        self.key = nn.Linear(n_input, self.head_size, bias=False)
        self.value = nn.Linear(n_input, self.head_size, bias=False)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor = None):
        """
        Args:
            x: Tensor of shape (batch, features)
            feature_mask: Optional mask of shape (batch, features) with 0s for features to mask

        Returns:
            attended: Tensor of shape (batch, features)
        """

        # Compute dot product between all features for each sample
        q = self.query(x)               # (batch, features)
        k = self.key(x)                 # (batch, features)
        v = self.value(x)               # (batch, features)

        # Compute attention: (batch, features, features)
        attn_weights = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) * self.head_size**-0.5     # per sample, scale by attention head size

        if feature_mask is not None:
            # Mask has shape (batch, features) --> (batch, 1, features)
            attn_weights = attn_weights.masked_fill(
                feature_mask.unsqueeze(1) == 0, -torch.inf
            )
        # Normalize attention weights to sum to 1
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch, features, features)
        # Apply dropout
        if self.dropout_rate > 0:
            attn_weights = self.dropout(attn_weights)
        # Apply attention: (batch, features, features) x (batch, features, 1) --> (batch, features, 1)
        attended = torch.bmm(attn_weights, v.unsqueeze(-1)).squeeze(-1)  # (batch, features)

        return attended
    

class MultiHeadAttentionIterative(nn.Module):
    """Multiple FeatureAttention heads in parallel."""

    def __init__(self, num_heads: int, n_input: int, head_size: int | None = None, dropout_rate: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([FeatureAttention(n_input=n_input, head_size=head_size) for _ in range(num_heads)])
        attn_dim = int(head_size * num_heads)
        self.proj = nn.Linear(attn_dim, n_input)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    """Single attention head with its own projections"""
    def __init__(self, embedding_dim: int, head_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.head_size = head_size
        # Each head has its own projections
        self.q_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.k_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.v_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_emb: (B,F,E) embedded input
        Returns:
            out: (B,F,L) attention output
            attn: (B,F,F) attention weights
        """
        # Project to Q,K,V
        q = self.q_proj(x_emb)  # (B,F,L)
        k = self.k_proj(x_emb)  # (B,F,L)
        v = self.v_proj(x_emb)  # (B,F,L)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)  # (B,F,F)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (B,F,L)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        num_heads: int,
        head_size: int | None = None,
        dropout_rate: float = 0.1,
        sequential: bool = True,
        emb_dim: int | None = None,
        buffer_n_step: int | None = None,
        min_buffer_steps: int = 5000,
        max_buffer_entries: int = 20,
        buffer_mode: Literal['train', 'val'] = 'train',
        use_ext_emb: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.n_input = n_input
        feature_input = n_input if emb_dim is None else emb_dim
        self.head_size = head_size if head_size is not None else feature_input // num_heads
        self.d_model = self.head_size * num_heads

        # learnable embeddings for each feature (gene)
        if not use_ext_emb:
            self.feature_emb = nn.Embedding(n_input, self.d_model)
        
        # Create separate attention heads
        self.heads = nn.ModuleList([
            AttentionHead(
                embedding_dim=self.d_model,
                head_size=self.head_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_heads)
        ])

        # output projection: concat(H*L) -> 1
        self.out_proj = nn.Linear(self.d_model, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.sequential = sequential
        # If not None and bigger than min steps, save a buffer of the attention layer
        self.record_each_n_step = buffer_n_step if buffer_n_step is not None and buffer_n_step > min_buffer_steps else None
        self.buffer_mode = buffer_mode
        self._buffer_idx = 0
        # Register buffer TODO: implement buffer_mode switch
        if self.record_each_n_step is not None:
            self.max_buffer_entries = max_buffer_entries
            self.register_buffer('attn_buffer', [], persistent=False)
        
    def forward(self, x: torch.Tensor, feature_emb: torch.Tensor | None = None, return_attn: bool = False):
        """
        Args:
            x: (B, F) expression matrix
            feature_emb (optional): (F, E) external feature embedding matrix
        Returns:
            out: (B, F)
            attn (optional): (B, H, F, F)  # only if return_attn=True
        """
        #self._buffer_idx += 1
        _record_step = False
        #_record_step = self.record_each_n_step is not None and self._buffer_idx % self.record_each_n_step == 0 and self.training
        # Get feature embeddings
        _feature_emb = feature_emb.T if feature_emb is not None else self.feature_emb.weight
        x_emb = torch.einsum("bf,fe->bfe", x, _feature_emb)  # (B,F,E)

        # Process each head
        out_heads = []
        attn_list = [] if return_attn else None

        # Run each attention head
        for head in self.heads:
            out_h, attn = head(x_emb)  # (B,F,L), (B,F,F)
            out_heads.append(out_h)
            if return_attn or _record_step:
                attn_list.append(attn.detach().cpu())

        # Concatenate outputs from all heads
        out = torch.cat(out_heads, dim=-1)  # (B,F,H*L)

        # Project back to output dimension
        out = self.out_proj(out).squeeze(-1)  # (B,F)
        out = self.dropout(out)

        # Save attention maps if requested
        if _record_step:
            if len(self.attn_buffer) > self.max_buffer_entries:
                self.register_buffer('attn_buffer', [], persistent=False)
            self.attn_buffer.append(torch.stack(attn_list, dim=1))  # (B,H,F,F)

        if return_attn:
            return out, (torch.stack(attn_list, dim=1) if attn_list else None)
        return out
    

class MultiHeadAttentionMM(nn.Module):
    """
    Multi-head feature self-attention (no sequence axis), loop-free via batched bmm.

    Shapes:
      x: (B, F) where F == n_input
      head_size = L  (defaults to n_input)
      q,k,v per head: (B, L)
      attention per head: (B, L, L)
      concat heads: (B, H*L)
      proj -> (B, n_input)

    Performance tips:
      - Keep record_attn=False during training (set True only when analyzing).
      - Use AMP/bfloat16 if possible.
      - Consider smaller head_size if L is large (cost ~ O(B*H*L^2)).
    """

    def __init__(
        self,
        num_heads: int,
        n_input: int,
        head_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        record_attn: bool = False,
        store_attn_on_cpu: bool = True,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.n_input = int(n_input)
        self.head_size = int(n_input if head_size is None else head_size)  # L
        self.scale = self.head_size ** -0.5

        H, L = self.num_heads, self.head_size

        # Single projections for all heads: (B, F) -> (B, H*L) -> view to (B, H, L)
        self.query = nn.Linear(self.n_input, H * L, bias=False)
        self.key   = nn.Linear(self.n_input, H * L, bias=False)
        self.value = nn.Linear(self.n_input, H * L, bias=False)

        # Concat heads -> project back to (B, n_input)
        self.proj = nn.Linear(H * L, self.n_input, bias=True)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        # Attention recording controls
        self.record_attn = bool(record_attn)
        self.store_attn_on_cpu = bool(store_attn_on_cpu)
        self.register_buffer("last_attn_weights", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _maybe_record(self, attn_bhll: torch.Tensor):
        if not self.record_attn:
            # keep an empty tensor to signal "not recording"
            self.last_attn_weights = attn_bhll.new_empty(0)
            return
        attn = attn_bhll.detach()
        if self.store_attn_on_cpu:
            attn = attn.to("cpu", non_blocking=True)
        self.last_attn_weights = attn  # shape: (B, H, L, L)

    def forward(
        self,
        x: torch.Tensor,                         # (B, F)
        feature_mask: Optional[torch.Tensor] = None,  # (B, L) with 1=keep, 0=mask (keys)
        return_attn: bool = False
    ):
        B, F = x.shape
        H, L = self.num_heads, self.head_size
        BH = B * H

        # Project all heads at once: (B, F) -> (B, H, L)
        q = self.query(x).view(B, H, L)
        k = self.key(x).view(B, H, L)
        v = self.value(x).view(B, H, L)

        # Flatten batch*heads for batched bmm
        q_bh = q.reshape(BH, L)                      # (BH, L)
        k_bh = k.reshape(BH, L)                      # (BH, L)
        v_bh = v.reshape(BH, L)                      # (BH, L)

        # Scores via outer products per (B,H): (BH, L, 1) @ (BH, 1, L) -> (BH, L, L)
        scores = torch.bmm(q_bh.unsqueeze(-1), k_bh.unsqueeze(-2))  # (BH, L, L)
        scores.mul_(self.scale)

        # Key mask: broadcast across query positions
        if feature_mask is not None:
            if feature_mask.shape != (B, L):
                raise ValueError(f"feature_mask must be (B, {L}) but got {tuple(feature_mask.shape)}")
            key_mask_bh = feature_mask.repeat_interleave(H, dim=0)          # (BH, L)
            # mask zeros (to -inf) along key axis
            scores = scores.masked_fill(key_mask_bh.unsqueeze(1) == 0,
                                        torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)          # (BH, L, L)
        if self.attn_dropout.p > 0:
            attn = self.attn_dropout(attn)

        # Weighted sum: (BH, L, L) @ (BH, L, 1) -> (BH, L)
        out_bh = torch.bmm(attn, v_bh.unsqueeze(-1)).squeeze(-1)  # (BH, L)

        # Restore (B, H, L) -> concat heads -> (B, H*L)
        out = out_bh.view(B, H, L).reshape(B, H * L)              # (B, H*L)

        # Final projection back to (B, n_input)
        out = self.proj(out)                                      # (B, n_input)
        if self.out_dropout.p > 0:
            out = self.out_dropout(out)

        # Record attention (reshape back to (B,H,L,L)) only if requested
        #self._maybe_record(attn.view(B, H, L, L))

        if return_attn:
            return out, self.last_attn_weights if self.last_attn_weights.numel() else attn.view(B, H, L, L)
        return out

# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        use_feature_mask: bool = False,
        drop_prob: float = 0.25,
        encoder_type: Literal['funnel', 'fc', 'transformer'] = 'funnel',
        **kwargs,
    ):
        super().__init__()

        # Choose encoder type
        if encoder_type == 'funnel':
            self.fclayers_class = FunnelFCLayers
        elif encoder_type == 'fc':
            self.fclayers_class = FCLayers
        elif encoder_type == 'transformer':
            self.fclayers_class = AttentionLayers

        self.encoder_type = encoder_type
        self.distribution = distribution
        self.var_eps = var_eps
        self.n_input = n_input
        # Setup encoder layers
        funnel_out_dim = 2 * n_output
        self.encoder = self.fclayers_class(
            n_in=n_input,
            n_out=funnel_out_dim,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(funnel_out_dim, n_output)
        self.var_encoder = nn.Linear(funnel_out_dim, n_output)
        self.return_dist = return_dist
        self.use_feature_mask = use_feature_mask
        self.drop_prob = drop_prob

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int, g: torch.Tensor | None = None):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        feature_mask = None
        if self.training and self.use_feature_mask and self.drop_prob > 0:
            # Sample mask: 1 for keep, 0 for drop
            feature_mask = torch.bernoulli(torch.full((self.n_input,), 1 - self.drop_prob))  # shape: (num_features,)
            feature_mask = feature_mask.view(1, -1)  # shape: (1, num_features)
            feature_mask = feature_mask.expand(x.shape[0], -1)  # broadcast to full batch
            feature_mask = feature_mask.to(x.device)
            x = x * feature_mask
        # Parameters for latent distribution
        if self.encoder_type == 'transformer':
            q = self.encoder(x, *cat_list, gene_embedding=g)
        else:
            q = self.encoder(x, *cat_list)
        # Project to mean and variance
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        # Create Normal distribution and sample a latent
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent

# Decoder
class DecoderSCVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        use_funnel: bool = False,    
        linear_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        funnel_out_dim = n_output // 2
        # Initialize px decoder
        if linear_decoder:
            # Create linear decoder
            self.px_decoder = FCLayers(
                n_in=n_input, 
                n_out=funnel_out_dim, 
                n_cat_list=n_cat_list, 
                dropout_rate=0,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                inverted=True,
                linear=True
            )
        else:
            # Create non-linear decoder, either funneled or transformer-like
            fc_layer_class = FunnelFCLayers if use_funnel else FCLayers
            
            self.px_decoder = fc_layer_class(
                n_in=n_input,
                n_out=funnel_out_dim,
                n_hidden=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                dropout_rate=0,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                inverted=True,
                **kwargs,
            )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(funnel_out_dim, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(funnel_out_dim, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(funnel_out_dim, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class EmbeddingClassifier(nn.Module):
    """Classifier where latent attends to class embeddings, optionally externally provided,
    using dot product or cosine similarity. Compatible with external CE loss logic.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        logits: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation_fn: nn.Module = nn.LeakyReLU,
        class_embed_dim: int = 128,
        shared_projection_dim: int | None = None,
        skip_projection: bool = False,
        use_cosine_similarity: bool = True,
        temperature: float = 1.0,
        return_latents: bool = True,
        mode: Literal["latent_to_class", "class_to_latent", "shared"] = "latent_to_class",
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        self.use_cosine_similarity = use_cosine_similarity
        self.class_embed_dim = class_embed_dim
        self.n_labels = n_labels
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.return_latents = return_latents
        self.mode = mode

        # ---- Helper for building projections ---- #
        def build_projection(n_in: int, n_out: int) -> nn.Module:
            """Creates a projection block based on n_hidden/n_layers settings."""
            if skip_projection and n_in == n_out:
                return nn.Identity()
            # Add funnel layer projections
            if n_hidden > 0 and n_layers > 0:
                return FunnelFCLayers(
                    n_in=n_in,
                    n_out=n_out,
                    n_layers=n_layers,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            # Project with a single linear layer
            else:
                layers = [nn.Linear(n_in, n_out)]
                if use_layer_norm:
                    layers.append(nn.LayerNorm(n_out))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                return nn.Sequential(*layers)

        # Latent --> Class embedding space projection
        if mode == "latent_to_class":
            self.latent_projection = build_projection(n_input, class_embed_dim)
            self.class_projection = nn.Identity()
        # Class embedding space --> latent space projection
        elif mode == "class_to_latent":
            self.latent_projection = nn.Identity()
            self.class_projection = build_projection(class_embed_dim, n_input)
        # Latent --> shared dim <-- Class embedding space
        elif mode == "shared":
            if shared_projection_dim is None:
                shared_projection_dim = min(n_input, class_embed_dim)
            self.latent_projection = build_projection(n_input, shared_projection_dim)
            self.class_projection = build_projection(class_embed_dim, shared_projection_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- Dropout ---- #
        self.dropout = nn.Dropout(p=dropout_rate)

        # ---- Learnable class embeddings ---- #
        self.learned_class_embeds = nn.Parameter(torch.randn(n_labels, class_embed_dim))

    def forward(self, x: torch.Tensor, class_embeds: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, latent_dim)
        class_embeds : Optional[Tensor], shape (n_labels, embed_dim)
        """
        # Get class embeddings
        class_embeds = (
            self.learned_class_embeds if class_embeds is None else class_embeds.to(x.device)
        )

        # Apply projections
        z = self.latent_projection(x)               # (batch, d)
        c = self.class_projection(class_embeds)     # (n_labels, d)
        if self.dropout_rate > 0:
            z = self.dropout(z)

        # Compute logits
        if self.use_cosine_similarity:
            z_norm = F.normalize(z, dim=-1)
            c_norm = F.normalize(c, dim=-1)
            logits = torch.matmul(z_norm, c_norm.T) / self.temperature
            _z, _c = z_norm, c_norm
        else:
            logits = torch.matmul(z, c.T) / self.temperature
            _z, _c = z, c

        output = logits if self.logits else F.softmax(logits, dim=-1)
        return (output, _z, _c) if self.return_latents else output


class Classifier(nn.Module):
    """Basic fully-connected NN classifier.

    Parameters
    ----------
    n_input
        Number of input dimensions
    n_hidden
        Number of nodes in hidden layer(s). If `0`, the classifier only consists of a
        single linear layer.
    n_labels
        Numput of outputs dimensions
    n_layers
        Number of hidden layers. If `0`, the classifier only consists of a single
        linear layer.
    dropout_rate
        dropout_rate for nodes
    logits
        Return logits or not
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    activation_fn
        Valid activation function from torch.nn
    **kwargs
        Keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        logits: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        layers = []

        if n_hidden > 0 and n_layers > 0:
            layers.append(
                FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            )
        else:
            n_hidden = n_input

        layers.append(nn.Linear(n_hidden, n_labels))

        if not logits:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """Forward computation."""
        return self.classifier(x)

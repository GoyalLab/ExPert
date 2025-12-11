import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from collections.abc import Callable, Iterable
from typing import Literal, Iterable, Optional

from torch.distributions import Normal
from transformers import AutoTokenizer, AutoModel

from src.utils.constants import MODULE_KEYS, PREDICTION_KEYS
from src.utils.io import to_tensor


class MemoryQueue:
    def __init__(
            self,
            dim: int,
            n: int = 12, 
            device: str = "cpu"
        ):
        self.size = int(n * dim)
        self.dim = dim
        self.device = device

        self.queue_z = torch.zeros(self.size, dim, device=device)
        self.queue_y = torch.full((self.size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.filled = False

    @torch.no_grad()
    def enqueue(self, z, y):
        """Store normalized embeddings and labels."""
        z = F.normalize(z, dim=-1)
        b = z.size(0)
        if b >= self.size:
            # if batch larger than queue, keep most recent samples
            self.queue_z = z[-self.size:].clone()
            self.queue_y = y[-self.size:].clone()
            self.ptr = 0
            self.filled = True
            return

        end = self.ptr + b
        if end > self.size:
            overflow = end - self.size
            self.queue_z[self.ptr:] = z[: b - overflow]
            self.queue_y[self.ptr:] = y[: b - overflow]
            self.queue_z[:overflow] = z[b - overflow :]
            self.queue_y[:overflow] = y[b - overflow :]
            self.ptr = overflow
            self.filled = True
        else:
            self.queue_z[self.ptr:end] = z
            self.queue_y[self.ptr:end] = y
            self.ptr = end % self.size
            if self.ptr == 0:
                self.filled = True

    def get(self):
        """Return current queue contents."""
        if not self.filled and self.ptr == 0:
            return None, None
        n = self.size if self.filled else self.ptr
        return self.queue_z[:n].clone(), self.queue_y[:n].clone()
    

class ContextEmbedding(nn.Module):
    """Wrapper for context embeddings with optional unseen-context buffer."""
    def __init__(
        self,
        n_contexts: int,
        emb_dim: int = 5,
        add_unseen_buffer: bool = True,
        set_buffer_to_mean: bool = True,
        observed_buffer_prob: float = 0.05,
        ext_emb: torch.Tensor | None = None,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        self.add_unseen_buffer = add_unseen_buffer
        self.set_buffer_to_mean = set_buffer_to_mean
        self.observed_buffer_prob = observed_buffer_prob

        n_total = n_contexts + (1 if add_unseen_buffer else 0)
        # Initialize a learnable context embedding
        if ext_emb is None:
            self.embedding = nn.Embedding(n_total, emb_dim)
        else:
            # Set inital weights to pre-trained embedding
            self.embedding = nn.Embedding.from_pretrained(ext_emb, freeze=freeze_pretrained)

        # Optionally initialize unseen buffer to mean of others
        if set_buffer_to_mean and add_unseen_buffer:
            with torch.no_grad():
                mean_vec = self.embedding.weight[:-1].mean(0, keepdim=True)
                self.embedding.weight[-1] = mean_vec

    def forward(self, context_idx: torch.Tensor):
        """Return inflated embedding by context indices of batch."""
        # With probability p, replace some context indices with buffer index
        if self.training and self.add_unseen_buffer and self.observed_buffer_prob > 0:
            mask = torch.rand_like(context_idx.float()) < self.observed_buffer_prob
            context_idx = torch.where(mask, 
                                      torch.full_like(context_idx, self.unseen_index), 
                                      context_idx)
        # Return inflated context embedding of (b,c)
        return self.embedding(context_idx)

    @property
    def unseen_index(self):
        """Index of the unseen context embedding."""
        return self.embedding.num_embeddings - 1
    
    @property
    def weight(self):
        """Return embedding weight"""
        return self.embedding.weight


class ExternalClassEmbedding(nn.Module):
    """External Class embedding. Can be a mix of static pre-trained embedding and learnable control embedding."""
    def __init__(
        self, 
        cls_emb: torch.Tensor,
        cls_sim: torch.Tensor | None,
        ctrl_class_idx: int | None,
        use_control: bool = True,
        device: str | None = None,
        **kwargs
    ):
        super().__init__()
        self.device = device
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

    def forward(self, labels: torch.Tensor | None = None, n_negatives: int | None = None, device: int | str | None = None) -> torch.Tensor:
        # Determine output device
        _device = device if device is not None else self.device
        # Update control embedding and class similarities
        if self.use_control_emb:
            # Return static embedding + learnable control embedding
            self.cls_emb[self.ctrl_class_idx] = self.control_emb.squeeze(0)
        # Create local output parameters
        cls_emb = self.cls_emb.to(device=_device)
        # Subset to observed classes only
        if labels is not None:
            # Remove all class indices that are not observed during training
            indices = torch.arange(self._ncls, device=labels.device)
            is_observed_mask = torch.isin(indices, labels.unique())
            # Randomly include negative labels
            if n_negatives is not None and n_negatives > 0:
                # Get number of total negatives
                n_neg = min(n_negatives, (~is_observed_mask).sum().item())
                # Get indices of unobserved classes
                neg_indices = indices[~is_observed_mask]
                # Randomly sample n_neg indices
                neg_idx = torch.randperm(len(neg_indices))[:n_neg]
                neg_samples = neg_indices[neg_idx]
                # Add negative samples to observed mask
                is_observed_mask[neg_samples.long()] = True
            # Mask all other classes with zeros
            cls_emb = F.normalize(cls_emb, dim=-1)
            cls_emb = cls_emb * (is_observed_mask).float().unsqueeze(-1)

        # Move to device and return
        return cls_emb
        
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
        activation_fn: nn.Module = nn.GELU,
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
        activation_fn: nn.Module = nn.GELU,
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

        # If n_hidden = -1, set it to n_in
        n_hidden = n_in if n_hidden == -1 else n_hidden

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
        activation_fn: nn.Module = nn.GELU,
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
        activation_fn: nn.Module = nn.GELU,
        add_last_linear: bool = False,
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

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, feature_dim: int, context_dim: int, weight: float = 0.1):
        super().__init__()
        self.gamma = nn.Linear(context_dim, feature_dim)
        self.beta = nn.Linear(context_dim, feature_dim)
        self.weight = weight

    def forward(self, z: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(context_emb) * self.weight
        beta = self.beta(context_emb) * self.weight
        return gamma * z + beta  # FiLM modulation

class ContextFeatureAttention(nn.Module):
    def __init__(self, n_features: int, d_ctx: int, d_hidden: int):
        super().__init__()
        self.query = nn.Linear(d_ctx, d_hidden)
        self.key   = nn.Linear(n_features, d_hidden)
        self.value = nn.Linear(n_features, n_features)
        self.scale = d_hidden ** 0.5

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor):
        # x: [batch, n_genes], c_emb: [batch, d_ctx]
        Q = self.query(c_emb).unsqueeze(1)       # [B, 1, d_h]
        K = self.key(x).unsqueeze(1)             # [B, 1, d_h]
        attn = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        x_attn = attn @ self.value(x).unsqueeze(1)
        return x + x_attn.squeeze(1)             # context-modulated features

# Encoder
class Encoder(nn.Module):
    """Encode data into latent space, optionally conditioned on context via concatenation or FiLM."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_dim_context_emb: int | None = None,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        use_feature_mask: bool = False,
        drop_prob: float = 0.01,
        encoder_type: Literal["funnel", "fc", "transformer"] = "funnel",
        context_integration_method : Literal["concat", "film", "attention"] | None = "attention",
        use_context_inference: bool = True,
        use_learnable_temperature: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Choose encoder class
        if encoder_type == "funnel":
            self.fclayers_class = FunnelFCLayers
        elif encoder_type == "fc":
            self.fclayers_class = FCLayers
        elif encoder_type == "transformer":
            self.fclayers_class = AttentionLayers
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Save init params
        self.encoder_type = encoder_type
        self.distribution = distribution
        self.var_eps = var_eps
        self.n_input = n_input
        self.n_dim_context_emb = n_dim_context_emb or 0
        self.use_feature_mask = use_feature_mask
        self.drop_prob = drop_prob
        self.context_integration_method = context_integration_method
        self.use_context_inference = use_context_inference
        self.use_learnable_temperature = use_learnable_temperature
        
        # Set encoder output to double the latent dimension
        funnel_out_dim = 2 * n_output
        if context_integration_method is not None and self.context_integration_method not in ["concat", "film", "attention"]:
            raise ValueError(f'Invalid context integration method: "{context_integration_method}"')
        # Use concatenated dimensions as input
        if self.context_integration_method == 'concat':
            encoder_input_dim = n_input + self.n_dim_context_emb
        # Use input dimension
        else:
            encoder_input_dim = n_input

        # Base encoder layers
        self.encoder = self.fclayers_class(
            n_in=encoder_input_dim,
            n_out=funnel_out_dim,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        # Initialize context intergration method:
        if self.context_integration_method == 'film' and self.n_dim_context_emb > 0:
            self.ctx_integration = FiLM(feature_dim=encoder_input_dim, context_dim=self.n_dim_context_emb)
        elif self.context_integration_method == 'attention' and self.n_dim_context_emb > 0:
            self.ctx_integration = ContextFeatureAttention(n_features=encoder_input_dim, d_ctx=self.n_dim_context_emb, d_hidden=n_hidden)
        else:
            # Just concatenate x and context
            self.ctx_integration = None
        # Disable context integration if method is set to None
        if self.context_integration_method is None:
            self.use_context_inference = False
        # Add context projection
        if self.use_context_inference:
            self.context_proj = nn.Linear(self.n_dim_context_emb, funnel_out_dim)
        else:
            # Skip context integration
            self.context_proj = None

        # Create a learnable temperature scaling
        if use_learnable_temperature:
            self._temperature = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # Fall back to static default temperature
        else:
            self._temperature = temperature

        # Create base vae projections
        self.mean_encoder = nn.Linear(funnel_out_dim, n_output)
        self.var_encoder = nn.Linear(funnel_out_dim, n_output)
        self.var_activation = torch.exp if var_activation is None else var_activation
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else _identity

        # Debug
        self.c = 0

    @property
    def temperature(self) -> torch.Tensor:
        if self.use_learnable_temperature:
            # Calculate logit scale
            return 1.0 / self._temperature.clamp(-1, 4.6).exp()
        else:
            return self._temperature
    
    def forward(self, x: torch.Tensor, *cat_list: int, g: torch.Tensor | None = None, context_emb: torch.Tensor | None = None):
        # Optional feature masking
        if self.training and self.use_feature_mask and self.drop_prob > 0:
            mask = torch.bernoulli(torch.full((self.n_input,), 1 - self.drop_prob, device=x.device)).view(1, -1)
            x = x * mask.expand(x.shape[0], -1)

        # Concatenate or modulate by context
        ctx_logits, b_ctx_emb = None, None
        if self.n_dim_context_emb > 0 and context_emb is not None and self.use_context_inference:
            # Encode x only TODO: try separate linear projection for disentanglement
            h_x = self.encoder(x, *cat_list) if self.encoder_type != "transformer" else self.encoder(x, *cat_list, gene_embedding=g)
            # Normalize latent space for cosine similarity
            h_x = F.normalize(h_x, dim=-1)
            # Project contexts to shared dimensionality
            h_c = F.normalize(self.context_proj(context_emb), dim=-1)            # (n_contexts, d_h)
            # Calculate context similarity
            ctx_logits = h_x @ h_c.T / self.temperature         # (b, d_h) @ (d_h, n_contexts) = (b, n_contexts)
            # Select batch-relevant context information based on highest softmax (b, d_h)
            b_ctx_emb = (torch.softmax(ctx_logits, dim=-1).unsqueeze(-1) * context_emb.unsqueeze(0)).sum(dim=1)
            
            # Add context integration to batch
            if self.ctx_integration is not None:
                # FiLM or attention
                x = self.ctx_integration(x, b_ctx_emb)
            else:
                # Simple concatenation
                x = torch.cat([x, b_ctx_emb], dim=-1)
            
            # Do a second encoder forward pass
            q = self.encoder(x, *cat_list) if self.encoder_type != "transformer" else self.encoder(x, *cat_list, gene_embedding=g)
        else:
            # Do a simple forward pass with just x and no additional information
            q = self.encoder(x, *cat_list) if self.encoder_type != "transformer" else self.encoder(x, *cat_list, gene_embedding=g)
        
        # Debug
        self.c = self.c + 1
        # Latent heads
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        qz = Normal(q_m, q_v.sqrt())
        z = self.z_transformation(qz.rsample())
        return {
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QZM_KEY: q_m,
            MODULE_KEYS.QZV_KEY: q_v,
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.CTX_LOGITS_KEY: ctx_logits,
        }


# Decoder
class DecoderSCVI(nn.Module):
    """Decodes latent variables, optionally conditioned on context via concatenation or FiLM."""

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
        n_dim_context_emb: int | None = None,
        n_ctx_frac: float | None = 0.5,
        n_context_compression: int = 2,
        linear_ctx_compression: bool = True,
        use_film: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_dim_context_emb = n_dim_context_emb or 0
        # Use fraction of latent dimension as target context dim
        if n_ctx_frac is not None:
            n_context_compression = int(n_input * n_ctx_frac)
        self.use_film = use_film

        # Compress context embedding dimension to reduce its effect
        if self.n_dim_context_emb > 0 and n_context_compression is not None and n_context_compression > 0:
            n_hidden_ctx_compr = n_context_compression ** 2
            if linear_ctx_compression:
                self.ctx_compression = nn.Linear(self.n_dim_context_emb, n_context_compression)
            else:
                self.ctx_compression = nn.Sequential(
                    nn.Linear(self.n_dim_context_emb, n_hidden_ctx_compr),
                    nn.GELU(),
                    nn.Linear(n_hidden_ctx_compr, n_context_compression)
                )
            self.n_dim_context_emb = n_context_compression
        else:
            self.ctx_compression = None

        # Determine decoder input dimensions
        decoder_input_dim = n_input if use_film else n_input + self.n_dim_context_emb
        funnel_out_dim = n_output // 2

        if linear_decoder:
            self.px_decoder = FCLayers(
                n_in=decoder_input_dim,
                n_out=funnel_out_dim,
                n_cat_list=n_cat_list,
                dropout_rate=0,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                inverted=True,
                linear=True,
            )
        else:
            if use_funnel:
                fc_layer_class = FunnelFCLayers
                n_hidden = -1
            else:
                fc_layer_class = FCLayers
            # Create main px decoder
            self.px_decoder = fc_layer_class(
                n_in=decoder_input_dim,
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
        
        # Initialize film modulation of z
        if use_film and self.n_dim_context_emb > 0:
            self.film = FiLM(feature_dim=decoder_input_dim, context_dim=self.n_dim_context_emb)

        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        else:
            raise ValueError(f"Unknown scale_activation: {scale_activation}")

        self.px_scale_decoder = nn.Sequential(nn.Linear(funnel_out_dim, n_output), px_scale_activation)
        self.px_r_decoder = nn.Linear(funnel_out_dim, n_output)
        self.px_dropout_decoder = nn.Linear(funnel_out_dim, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int, context_emb: torch.Tensor | None = None):
        # Concatenate or modulate by context
        if self.n_dim_context_emb > 0 and context_emb is not None:
            # Use FiLM to combine the latent space with batch information
            if self.use_film:
                px = self.film(z, context_emb)
            # Use simple concatenation
            else:
                if self.ctx_compression is not None:
                    ctx_emb = self.ctx_compression(context_emb)
                else:
                    ctx_emb = context_emb
                z = torch.cat([z, ctx_emb], dim=-1)
            px = self.px_decoder(z, *cat_list)
        else:
            px = self.px_decoder(z, *cat_list)

        # Output heads
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
    

class LatentProjection(nn.Module):
    """
    Flexible projection MLP for mapping z to h (e.g. VAE latent → shared embedding space).
    Allows dynamic number of layers, hidden size, and activation choice.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation_fn: nn.Module = nn.GELU,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        layers = []
        in_dim = n_in

        # Build hidden layers dynamically
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, n_hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(n_hidden))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = n_hidden

        # Final layer to n_out
        layers.append(nn.Linear(in_dim, n_out))
        if use_layer_norm:
            layers.append(nn.LayerNorm(n_out))

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class ContextClassAligner(nn.Module):
    """
    Dual-embedding aligner for joint context-class modeling.
    Takes latent features h and aligns them with both context and class embeddings.
    Returns logits for both (for CE/CLIP) and projected latents for shared-space training.
    """

    def __init__(
        self,
        n_input: int,
        ctx_emb_dim: int,
        cls_emb_dim: int,
        n_shared: int = 128,
        n_hidden: int = 256,
        n_layers: int = 1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        temperature: float = 0.1,
        min_temperature: float = 0.05,
        max_temperature: float = 2.0,
        sigmoid_temp_scaling: bool = False,
        noise_sigma: float | None = 1e-6,
        use_learnable_temperature: bool = True,
        use_learnable_sigma: bool = True,
        use_film: bool = False,
        linear_ctx_proj: bool = True,
        linear_cls_proj: bool = False,
        unseen_buffer_prob: float = 0.1,
        return_projections: bool = True,
        use_z: bool = True,
        n_heads: int = 1,
        use_elementwise_combination: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.T = temperature
        self.T_min = min_temperature
        self.T_max = max_temperature
        self.sigmoid_temp_scaling = sigmoid_temp_scaling
        self.noise_sigma = noise_sigma
        self.return_projections = return_projections
        self.unseen_buffer_prob = unseen_buffer_prob
        self.n_heads = n_heads
        self.use_film = use_film
        self.use_learnable_temperature = use_learnable_temperature
        self.use_elementwise_combination = use_elementwise_combination
        # Joint c buffers
        self.register_buffer("c_joint_bank", None, persistent=False)
        self.register_buffer("n_ctx_cls", torch.zeros(2, dtype=torch.long), persistent=False)

        # --------- Temperature ---------
        if use_learnable_temperature:
            self._ctx_temperature = nn.Parameter(torch.tensor(0.0))
            self._cls_temperature = nn.Parameter(torch.tensor(0.0))
            self._joint_temperature = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("_ctx_temperature", torch.tensor(self.T))
            self.register_buffer("_cls_temperature", torch.tensor(self.T))
            self.register_buffer("_joint_temperature", torch.tensor(self.T))

        # --------- Sigma ---------
        if use_learnable_sigma and noise_sigma is not None:
            self.noise_sigma = nn.Parameter(torch.tensor(noise_sigma))

        # Set number of hidden to inp
        # --------- Latent projection(s) ---------
        # Use z as shared dimension
        if use_z and n_input == n_shared:
            self.latent_projection = nn.Identity()
        elif n_hidden is None and n_layers is None:
            # Simple linear projection
            self.latent_projection = nn.Linear(n_input, n_shared)
        else:
            # Add a small fc network
            self.latent_projection = FunnelFCLayers(
                n_input, n_shared, 
                n_hidden=n_hidden, 
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation_fn=nn.GELU,
            )

        # --------- External embedding projections ---------
        if linear_ctx_proj:
            self.ctx_projection = nn.Linear(ctx_emb_dim, n_shared)
        else:
            self.ctx_projection = FunnelFCLayers(
                ctx_emb_dim, n_shared, 
                n_hidden=n_hidden, 
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation_fn=nn.GELU,
            )
        # Skip class projection if its already in the correct dimension
        if cls_emb_dim == n_shared:
            self.cls_projection = nn.Identity()
        elif linear_cls_proj:
            self.cls_projection = nn.Linear(cls_emb_dim, n_shared)
        else:
            self.cls_projection = FunnelFCLayers(
                cls_emb_dim, n_shared, 
                n_hidden=n_hidden, 
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation_fn=nn.GELU,
            )

        # --------- FiLM ------------
        if use_film:
            self.film = FiLM(n_shared, context_dim=n_shared)

        # --------- Dropout ---------
        self.dropout = nn.Dropout(p=dropout_rate)
        # Cache
        self.alpha = None

    @property
    def ctx_temperature(self):
        if self.use_learnable_temperature:
            return self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self._ctx_temperature)
        else:
            return self.T
        
    @property
    def cls_temperature(self):
        if self.use_learnable_temperature:
            if self.sigmoid_temp_scaling:
                return self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self._cls_temperature)
            else:
                return self._cls_temperature.clamp(self.T_min, self.T_max)
        else:
            return self.T
        
    @property
    def joint_temperature(self):
        if self.use_learnable_temperature:
            if self.sigmoid_temp_scaling:
                return self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self._joint_temperature)
            else:
                return self._joint_temperature.clamp(self.T_min, self.T_max)
        else:
            return self.T
        
    def get_temp_reg_loss(self) -> torch.Tensor:
        if not self.use_learnable_temperature:
            return torch.tensor(0.0)
        m = (self.T_max - self.T_min)
        loss = (
            (self.ctx_temperature - m)**2 +
            (self.cls_temperature - m)**2 +
            (self.joint_temperature - m)**2
        )
        return loss.mean()
    
    def join_emb(self, ctx_emb: torch.Tensor, cls_emb: torch.Tensor, alpha: float = 0.9):
        # Set embedding weights for combination
        ctx_w = 1 - alpha
        cls_w = alpha
        # Normalize embeddings before combining
        b_ctx = F.normalize(ctx_emb * ctx_w, dim=-1)
        b_cls = F.normalize(cls_emb * cls_w, dim=-1)
        # Joint representation (full)
        joint = b_ctx + b_cls
        # Add elementwise combinations
        if self.use_elementwise_combination:
            joint = joint + (b_ctx * b_cls)
        # Return normalized joint embedding
        return F.normalize(joint, dim=-1)
   
    def forward(
        self,
        z: torch.Tensor,
        ctx_emb: torch.Tensor,
        ctx_idx: torch.Tensor,
        cls_emb: torch.Tensor,
        cls_idx: torch.Tensor,
        return_logits: bool = True,
        T: torch.Tensor | None = None,
        ctrl_idx: int | None = None,
        alpha: float = 0.8,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        z : Tensor, shape (B, latent_dim)
        ctx_idx : Tensor, shape (B,)
        ctx_emb : Tensor, shape (N_ctx, ctx_emb_dim)
        cls_idx : Tensor, shape (B,)
        cls_emb : Tensor, shape (N_cls, cls_emb_dim)
        """
        # Move embeddings to correct device
        ctx_emb, cls_emb = ctx_emb.to(z.device), cls_emb.to(z.device)
        
        # Cache alpha
        self.alpha = alpha

        # ----- Latent projection -----
        h = self.latent_projection(z)

        if self.dropout_rate > 0:
            h = self.dropout(h)
            
        # Optional noise regularization
        if self.noise_sigma is not None and self.noise_sigma > 0:
            noise_ctx = F.normalize(torch.randn_like(ctx_emb), dim=-1)
            noise_cls = F.normalize(torch.randn_like(cls_emb), dim=-1)
            ctx_emb = ctx_emb + self.noise_sigma * noise_ctx
            cls_emb = cls_emb + self.noise_sigma * noise_cls

        # ----- External projections -----
        c_ctx = self.ctx_projection(ctx_emb)        # (N_ctx, D)
        c_cls = self.cls_projection(cls_emb)        # (N_cls, D)
            
        # Set control index to -inf if given
        if ctrl_idx is not None:
            c_cls[torch.as_tensor(ctrl_idx, device=c_cls.device, dtype=torch.long)] = -float('inf')
        
        # Get batch-specific embeddings
        b_ctx = c_ctx[ctx_idx.flatten()] if ctx_idx is not None else c_ctx
        b_cls = c_cls[cls_idx.flatten()] if cls_idx is not None else c_cls
        # Get batch-specific joint embeddingd
        b_joint_norm = self.join_emb(
            ctx_emb=b_ctx, 
            cls_emb=b_cls, 
            alpha=alpha
        )
        
        # Normalize
        h_norm = F.normalize(h, dim=-1)
        c_ctx_norm = F.normalize(c_ctx, dim=-1)
        c_cls_norm = F.normalize(c_cls, dim=-1)

        # Outputs
        outputs = {
            MODULE_KEYS.Z_SHARED_KEY: h_norm,
            MODULE_KEYS.CTX_PROJ_KEY: c_ctx_norm,
            MODULE_KEYS.CLS_PROJ_KEY: c_cls_norm,
            MODULE_KEYS.JOINT_PROJ_KEY: b_joint_norm,
        }
        # Calculate logits here and return
        if return_logits:
            outputs.update({
                MODULE_KEYS.CTX_LOGITS_KEY: self.get_ctx_logits(h_norm, c_ctx_norm, T=T),
                MODULE_KEYS.CLS_LOGITS_KEY: self.get_cls_logits(h_norm, c_cls_norm, T=T),
                MODULE_KEYS.JOINT_LOGITS_KEY: self.get_joint_logits(h_norm, b_joint_norm, T=T),
            })
        return outputs
    
    def get_ctx_logits(self, h_norm: torch.Tensor, c_ctx_norm: torch.Tensor, T: torch.Tensor | None) -> torch.Tensor:
        # Choose interal T or given
        _T = T if T is not None else self.ctx_temperature
        # Alignment via cosine similarity
        return (h_norm @ c_ctx_norm.T) / _T     # (B, N_ctx)
    
    def get_cls_logits(self, h_norm: torch.Tensor, c_cls_norm: torch.Tensor, T: torch.Tensor | None) -> torch.Tensor:
        # Choose interal T or given
        _T = T if T is not None else self.cls_temperature
        # Alignment via cosine similarity
        return (h_norm @ c_cls_norm.T) / _T     # (B, N_cls)
    
    def get_joint_logits(self, h_norm: torch.Tensor, c_joint_norm: torch.Tensor, T: torch.Tensor | None) -> torch.Tensor:
        # Choose interal T or given
        _T = T if T is not None else self.joint_temperature
        # Alignment via cosine similarity
        return (h_norm @ c_joint_norm.T) / _T     # (B, B)
        
    @torch.no_grad()
    def precompute_joint_bank(
        self,
        ctx_emb: torch.Tensor,
        cls_emb: torch.Tensor,
        device: str | torch.device | None = None,
        alpha: float = 0.8,
        cache: bool = True,
    ):
        """
        Precompute all possible (context, class) joint embeddings.

        Args:
            ctx_emb: Context embeddings, shape (N_ctx, D_ctx)
            cls_emb: Class embeddings, shape (N_cls, D_cls)
            device: Optional target device (defaults to model device)
            alpha: Weight controlling context vs. class influence
            cache: Whether to store result in self.c_joint_bank for reuse
        """
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Precomputing joint embeddings (contexts={ctx_emb.size(0)}, classes={cls_emb.size(0)}).")

        # Resolve device
        device = device or next(self.parameters()).device

        # Move to correct device and normalize
        c_ctx = F.normalize(self.ctx_projection(ctx_emb.to(device)) * (1 - alpha), dim=-1)
        c_cls = F.normalize(self.cls_projection(cls_emb.to(device)) * alpha, dim=-1)

        # Compute joint embeddings using broadcasting
        # (N_ctx, 1, D) + (1, N_cls, D) + elementwise interaction
        joint = c_ctx.unsqueeze(1) + c_cls.unsqueeze(0) + (c_ctx.unsqueeze(1) * c_cls.unsqueeze(0))
        joint = F.normalize(joint.view(-1, joint.size(-1)), dim=-1)

        if cache:
            self.c_joint_bank = joint.detach()
            self.n_ctx_cls = torch.tensor([c_ctx.size(0), c_cls.size(0)], device=device)
            log.info(f"Cached joint bank with {joint.size(0):,} entries of dim {joint.size(-1)}.")
            return self.c_joint_bank
        else:
            return joint

    
    @torch.no_grad()
    def classify(
        self,
        z: torch.Tensor,
        ctx_emb: torch.Tensor,
        cls_emb: torch.Tensor,
        alpha: float | None = 0.8,
        T: float | None = None,
        use_softmax: bool = False,
        cache: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Compute cosine similarities between query latent(s) and all joint embeddings
        for inference-time classification.

        Parameters
        ----------
        z : Tensor, (B, latent_dim)
            Query latent(s) to classify.
        ctx_emb : Tensor, (N_ctx, ctx_emb_dim)
            External context embeddings.
        cls_emb : Tensor, (N_cls, cls_emb_dim)
            External class embeddings.
        T : Optional[float]
            If given, overrides learned temperature.

        Returns
        -------
        dict[str, Tensor]
        """
        # Set to latest alpha if None else use given
        alpha = alpha if alpha is not None else self.alpha
        # ---- Project ----
        h = F.normalize(self.latent_projection(z), dim=-1)
        # Project to shared space
        c_ctx = F.normalize(self.ctx_projection(ctx_emb) * (1-alpha), dim=-1)
        c_cls = F.normalize(self.cls_projection(cls_emb) * alpha, dim=-1)
        
        # ---- Compute all possible joint embeddings ----
        if not self.training and cache:
            # Cache embeddings if possible
            diff_n = (self.n_ctx_cls != torch.tensor([ctx_emb.size(0), cls_emb.size(0)], device=self.n_ctx_cls.device)).any()
            if not hasattr(self, 'c_joint_bank') or diff_n:
                # Re-compute embeddings
                self.precompute_joint_bank(ctx_emb, cls_emb, alpha=alpha)
            # Get embeddings
            c_joint = self.c_joint_bank
        else:
            # Compute joint embeddings using broadcasting
            # (N_ctx, 1, D) + (1, N_cls, D) 
            joint = c_ctx.unsqueeze(1) + c_cls.unsqueeze(0)
            # + elementwise interaction
            if self.use_elementwise_combination:
                joint = joint + (c_ctx.unsqueeze(1) * c_cls.unsqueeze(0))
            # Normalize joint embedding
            c_joint = F.normalize(joint.view(-1, joint.size(-1)), dim=-1)
        
        # ---- Similarity scores ----
        T = T if T is not None else self.joint_temperature
        logits = (h @ c_joint.T) / T  # (B, N_ctx * N_cls)
        # --- Compute joint softmax ---
        logits = F.softmax(logits, dim=-1) if use_softmax else logits # (B, N_ctx * N_cls)
        # --- Reshape into 3D (B, N_ctx, N_cls) ---
        p_joint = logits.view(z.size(0), ctx_emb.size(0), cls_emb.size(0))
        
        # --- Marginalize ---
        if use_softmax:
            ctx_logits = p_joint.sum(dim=-1)   # (B, N_ctx)
            cls_logits = p_joint.sum(dim=-2)   # (B, N_cls)
        else:
            ctx_logits = p_joint.mean(dim=-1)   # (B, N_ctx)
            cls_logits = p_joint.mean(dim=-2)   # (B, N_cls)
        # ---- Return result object ----
        return {
            MODULE_KEYS.Z_SHARED_KEY: h,
            MODULE_KEYS.JOINT_LOGITS_KEY: logits,
            MODULE_KEYS.CTX_LOGITS_KEY: ctx_logits,
            MODULE_KEYS.CLS_LOGITS_KEY: cls_logits,
            MODULE_KEYS.CTX_PROJ_KEY: c_ctx,
            MODULE_KEYS.CLS_PROJ_KEY: c_cls,
            MODULE_KEYS.JOINT_PROJ_KEY: c_joint,
        }
        
        
class ClassEmbedding(nn.Module):
    """
    Provides class embeddings from either:
      1) A pre-trained embedding matrix (cls_emb)
      2) A transformer encoder applied to class_texts (dict)

    Forward() returns embeddings only for requested class indices.
    """

    def __init__(
        self,
        pretrained_emb: torch.Tensor | None = None,
        class_texts: dict | None = None,
        n_output: int | None = None,
        transformer_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 64,
        freeze_encoder: bool = False,
        train_last_n_layers: int = 1,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_dropout: int = 0.1,
        ignore_texts: bool = False,
    ):
        super().__init__()

        self.device = device
        self.class_texts = class_texts
        self.max_length = max_length
        self.batch_size = batch_size

        # -----------------------------
        # Case 1: Transformer text encoder
        # -----------------------------
        if class_texts is not None and not ignore_texts:
            # Load tokenizer + encoder
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
            self.encoder = AutoModel.from_pretrained(transformer_name)

            self.class_names = list(class_texts.keys())
            self._num_classes = len(self.class_names)
            self._emb_dim = self.encoder.config.hidden_size
            self.embedding = None

            # --- Freeze or partially freeze encoder ---
            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            
            # Use LoRA to reduce trainable parameters
            if use_lora:
                from peft import get_peft_model, LoraConfig, TaskType
                # Apply lora
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_rank,
                    lora_alpha=int(lora_rank*2),
                    lora_dropout=lora_dropout,
                    target_modules=["query", "value", "key"],
                    bias="none",
                    inference_mode=False,
                )
                # Update encoder
                self.encoder = get_peft_model(self.encoder, lora_config)
            else:
                # Unfreeze last n layers
                if train_last_n_layers > 0:
                    for layer in self.encoder.encoder.layer[-train_last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True
            
            # Always make pool trainable
            if hasattr(self.encoder, "pooler"):
                for p in self.encoder.pooler.parameters():
                    p.requires_grad = True

            # Disable parallel toeknization
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # Pre-tokenize ALL class texts
            self.pretokenized_inputs = []
            for name in self.class_names:
                text = class_texts[name]
                tokens = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                # Keep pretokenized CPU tensors for safety
                self.pretokenized_inputs.append(tokens)

        # -----------------------------
        # Case 2: Pre-trained embedding
        # -----------------------------
        else:
            assert pretrained_emb is not None, \
                "Provide either pretrained_emb or class_texts."

            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            self.encoder = None
            self.tokenizer = None

            self._num_classes, self._emb_dim = pretrained_emb.shape

        # Add final output adapter if n output is not None
        if n_output is not None:
            self.adapter = nn.Linear(self._emb_dim, n_output)
            self._emb_dim = n_output
        else:
            self.adapter = nn.Identity()

        # Move module to device
        self.to(device)

    # -------------------------------------------------------
    @property
    def shape(self):
        return (self._num_classes, self._emb_dim)

    # -------------------------------------------------------
    def forward(self, class_indices: torch.Tensor | list | None = None):
        """
        Compute class embeddings.

        Args:
            class_indices: Optional list/tensor of class indices to return.
                           If None → return all class embeddings.

        Returns:
            Tensor of shape (K, emb_dim)
        """
        # ---------------------------------------------------------
        # Case 1: Pretrained embedding → trivial indexing
        # ---------------------------------------------------------
        if self.embedding is not None:
            if class_indices is None:
                embeddings = self.embedding.weight
            else:
                embeddings = self.embedding.weight[class_indices]
            # Add adapter projection if given
            return self.adapter(embeddings)

        # ---------------------------------------------------------
        # Case 2: Transformer text embeddings (pretokenized)
        # ---------------------------------------------------------

        # If subset not provided → use all
        if class_indices is None:
            indices = list(range(self._num_classes))
        else:
            if isinstance(class_indices, torch.Tensor):
                class_indices = class_indices.tolist()
            indices = class_indices

        # Select pre-tokenized inputs
        selected_tokens = [self.pretokenized_inputs[i] for i in indices]

        all_embeddings = []

        # Process in small chunks (tokens already on CPU)
        for i in range(0, len(indices), self.batch_size):
            batch_tokens = selected_tokens[i : i + self.batch_size]

            # Merge token dicts into a batch
            merged = {
                key: torch.cat([t[key] for t in batch_tokens], dim=0).to(self.device)
                for key in batch_tokens[0]
            }
            # Forward pass through encoder
            out = self.encoder(**merged)
            cls = out.last_hidden_state[:, 0]

            all_embeddings.append(cls)

        # Collect batched text embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        # Add adaptor
        embeddings = self.adapter(embeddings)
        return embeddings


class EmbeddingAligner(nn.Module):
    """
    Classifier where latent attends to external class embeddings using dot product or cosine similarity. 
    Returns classifier style logits to be used for clip or kl loss.
    """

    def __init__(
        self,
        n_input: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        logits: bool = True,
        funnel: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation_fn: nn.Module = nn.GELU,
        class_embed_dim: int = 128,
        shared_projection_dim: int | None = None,
        skip_projection: bool = False,
        return_latents: bool = True,
        temperature: float = 0.1,
        use_learnable_temperature: bool = True,
        mode: Literal["latent_to_class", "class_to_latent", "shared"] = "latent_to_class",
        use_cross_attention: bool = True,
        n_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        self.class_embed_dim = class_embed_dim
        self.dropout_rate = dropout_rate
        self.funnel = funnel
        self.return_latents = return_latents
        self.mode = mode
        self._temperature = temperature
        self.use_learnable_temperature = use_learnable_temperature
        self.use_cross_attention = use_cross_attention
        # Create a learnable temperature scaling
        if use_learnable_temperature:
            self._temperature = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # ---- Helper for building projections ---- #
        def build_projection(n_in: int, n_out: int) -> nn.Module:
            """Creates a projection block based on n_hidden/n_layers settings."""
            if skip_projection and n_in == n_out:
                return nn.Identity()
            # Add funnel layer projections
            if n_layers > 0 and n_hidden > 0:
                layer_cls = FunnelFCLayers if funnel else FCLayers
                return layer_cls(
                    n_in=n_in,
                    n_out=n_out,
                    n_hidden=n_hidden,
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
            self.n_output = class_embed_dim
            self.latent_projection = build_projection(n_input, class_embed_dim)
            self.class_projection = nn.Identity()
        # Class embedding space --> latent space projection
        elif mode == "class_to_latent":
            self.n_output = n_input
            self.latent_projection = nn.Identity()
            self.class_projection = build_projection(class_embed_dim, n_input)
        # Latent --> shared dim <-- Class embedding space
        elif mode == "shared":
            if shared_projection_dim is None:
                shared_projection_dim = min(n_input, class_embed_dim)
            self.latent_projection = build_projection(n_input, shared_projection_dim)
            self.class_projection = build_projection(class_embed_dim, shared_projection_dim)
            self.n_output = shared_projection_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- Dropout ---- #
        self.dropout = nn.Dropout(p=dropout_rate)

    @property
    def temperature(self) -> torch.Tensor:
        if self.use_learnable_temperature:
            # Calculate logit scale
            return 1.0 / self._temperature.clamp(-1, 4.6).exp()
        else:
            return self._temperature
    
    def forward(self, x: torch.Tensor, class_embeds: torch.Tensor, noise_sigma: float | None = None):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, latent_dim)
        class_embeds : Optional[Tensor], shape (n_labels, embed_dim)
        labels : Optional[Tensor], shape (batch,)
        """
        # Get class embeddings
        class_embeds = class_embeds.to(x.device)

        # Apply projections
        z = self.latent_projection(x)               # (batch, d)
        c = self.class_projection(class_embeds)     # (n_labels, d)
        # Add some noise to class embedding to avoid having a fixed point
        if noise_sigma is not None and noise_sigma > 0:
            c = c + noise_sigma * F.normalize((torch.rand_like(c) * 2 - 1), dim=-1)
        if self.dropout_rate > 0:
            z = self.dropout(z)
        # Apply normalization to projections
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(c, dim=-1)

        # ---- OPTION 1: Cross-Attention path ---- #
        if self.use_cross_attention:
            B, D = z_norm.shape
            # Reshape for MultiheadAttention: [B, 1, D] attends to [1, C, D]
            q = z.unsqueeze(1) / self.temperature         # queries = latents
            # [B, C, D]
            k = c_norm.unsqueeze(0).repeat(B, 1, 1)       # keys = class embeddings
            v = c.unsqueeze(0).repeat(B, 1, 1)
            # Apply attention
            scores = (q @ k.transpose(1, 2)) / math.sqrt(D)
            attn_weights = torch.softmax(scores, dim=-1)
            z_attn = attn_weights @ v
            # Apply cross attention
            #z_attn, attn_weights = self.cross_attn(q, k, v)  # [B,1,D], [B,1,C]
            z_proj = z_attn.squeeze(1)
            logits = attn_weights.squeeze(1)  # attention weights as logits (before softmax)
            #logits = logits / self.temperature      # apply temperature scaling

        # ---- OPTION 2: Standard cosine similarity ---- #
        else:
            logits = torch.matmul(z_norm, c_norm.T).clamp(-1, 1)
            logits = logits / self.temperature
            z_proj = z_norm

        # Optionally return softmax
        logits = logits if self.logits else F.softmax(logits, dim=-1)
        # Return logits only or tuple of logits and latent spaces
        return (logits, z_proj, c_norm) if self.return_latents else logits
    

class ArcClassifier(nn.Module):
    """
    ArcFace-style classifier for angular-margin learning.

    Parameters
    ----------
    n_input : int
        Input feature dimension (e.g., encoder output).
    n_hidden : int
        Hidden dimension for optional intermediate layer (set 0 to disable).
    n_labels : int
        Number of class labels.
    n_layers : int
        Number of hidden layers before ArcFace head.
    dropout_rate : float
        Dropout probability.
    margin : float
        Additive angular margin (m).
    scale : float
        Feature scale (s).
    use_batch_norm : bool
        Apply batch normalization in hidden layers.
    use_layer_norm : bool
        Apply layer normalization in hidden layers.
    activation_fn : nn.Module
        Activation function.
    temperature: float
        Temperature scaling for logits
    use_learnable_temperature: bool
        Learn a temperature scaling from the static base temperature
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        margin: float = 0.3,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        temperature: float = 1.0,
        use_learnable_temperature: bool = True,
        return_latents: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.margin = margin
        self.n_labels = n_labels
        self.use_learnable_temperature = use_learnable_temperature
        self.return_latents = return_latents

        # Create a learnable temperature scaling
        if use_learnable_temperature:
            self._temperature = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # Fall back to static default temperature
        else:
            self._temperature = temperature

        # Optional hidden layers
        if n_hidden > 0 and n_layers > 0:
            self.backbone = FCLayers(
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
            feat_dim = n_hidden
        else:
            self.backbone = nn.Identity()
            feat_dim = n_input

        # ArcFace class weight matrix
        self.W = nn.Parameter(torch.randn(n_labels, feat_dim))
        nn.init.xavier_uniform_(self.W)

    @property
    def temperature(self) -> torch.Tensor:
        if self.use_learnable_temperature:
            # Calculate logit scale
            return 1.0 / self._temperature.clamp(-1, 4.6).exp()
        else:
            return self._temperature

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Forward pass.
        If y is provided, apply the angular margin and scaling (training mode).
        If y is None, return raw cosine logits (inference mode).
        """
        x = self.backbone(x)
        x = F.normalize(x, dim=-1)
        W = F.normalize(self.W, dim=-1)

        # Cosine similarity between features and class weights
        cosine = torch.matmul(x, W.T).clamp(-1, 1)

        if labels is not None:
            # Compute θ and apply margin to true classes
            theta = torch.acos(cosine)
            target_logits = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, num_classes=self.n_labels).bool()
            # Apply margin only to true class positions
            logits = torch.where(one_hot, target_logits, cosine)
        else:
            logits = cosine
        # Scale logits by temperature
        logits = logits / self.temperature
        return logits, x, W if self.return_latents else logits


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
        logits: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        temperature: float = 1.0,
        use_learnable_temperature: bool = True,
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
        # Number of output dimensions
        self.n_output = n_labels

        # Create a learnable temperature scaling
        self.use_learnable_temperature = use_learnable_temperature
        if use_learnable_temperature:
            self._temperature = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # Fall back to static default temperature
        else:
            self._temperature = temperature

    @property
    def temperature(self) -> torch.Tensor:
        if self.use_learnable_temperature:
            # Calculate logit scale
            return 1.0 / self._temperature.clamp(-1, 4.6).exp()
        else:
            return self._temperature

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward computation. Ignore additional parameters passed to other classes."""
        return self.classifier(x) / self.temperature

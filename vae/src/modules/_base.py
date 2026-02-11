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

from src.modules._blocks import ProjectionBlock, FcBlock

import logging


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


class FunnelFCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = -1,
        n_cat_list: Optional[Iterable[int]] = None,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_activation: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.GELU,
        **kwargs,
    ):
        super().__init__()

        self.inject_covariates = inject_covariates
        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm
        self.kwargs = kwargs

        # Init modules
        self.layer_norm = nn.LayerNorm(n_out)

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
        
        # Init layers
        self.layers = nn.ModuleList()

        for i in range(n_layers + e):
            in_dim = hidden_dims[i] + self.cat_dim * self.inject_into_layer(i)
            out_dim = hidden_dims[i + 1]
            # Create block for each layer
            block = ProjectionBlock(
                n_input=in_dim, 
                n_output=out_dim, 
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=use_activation,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
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


class MixedLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 128,
        n_cat_list: Optional[Iterable[int]] = None,
        n_encoder_layers: int = 2,
        n_layers: int = 3,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        use_attention: bool = False,
        bias: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.GELU,
        noise_std: float = 0.0,
        linear_encoder: bool = True,
        use_residuals: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.inject_covariates = inject_covariates
        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.n_hidden = n_hidden
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
                n_layers=n_encoder_layers,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
            )
        
        # Introduce multiple attention layers of same dimension
        self.layers = nn.ModuleList()
        for _ in np.arange(n_layers):
            # Create block for each layer
            block = FcBlock(
                n_input=n_out,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
                bias=bias,
                noise_std=noise_std,
                use_residuals=use_residuals,
                **kwargs
            )
            self.layers.append(block)
    
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor, **kwargs) -> torch.Tensor:
        # First linear projection
        x = self.encoder(x)
        
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
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.GELU,
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
    

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer (identity-centered)."""
    def __init__(self, feature_dim: int, context_dim: int, weight: float = 0.1, init_zero: bool = True):
        super().__init__()
        self.gamma = nn.Linear(context_dim, feature_dim)
        self.beta = nn.Linear(context_dim, feature_dim)
        self.weight = weight

        # Optional: init close to identity
        if init_zero:
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    def forward(self, z: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(context_emb) * self.weight
        beta = self.beta(context_emb) * self.weight
        return (1.0 + gamma) * z + beta


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
        drop_prob: float = 0.0,
        noise_sigma: float = 0.0,
        encoder_type: Literal["funnel", "fc", "transformer"] = "funnel",
        context_integration_method : Literal["concat", "film", "attention"] | None = None,
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
            self.fclayers_class = MixedLayers
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
        self.noise_sigma = noise_sigma
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
    
    def forward(self, x: torch.Tensor, *cat_list: int, g: torch.Tensor | None = None, ctx_label: torch.Tensor | None = None, context_emb: torch.Tensor | None = None):
        # Optional feature masking
        if self.training and self.use_feature_mask and self.drop_prob > 0:
            mask = torch.bernoulli(torch.full((self.n_input,), 1 - self.drop_prob, device=x.device)).view(1, -1)
            x = x * mask.expand(x.shape[0], -1)
        # Optional noise injection
        if self.training and self.noise_sigma:
            noise = torch.exp(
                torch.randn_like(x) * self.noise_sigma
            )
            x = x * noise

        # Concatenate or modulate by context
        ctx_logits, b_ctx_emb = None, None
        if self.n_dim_context_emb > 0 and context_emb is not None and self.use_context_inference:
            # Add context integration to batch
            if self.ctx_integration is not None:
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
                
                # FiLM or attention
                x = self.ctx_integration(x, b_ctx_emb)
            else:
                b_ctx_emb = context_emb[ctx_label.flatten()]
                # Simple concatenation
                x = torch.cat([x, b_ctx_emb], dim=-1)
            
            # Do a second encoder forward pass
            q = self.encoder(x, *cat_list)
        else:
            # Do a simple forward pass with just x and no additional information
            q = self.encoder(x, *cat_list)
        
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
        
        
class GeneEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_dim_gene_emb: int | None = None,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        use_token_dropout: bool = False,
        film_weight: float = 1.0,
        sem_dropout_rate: float = 0.1,
        T: float | None = 0.1,
        topk_features: int = 64,
        **kwargs,
    ):
        super().__init__()
        # Save class params
        self.use_token_dropout = use_token_dropout

        # Gene token dropout
        if self.use_token_dropout:
            self.token_dropout = nn.Dropout(dropout_rate)
            
        # Add gene embedding information as semantic attention
        self.use_gene_emb = film_weight > 0 and n_dim_gene_emb is not None
        if self.use_gene_emb:
            # Set information bottleneck
            h_low = n_hidden // 2
            # Set expansion module
            self.x_expand = nn.Sequential(
                nn.Linear(h_low, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            # Create film
            self.film = FiLM(h_low, h_low, weight=film_weight, init_zero=False)
            # Add gene adapter
            self.gene_adapter = nn.Linear(n_dim_gene_emb, h_low, bias=False)
        else:
            self.film = None
            h_low = n_hidden

        # Main X encoder
        x_encoder_layers = [
                    nn.Linear(n_input, h_low),
                    nn.LayerNorm(h_low),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
        for _ in range(n_layers-1):
            x_encoder_layers.extend([
                    nn.Linear(h_low, h_low),
                    nn.LayerNorm(h_low),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ])
        # Add final projection layer
        self.x_encoder = nn.Sequential(*x_encoder_layers)
        
        # Latent heads
        self.z_mu = nn.Linear(n_hidden, n_output)
        self.z_var = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(sem_dropout_rate)
        # Activations
        self.var_eps = var_eps
        self.var_activation = torch.exp if var_activation is None else var_activation
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else nn.Identity()
        self.T = math.sqrt(n_hidden) if T is None else T
        self.topk_features = topk_features
        
    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None, **kwargs):
        # Encode x
        h_x = self.x_encoder(x)
        # Add cell-specific semantic information
        if self.use_gene_emb and g is not None:
            # Gene embedding
            e = self.gene_adapter(g.to(x.device))
            e = F.normalize(e, dim=-1)
            h_x = F.normalize(h_x, dim=-1)
            # Use top-k features
            scores = (h_x @ e.T) / self.T          # (B, G)
            topk = torch.topk(scores, k=self.topk_features, dim=-1)
            mask = torch.zeros_like(scores).scatter_(1, topk.indices, 1.0)
            # Get masked attention
            attn = torch.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1)
            ctx = attn @ e

            # FiLM modulation
            h_cell = self.film(h_x, ctx)
            # Expand back to n_hidden
            h_cell = self.x_expand(h_cell)
        else:
            h_cell = h_x

        return h_cell
        

class SplitEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_dim_gene_emb: int | None = None,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        use_funnel_zlocal: bool = True,
        **kwargs,
    ):
        super().__init__()

        # -------------------------
        # Global path (reconstruction)
        # -------------------------
        self.global_encoder = FunnelFCLayers(
            n_in=n_input,
            n_out=n_output,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            dropout_rate=dropout_rate,
        )

        self.global_mu = nn.Linear(n_output, n_output)
        self.global_var = nn.Linear(n_output, n_output)

        # -------------------------
        # Local path (semantic / CLIP)
        # -------------------------
        if use_funnel_zlocal:
            self.local_encoder = FunnelFCLayers(
            n_in=n_input,
            n_out=n_output,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            dropout_rate=dropout_rate,
        )
        else:
            self.local_encoder = GeneEmbeddingEncoder(
                n_input=n_input,
                n_output=n_output,
                n_hidden=n_output,
                n_dim_gene_emb=n_dim_gene_emb,
                dropout_rate=dropout_rate,
            )
        self.local_mu = nn.Linear(n_output, n_output)
        self.local_var = nn.Linear(n_output, n_output)
        
        # Var activations
        self.var_activation = torch.exp if var_activation is None else var_activation
        self.var_eps = var_eps
        
    def decorrelation_reg(self, z_g: torch.Tensor, z_l: torch.Tensor, eps: float = 1e-6):
        """
        Penalize linear correlation between global and local latents.
        Scale-invariant version.
        """
        # Center
        z_g = z_g - z_g.mean(dim=0, keepdim=True)
        z_l = z_l - z_l.mean(dim=0, keepdim=True)

        # Normalize per-dimension variance
        z_g = z_g / (z_g.std(dim=0, keepdim=True) + eps)
        z_l = z_l / (z_l.std(dim=0, keepdim=True) + eps)

        # Cross-correlation
        C = (z_g.T @ z_l) / z_g.size(0)

        return C ** 2


    def forward(self, x: torch.Tensor, *cat_list, g: torch.Tensor, **kwargs):
        # ---- global (decoder) ----
        h_g = self.global_encoder(x, *cat_list)
        mu_g = self.global_mu(h_g)
        var_g = self.var_activation(self.global_var(h_g)) + self.var_eps
        qzg = Normal(mu_g, var_g.sqrt())
        z_g = qzg.rsample()

        # ---- local (CLIP) ----
        local_out = self.local_encoder(x, g)
        # Latent heads
        q_ml = self.local_mu(local_out)
        q_vl = self.var_activation(self.local_var(local_out)) + self.var_eps
        qzl = Normal(q_ml, q_vl.sqrt())
        zl = qzl.rsample()

        return {
            MODULE_KEYS.Z_KEY: z_g,
            MODULE_KEYS.QZ_KEY: qzg,
            MODULE_KEYS.QZM_KEY: mu_g,
            MODULE_KEYS.QZV_KEY: var_g,
            "zl": zl,
            "qzl": qzl,
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
        n_context_compression: int | None = None,
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
            self.latent_projection = nn.Linear(n_input, n_shared, bias=False)
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
            self.ctx_projection = nn.Linear(ctx_emb_dim, n_shared, bias=False)
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
            self.cls_projection = nn.Linear(cls_emb_dim, n_shared, bias=False)
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
        freeze: bool = False,
        train_last_n_layers: int = 1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_dropout: int = 0.1,
        ignore_texts: bool = False,
        n_prototypes: int = 4,
    ):
        super().__init__()

        self.device = device
        self.class_texts = class_texts
        self.max_length = max_length
        self.batch_size = batch_size
        self.freeze = freeze

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

            # --- Freeze encoder ---
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            
            # Use LoRA to reduce trainable parameters
            if not freeze and use_lora:
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
                logging.info(f'Applied LoRA to pre-trained text encoder.')
            elif not freeze and train_last_n_layers > 0:
                # Unfreeze last n layers
                for layer in self.encoder.encoder.layer[-train_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                logging.info(f'Unfroze last {train_last_n_layers} layers of pre-trained text encoder.')
            # Always make pool trainable
            if hasattr(self.encoder, "pooler") and not freeze:
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
            self.adapter = nn.Linear(self._emb_dim, n_output, bias=False)
            self.use_adapter = True
            # Initialize small weights
            with torch.no_grad():
                nn.init.zeros_(self.adapter.weight)
                nn.init.zeros_(self.adapter.bias)

                d = min(self._emb_dim, n_output)
                self.adapter.weight[:d, :d].copy_(torch.eye(d))
            self._emb_dim = n_output
        else:
            self.adapter = nn.Identity()
            self.use_adapter = False

        # Use multi-prototypes (optional)
        if n_prototypes > 1:
            C, D = self._num_classes, self._emb_dim
            self.delta = nn.Parameter(torch.randn(C, n_prototypes, D) * 0.01)
            self.use_prototypes = True
            self.n_prototypes = n_prototypes
        else:
            self.use_prototypes = False
            
        # Move module to device
        self.to(device)
        
    def orthogonal_reg(self):
        if not self.use_adapter:
            return torch.tensor(0.0)
        W = self.adapter.weight
        I = torch.eye(W.size(1), device=W.device)
        return ((W.T @ W - I)**2).mean()

    # -------------------------------------------------------
    @property
    def shape(self):
        if not self.use_prototypes:
            return (self._num_classes, self._emb_dim)
        else:
            return (self._num_classes, self.n_prototypes, self._emb_dim)
    
    def prototype_div_reg(self, emb: torch.Tensor):
        """
        Diversity regularizer for multi-prototype embeddings.
        Encourages prototypes within each class to be orthogonal.

        emb: (C, M, D)
        """
        if not self.use_prototypes or emb.ndim != 3:
            return torch.tensor(0.0, device=emb.device)

        # Normalize to be safe
        emb = F.normalize(emb, dim=-1)

        C, M, D = emb.shape

        # Compute Gram matrices per class: (C, M, M)
        gram = torch.einsum("cmd,cnd->cmn", emb, emb)

        # Target is identity (prototypes should be orthogonal)
        eye = torch.eye(M, device=emb.device).unsqueeze(0)  # (1, M, M)

        # Frobenius norm of deviation from identity
        loss = ((gram - eye) ** 2).mean()

        return loss

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
            embeddings = self.adapter(embeddings)
            if self.use_prototypes:
                embeddings = embeddings[:, None, :] + self.delta
            return embeddings

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
        # Add embeddings to cache if model is frozen and not already replaced with a pretrained embedding
        if self.freeze and self.embedding is None:
            logging.info('Registered frozen text embeddings.')
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        # Pass through adapter (optional)
        embeddings = self.adapter(embeddings)
        # Create multi-prototypes (optional)
        if self.use_prototypes:
            embeddings = embeddings[:, None, :] + self.delta
            
        return embeddings
    

class HierarchicalAligner(nn.Module):
    """
    Aligner class that takes in an encoded latent space and alignes it to class text representations.

    Args:
    """
    
    def __init__(
        self,
        n_latent: int,
        n_emb: int,
        n_small: int = 32,
        n_medium: int = 64,
        n_class: int = 128,
        bias: bool = False,
        T: float = 0.1,
        small_T: float = 0.7,
        med_T: float = 0.3,
        noise: float = 0.05,
        dropout_rate: float = 0.1,
        linear_z_proj: bool = False,
        n_layers: int = 2,
        ff_mult: int = 4,
        n_hidden: int | None = None,
        n_small_proxies: int | None = None,
        n_med_proxies: int | None = None,
        use_learnable_proxies: bool = True,
        use_classifiers: bool = True,
        **kwargs
    ):
        super().__init__()
        # Define input dimensions
        self.n_latent = n_latent
        self.n_emb = n_emb
        self.T = T
        self.small_T = small_T
        self.med_T = med_T
        self.noise = noise
        self.dropout_rate = dropout_rate
        # Define hierachical dimensions
        self.n_small = n_small
        self.n_medium = n_medium
        self.n_class = n_class
        
        # Build RNA projections
        if linear_z_proj:
            self.z2s = nn.Linear(self.n_latent, self.n_small, bias=bias)
            self.z2m = nn.Linear(self.n_latent, self.n_medium, bias=bias)
            self.z2c = nn.Linear(self.n_latent, self.n_class, bias=bias)
        else:
            # Create non-linear FClayer projections
            n_hidden = n_hidden if n_hidden is not None else int(self.n_latent*ff_mult)
            if n_hidden == -1:
                n_hidden = n_latent
            self.z2s = FCLayers(self.n_latent, self.n_small, n_hidden=n_hidden, n_layers=n_layers)
            self.z2m = FCLayers(self.n_latent, self.n_medium, n_hidden=n_hidden, n_layers=n_layers)
            self.z2c = FCLayers(self.n_latent, self.n_class, n_hidden=n_hidden, n_layers=n_layers)
        
        # Build individual text projections
        self.t2s = nn.Linear(self.n_emb, self.n_small, bias=bias)
        self.t2m = nn.Linear(self.n_emb, self.n_medium, bias=bias)
        self.t2c = nn.Linear(self.n_emb, self.n_class, bias=bias)
        
        # Create learnable intermediate level proxies
        if n_small_proxies is not None and n_med_proxies is not None and use_learnable_proxies:
            self.small_proxies = nn.Parameter(torch.randn(n_small_proxies, n_small))
            self.med_proxies = nn.Parameter(torch.randn(n_med_proxies, n_medium))
        else:
            self.small_proxies = None
            self.med_proxies = None
        
        # Create classifiers for intermediate levels
        self.use_classifiers = use_classifiers
        if use_classifiers:
            self.small_cls = Classifier(n_latent, n_labels=n_small_proxies, n_layers=0)
            self.med_cls = Classifier(n_latent, n_labels=n_med_proxies, n_layers=0)
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        z: torch.Tensor,
        small_emb: torch.Tensor | None,      # (P, n_emb)  pathway text/prototypes
        med_emb: torch.Tensor | None,        # (M, n_emb)  module text/prototypes
        cls_emb: torch.Tensor,        # (C, n_emb)  class text embeddings
        cls2module: torch.Tensor,     # (C,) class -> module id
        module2pw: torch.Tensor,      # (M,) module -> pathway id
        **kwargs
    ):
        # Noise injection and dropout
        if self.training:
            if self.noise > 0:
                z = z + self.noise * torch.randn_like(z)
            if self.dropout_rate > 0:
                z = self.dropout(z)
            
        # Shift data to current device
        device = z.device
        cls2module = cls2module.to(device)
        module2pw = module2pw.to(device)
        # RNA
        # TODO: add orthogonality loss between levels
        z_s = self.z2s(z)      # (B,S)
        z_m = self.z2m(z)      # (B,M)
        
        # Text levels, use either projections from aggregated text or learnable proxies
        if small_emb is None or self.small_proxies is not None:
            se = self.small_proxies
        else:
            se = self.t2s(small_emb) 
        if med_emb is None or self.med_proxies is not None:
            me = self.med_proxies
        else:
            me = self.t2m(med_emb) 
        # Normalize embeddings
        t_s = F.normalize(se, dim=-1)
        t_m = F.normalize(me, dim=-1)
        # Use full text embedding on the class level
        t_c = F.normalize(self.t2c(cls_emb), dim=-1)
        
        # Calculate pathway logits (root-level)
        if self.use_classifiers:
            logits_s = self.small_cls(z) / self.small_T
        else:
            logits_s = F.normalize(z_s, dim=-1) @ t_s.T / self.small_T            # (B, P)
        # Get residual module level
        if self.use_classifiers:
            logits_mod_res = self.med_cls(z) / self.med_T
        else:
            logits_mod_res = F.normalize(z_m, dim=-1) @ t_m.T / self.med_T   # (B, M)

        # Parent pathway logits for each module: (B, M)
        parent_pw_logits = logits_s[:, module2pw]       # broadcast by indexing
        # Combine resdiual with parent logits
        logits_m = parent_pw_logits + logits_mod_res    # (B, M)
        
        # RNA Projection to class space
        z_c = self.z2c(z)      # (B,C)
        # Class logits = parent module + residual
        logits_c_res = z_c @ t_c.T / self.T     # (B, C)
        # parent module logits per class: (B, C)
        parent_mod_logits = logits_m[:, cls2module]
        # Combine class residuals with module parents
        logits_c = parent_mod_logits + logits_c_res      # (B, C)

        # Return individual results
        return {
            MODULE_KEYS.Z_SHARED_KEY: z_c,
            MODULE_KEYS.CLS_PROJ_KEY: t_c,
            MODULE_KEYS.CLS_LOGITS_KEY: logits_c,
            "z_s": z_s, "z_m": z_m, "z_c": z_c,
            "t_s": t_s, "t_m": t_m, "t_c": t_c,
            "logits_s": logits_s, "logits_m": logits_m, "logits_c": logits_c,
        }


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
        skip_projection: bool = True,
        temperature: float = 0.1,
        min_temperature: float = 0.07,
        use_learnable_temperature: bool = False,
        use_controller: bool = True,
        mode: Literal["latent_to_class", "class_to_latent", "shared"] = "latent_to_class",
        use_cross_attention: bool = False,
        linear_z_proj: bool = True,
        linear_c_proj: bool = True,
        noise_sigma: float | None = None,
        freeze_cls_proj: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        self.class_embed_dim = class_embed_dim
        self.dropout_rate = dropout_rate
        self.funnel = funnel
        self.mode = mode
        self._temperature = temperature
        self._min_temperature = min_temperature
        self.use_learnable_temperature = use_learnable_temperature
        self.use_cross_attention = use_cross_attention
        self.noise_sigma = noise_sigma
        self.freeze_cls_proj = freeze_cls_proj
        # Create a learnable temperature scaling
        if use_learnable_temperature:
            self._temperature = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # ---- Helper for building projections ---- #
        def build_projection(n_in: int, n_out: int, linear: bool = False) -> nn.Module:
            """Creates a projection block based on n_hidden/n_layers settings."""
            if skip_projection and n_in == n_out:
                return nn.Identity()
            if linear:
                return nn.Linear(n_in, n_out, bias=False)
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
            self.latent_projection = build_projection(n_input, class_embed_dim, linear=linear_z_proj)
            self.class_projection = nn.Identity()
        # Class embedding space --> latent space projection
        elif mode == "class_to_latent":
            self.n_output = n_input
            self.latent_projection = nn.Identity()
            self.class_projection = build_projection(class_embed_dim, n_input, linear=linear_c_proj)
        # Latent --> shared dim <-- Class embedding space
        elif mode == "shared":
            if shared_projection_dim is None:
                shared_projection_dim = min(n_input, class_embed_dim)
            # Create z projection
            self.latent_projection = build_projection(n_input, shared_projection_dim, linear=linear_z_proj)
            # Create class projection
            self.class_projection = build_projection(class_embed_dim, shared_projection_dim, linear=linear_c_proj)
            # Update output dimension
            self.n_output = shared_projection_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- Dropout ---- #
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cdropout = nn.Dropout(p=dropout_rate)
        
        # Toggle freeze
        self.cls_proj_frozen = False
        
        # Clip controller
        self.use_controller = use_controller
        if use_controller:
            from src._train.controller import ClipController
            self.controller = ClipController(T_init=temperature, T_min=min_temperature)

    @property
    def temperature(self) -> torch.Tensor:
        # Use minimum temperature at inference
        if not self.training:
            return self._min_temperature
        # Return training temperature
        if self.use_learnable_temperature:
            # Calculate logit scale
            return 1.0 / self._temperature.clamp(-1, 4.6).exp()
        # Use fixed / annealed T
        elif self.use_controller:
            return self.controller.T
        else:
            return self._temperature
        
    def _freeze_cls_proj(self):
        # Freeze class projection module
        if self.class_projection is not nn.Identity() and not self.cls_proj_frozen:
            logging.info('Freezing cls projection.')
            for p in self.class_projection.parameters():
                p.requires_grad = False
            self.cls_proj_frozen = True
    
    def forward(self, x: torch.Tensor, cls_emb: torch.Tensor, return_logits: bool = False, T: float | None = None):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, latent_dim)
        cls_emb : Optional[Tensor], shape (n_labels, embed_dim)
        labels : Optional[Tensor], shape (batch,)
        """
        # Check controller freeze
        if self.freeze_cls_proj and self.use_controller and self.controller.collapse:
            self._freeze_cls_proj()
        # Get class embeddings
        cls_emb = cls_emb.to(x.device)

        # Apply projections
        z = self.latent_projection(x)               # (batch, d)
        c = self.class_projection(cls_emb)     # (n_labels, d)
        # Add some noise to class embedding to avoid having a fixed point
        if self.noise_sigma is not None and self.noise_sigma > 0:
            c = c + self.noise_sigma * F.normalize((torch.rand_like(c) * 2 - 1), dim=-1)
        if self.dropout_rate > 0:
            z = self.dropout(z)
        # Apply normalization to projections
        z_norm = F.normalize(z, dim=-1)
        c_norm = F.normalize(c, dim=-1)
        
        if not return_logits:
            return {
                MODULE_KEYS.Z_SHARED_KEY: z_norm,
                MODULE_KEYS.CLS_PROJ_KEY: c_norm
            }
        
        # Set temperature
        T = T if T is not None else self.temperature

        # ---- OPTION 1: Cross-Attention path ---- #
        if self.use_cross_attention:
            B, D = z_norm.shape
            # Reshape for MultiheadAttention: [B, 1, D] attends to [1, C, D]
            q = z.unsqueeze(1) / T         # queries = latents
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
            #logits = logits / T      # apply temperature scaling

        # ---- OPTION 2: Standard cosine similarity ---- #
        else:
            if c_norm.ndim == 3:
                sim_aug = torch.einsum("bd,cmd->bcm", z_norm, c_norm)   # (B, C, M)
                if self.training:
                    logits = torch.logsumexp(sim_aug / T, dim=-1)  # (B, C)
                else:
                    logits = torch.max(sim_aug / T, dim=-1).values  # (B, C)
            else:
                logits = (z_norm @ c_norm.T) / T
            z_proj = z_norm

        # Optionally return softmax
        logits = logits if self.logits else F.softmax(logits, dim=-1)
        # Return alignment
        return {
            MODULE_KEYS.CLS_LOGITS_KEY: logits,
            MODULE_KEYS.Z_SHARED_KEY: z_proj,
            MODULE_KEYS.CLS_PROJ_KEY: c_norm
        }
        
        
class Reranker(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*n_input + 1, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, 1)   # score
        )
        self.K = None
        
    @property
    def active(self):
        return self.K is not None and self.K > 0

    def forward(self, z, t, s):
        # z: (B, K, d)
        # t: (B, K, d)
        # s: (B, K, 1)
        # Save last K
        self.K = z.shape[1]
        x = torch.cat([z, t, z * t, s], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B, K)
    

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

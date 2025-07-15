import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import collections
from collections.abc import Callable, Iterable
from typing import Literal, Iterable, Optional

from torch.distributions import Normal


def _identity(x):
    return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable



class Block(nn.Module):
    """Simple layer block"""

    def __init__(
            self, 
            n_input: int, 
            n_output: int,
            n_head: int,
            use_batch_norm: bool = False,
            use_layer_norm: bool = True,
            use_activation: bool = True,
            use_attention: bool = True,
            dropout_rate: float = 0.1,
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
        ):
        super().__init__()
        # Save hyperparameters
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_activation = use_activation
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        # Init modules
        head_size = n_input // n_head
        self.root = nn.Linear(n_input, n_output, bias=bias)
        if self.use_attention:
            self.attn = MultiHeadAttention(num_heads=n_head, n_input=n_input, head_size=head_size, dropout_rate=dropout_rate)
        else:
            self.attn = _identity
        self.feedfw = nn.Linear(n_input, n_output, bias=bias)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_output)
        else:
            self.batch_norm = _identity
        if self.use_layer_norm:
            self.layer_norm_1 = nn.LayerNorm(n_input)
            self.layer_norm_2 = nn.LayerNorm(n_input)
        else:
            self.layer_norm_1, self.layer_norm_2 = _identity, _identity
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(n_output, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply multi-head self attention and add new branch of x
        if self.use_attention:
            x = x + self.attn(self.layer_norm_1(x))
        # Apply layer norm
        if self.use_layer_norm:
            x = self.layer_norm_2(x)
        # Branch off into residual pathway
        r = self.feedfw(x)
        # Project main branch
        x = self.root(x)
        # Apply batch norm
        if self.use_batch_norm:
            r = self.batch_norm(r)
        # Apply activation
        if self.use_activation:
            r = self.activation(r)
        # Use final projection layer
        r = self.proj(r)
        # Apply dropout
        if self.dropout_rate > 0:
            r = self.dropout(r)
        # Add second branch to x
        x = x + r
        return x


class FunnelFCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Optional[Iterable[int]] = None,
        n_layers: int = 2,
        min_attn_dim: int = 512,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        use_attention: bool = True,
        n_head: int = 4,
        bias: bool = True,
        inverted: bool = False,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
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

        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in (n_cat_list or [])]
        self.cat_dim = sum(self.n_cat_list)

        # Geometric decay or growth
        hidden_dims = np.geomspace(n_in, n_out, num=n_layers + 1).astype(int)
        # Ensure its divisible by n_heads if attention is used
        if use_attention:
            if inverted:
                hidden_dims -= np.concatenate(((hidden_dims % n_head)[:-1], [0]))
            else:
                hidden_dims -= np.concatenate(([0], (hidden_dims % n_head)[1:]))
        # Init layers
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            in_dim = hidden_dims[i] + self.cat_dim * self.inject_into_layer(i)
            out_dim = hidden_dims[i + 1]
            # Determine whether to to use attention for block, never use on the first layer
            is_attention_block = use_attention and in_dim <= min_attn_dim and i > 0
            # Create block for each layer
            block = Block(
                n_input=in_dim, 
                n_output=out_dim, 
                n_head=n_head,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=use_activation,
                use_attention=is_attention_block,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
                bias=bias
            )
            self.layers.append(block)
    
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        return layer_num == 0 or (layer_num > 0 and self.inject_covariates)

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
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
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
    

class MultiHeadAttention(nn.Module):
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
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        use_feature_mask: bool = False,
        drop_prob: float = 0.25,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.n_input = n_input
        # Setup encoder layers
        self.encoder = FunnelFCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)
        self.return_dist = return_dist
        self.use_feature_mask = use_feature_mask
        self.drop_prob = drop_prob

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
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
        if self.use_feature_mask:
            # Sample mask: 1 for keep, 0 for drop
            feature_mask = torch.bernoulli(torch.full((self.n_input,), 1 - self.drop_prob))  # shape: (num_features,)
            feature_mask = feature_mask.view(1, -1)  # shape: (1, num_features)
            feature_mask = feature_mask.expand(x.shape[0], -1)  # broadcast to full batch
            feature_mask = feature_mask.to(x.device)
            x = x * feature_mask
        # Parameters for latent distribution
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
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FunnelFCLayers(
            n_in=n_input,
            n_out=n_output,
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
            nn.Linear(n_output, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_output, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_output, n_output)

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
        logits: bool = True,  # <- for CE loss, return logits
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        class_embed_dim: int = 128,
        use_multihead: bool = False,
        use_cosine_similarity: bool = True,
        n_heads: int = 4,
        temperature: float = 0.05,
        return_latents: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        self.use_multihead = use_multihead
        self.use_cosine_similarity = use_cosine_similarity
        self.class_embed_dim = class_embed_dim
        self.n_labels = n_labels
        self.temperature = temperature
        self.return_latents = return_latents

        # Initialize layers
        if n_hidden > 0 and n_layers > 0:
            self.encoder = FunnelFCLayers(
                n_in=n_input,
                n_out=class_embed_dim,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation_fn=activation_fn,
                **kwargs,
            )
        else:
            # Set encoder to a single linear layer
            self.encoder = nn.Linear(n_input, class_embed_dim)
            class_embed_dim = n_input

        # 2. Check compatibility
        if use_multihead and class_embed_dim % n_heads != 0:
            fallback = 'cosine similarity' if use_cosine_similarity else 'dot product'
            logging.warning(f"class_embed_dim must be divisible by n_heads. Falling back to {fallback}.")
            self.use_multihead = False

        # 3. Class embeddings (used if external not provided)
        self.learned_class_embeds = nn.Parameter(torch.randn(n_labels, class_embed_dim))

        # 4. Optional attention layer
        if self.use_multihead:
            self.attn = nn.MultiheadAttention(
                embed_dim=class_embed_dim, num_heads=n_heads, batch_first=True
            )
        else:
            self.attn = None

    def forward(self, x, class_embeds: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x : Tensor
            Input latent features, shape (batch, n_input)
        class_embeds : Optional[Tensor]
            External class embeddings, shape (n_labels, class_embed_dim)
        """
        z = self.encoder(x)  # (batch, d)

        if class_embeds is None:
            class_embeds = self.learned_class_embeds
        else:
            class_embeds = class_embeds.to(z.device)
        # Calculate classification logits
        if self.use_cosine_similarity:
            z_norm = F.normalize(z, dim=-1)            # (batch, d)
            c_norm = F.normalize(class_embeds, dim=-1) # (n_labels, d)
            logits = torch.matmul(z_norm, c_norm.T) / self.temperature    # (batch, n_labels)
            _z = z_norm
        elif self.attn is not None:
            z_seq = z.unsqueeze(1)  # (batch, 1, d)
            class_seq = class_embeds.unsqueeze(0).expand(z.size(0), -1, -1)  # (batch, n_labels, d)
            attn_out, _ = self.attn(z_seq, class_seq, class_seq)  # (batch, 1, d)
            attn_out = attn_out.squeeze(1)                        # (batch, d)
            logits = torch.matmul(attn_out, class_embeds.T)       # (batch, n_labels)
            _z = attn_out
        else:
            z = z.unsqueeze(1)  # (batch, 1, d)
            class_seq = class_embeds.unsqueeze(0).expand(z.size(0), -1, -1)  # (batch, n_labels, d)
            logits = torch.bmm(z, class_seq.transpose(1, 2)).squeeze(1)      # (batch, n_labels)
            _z = z
        l = logits if self.logits else F.softmax(logits, dim=-1)
        if self.return_latents:
            return l, _z
        else:
            return l

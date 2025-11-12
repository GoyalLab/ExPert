from typing import Iterable

from src.modules._base import (
    Encoder, 
    DecoderSCVI, 
    ContextClassAligner, 
    Classifier
)
import src.utils.io as io
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS, PREDICTION_KEYS
import src.utils.embeddings as emb_utils
from src.utils.common import grad_reverse

from typing import Iterable

import torch
import torch.nn.functional as F

from scvi.data import _constants
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module.base import (
    LossOutput,
    auto_move_data,
)
from scvi.module._vae import VAE

from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution
from scvi.model.base import BaseModelClass

import logging
log = logging.getLogger(__name__)


class XPert(VAE):
    """
    Adaption of scVI and scanVI models to predict Perturb-seq perturbations using context- and class-embedding data.
    """
    _cls_loss_strategies = ['ce', 'focal']
    _align_ext_emb_loss_strategies = ['kl', 'clip']
    _ctx_key = 'ctx'
    _cls_key = 'cls'

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int,
        n_hidden: int = 256,
        n_latent: int = 128,
        n_layers: int = 2,
        n_shared: int = 512,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        ctrl_class_idx: int | None = None,
        use_reconstruction_control: bool = False,
        use_kl_control: bool = True,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.2,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        use_cpm: bool = False,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        decode_covariates: bool = True,
        decode_context_projection: bool = True,
        decode_shared_space: bool = True,
        deeply_inject_covariates: bool = False,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        l1_lambda: float | None = 1e-5,
        l2_lambda: float | None = 1e-3,
        l_mask: str | list[str] | None = None,
        min_kl: float | None = 1.0,
        align_ext_emb_loss_strategy: Literal['kl', 'clip'] | list[str] = 'clip',            # Secondary classifier module (links external embedding)
        use_posterior_mean: bool = False,
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        use_feature_mask: bool = False,
        drop_prob: float = 1e-3,
        decay_rate: float = 0.9,
        use_ctx_adv_cls: bool = True,
        use_ctrl_cls: bool = True,
        use_semantic_target_weights: bool = True,
        extra_encoder_kwargs: dict | None = {},
        extra_decoder_kwargs: dict | None = {},
        extra_ctx_adv_classifier_kwargs: dict | None = {},
        extra_aligner_kwargs: dict | None = {},
        extra_cls_kwargs: dict | None = {},
        pretrain: bool = False,
        **vae_kwargs
    ):
        # Initialize base model class
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=False,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=False,
            use_observed_lib_size=True,
            var_activation=var_activation,
            **vae_kwargs
        )

        # Setup control index
        self.ctrl_class_idx = ctrl_class_idx
        self.use_reconstruction_control = use_reconstruction_control
        self.use_kl_control = use_kl_control
        # Update labels, we don't predict control as a class
        self.n_labels = n_labels if self.ctrl_class_idx is None else n_labels - 1

        # Setup l-norm params
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l_mask = l_mask

        # Setup extra batch transformation params
        self.use_cpm = use_cpm

        # Setup extra basic vae args
        self.min_kl = min_kl
        self.use_posterior_mean = use_posterior_mean

        # Remove control embedding if it exists
        if self.ctrl_class_idx is not None:
            no_ctrl_emb_mask = torch.arange(cls_emb.size(0)) != self.ctrl_class_idx
            cls_emb = cls_emb[no_ctrl_emb_mask]

        # Setup embedding params
        self.n_ctx, self.n_ctx_dim = ctx_emb.shape
        self.n_cls, self.n_cls_dim = cls_emb.shape
        # Save if we have unseen embeddings or not
        self.has_unseen_ctx = self.n_ctx > n_batch
        self.has_unseen_cls = self.n_cls > n_labels
        self.n_shared = n_shared
        self.use_semantic_target_weights = use_semantic_target_weights

        # Set reduction metrics
        if reduction not in ['batchmean', 'mean']:
            raise ValueError(f'Invalid reduction for elbo loss metrics: {reduction}, choose either "batchmean", or "mean".')
        if non_elbo_reduction not in ['batchmean', 'mean', 'sum']:
            raise ValueError(f'Invalid reduction for extra loss metrics: {non_elbo_reduction}, choose either "batchmean", or "mean".')
        self.reduction = reduction
        self.non_elbo_reduction = non_elbo_reduction

        # Setup external embeddings
        self.ctx_emb = torch.nn.Embedding.from_pretrained(ctx_emb, freeze=True)
        self.cls_emb = torch.nn.Embedding.from_pretrained(cls_emb, freeze=True)
        self.cls_sim = torch.nn.Embedding.from_pretrained(cls_sim, freeze=True) if cls_sim is not None else None

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # ----- Setup encoder module -----
        self.default_encoder_kwargs = {
            'n_input': n_input,
            'n_output': n_latent,
            'n_hidden': n_hidden,
            'n_cat_list': None,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm_encoder,
            'use_layer_norm': use_layer_norm_encoder,
            'var_activation': var_activation,
            'use_feature_mask': use_feature_mask,
            'drop_prob': drop_prob,
            'n_dim_context_emb': None,
            'use_context_inference': None,
        }
        extra_encoder_kwargs.update(self.default_encoder_kwargs)
        self.z_encoder = Encoder(**extra_encoder_kwargs)

        # ----- Setup decoder module -----
        # Whether to decode from z or z_shared TODO: fix that we can actually decode from z_shared, currently does not work because of pz shape mismatch
        self.decode_shared_space = decode_shared_space
        self.decode_covariates = decode_covariates
        self.decode_context_projection = decode_context_projection
        z_dim = n_shared if decode_shared_space else n_latent
        # Whether to decode covariates
        n_input_decoder = z_dim + n_continuous_cov * decode_covariates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        decoder_cat_list = cat_list if decode_covariates else None
        n_ctx_dim_decoder = None
        # Whether to include covariates in decoder
        if decode_covariates:
            # Use projected context embedding
            if decode_context_projection:
                n_ctx_dim_decoder = self.n_shared
            # Use raw context embedding
            else:
                n_ctx_dim_decoder = self.n_ctx_dim
        # Init actual decoder
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=decoder_cat_list,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation='softmax',
            inject_covariates=deeply_inject_covariates,
            n_dim_context_emb=n_ctx_dim_decoder,
            **extra_decoder_kwargs,
        )

        # Setup adversial context classifier
        if use_ctx_adv_cls:
            self.ctx_adv_classifier = Classifier(
                n_input=n_latent,
                n_labels=self.n_batch,
                **extra_ctx_adv_classifier_kwargs,
            )
        else:
            self.ctx_adv_classifier = None

        # ----- Setup external embedding aligner -----

        # Setup an additional classifier on z, purely supervised
        self.classifier = Classifier(
            n_input=n_latent,
            n_labels=self.n_labels,
            **extra_cls_kwargs,
        )
       
        # Setup aligner
        self.aligner = ContextClassAligner(
            n_input=n_latent,
            n_shared=n_shared,
            ctx_emb_dim=self.n_ctx_dim,
            cls_emb_dim=self.n_cls_dim,
            **extra_aligner_kwargs
        )

        # Check loss strategies
        self.set_align_ext_emb_strategies(align_ext_emb_loss_strategy)

        # Set semantic embedding targets for cross entropy
        if self.use_semantic_target_weights:
            log.info(f'Computing context similarity weights.')
            self.ctx_targets = self.compute_semantic_embedding_targets(emb=self.ctx_emb.weight.data)
            log.info(f'Computing class similarity weights.')
            self.cls_targets = self.compute_semantic_embedding_targets(emb=self.cls_emb.weight.data)
        else:
            self.ctx_targets = None
            self.cls_targets = None

        # Set state
        self.pretrain = pretrain

        # Buffers
        self.decay_rate = decay_rate
        # ----- Debug -----
        self._step = 0

    def draw_module(self):
        return
        from torchview import draw_graph

        graph = draw_graph(self, input_size=(self), expand_nested=True)
        graph.visual_graph.render("lightning_graph", format="png")

    def compute_semantic_embedding_targets(
        self,
        emb: torch.Tensor,
        use_gpu: bool = True,
        batch_size: int = 512,
        batched: bool = False,
    ) -> torch.Tensor:
        """
        Compute class similarity matrix for semantic label smoothing in batches.
        emb: (N, D)
        returns: (N, N) normalized similarity matrix.
        """
        device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
        emb = F.normalize(emb.to(device), dim=-1)
        # Do simple MM if tensor is not too large
        if not batched:
            return emb @ emb.T
        
        # Do batched version
        N, D = emb.shape
        targets = torch.zeros((N, N), device=device, dtype=emb.dtype)

        for i in range(0, N, batch_size):
            j_end = min(i + batch_size, N)
            # Compute cosine similarity for this batch vs all embeddings
            batch = emb[i:j_end]                       # (B, D)
            targets[i:j_end] = batch @ emb.T                       # fill in chunk

        return targets

    def set_align_ext_emb_strategies(self, align_ext_emb_loss_strategy) -> None:
        """Set alignment strategies."""
        # Make sure the resulting type is a list
        self.align_ext_emb_loss_strategies: list[str] = align_ext_emb_loss_strategy if isinstance(align_ext_emb_loss_strategy, list) else [align_ext_emb_loss_strategy]
        # Check for allowed options
        for align_strategy in self.align_ext_emb_loss_strategies:
            if align_strategy not in self._align_ext_emb_loss_strategies:
                raise ValueError(f'Unrecognized alignment strategy: {align_strategy}, choose one of {self._align_ext_emb_loss_strategies}')
        
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")
        # Do full forward pass
        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.LABEL_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
                MODULE_KEYS.G_EMB_KEY: tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None),
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
        # Perform a cached forward passed with minified model
        else:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }
    
    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        label: torch.Tensor,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        # Apply CMP normalization to x
        if self.use_cpm:
            x_ = x_ / library * 1e6
        # Apply log1p transformation to input batch
        if self.log_variational:
            x_ = torch.log1p(x_)
        # TODO: add min-max scaling

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Perform forward pass through encoder
        inference_out: dict[str, torch.Tensor] = self.z_encoder(encoder_input, *categorical_input, g=g)
        # Unpack encoder output
        qz = inference_out[MODULE_KEYS.QZ_KEY]
        z = inference_out[MODULE_KEYS.Z_KEY]
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == 'embedding':
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        # Construct output object
        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            REGISTRY_KEYS.LABELS_KEY: label,
            REGISTRY_KEYS.BATCH_KEY: batch_index
        }
    
    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        # Set z either to initial encoder output or to aligner projection
        z_key = MODULE_KEYS.Z_SHARED_KEY if self.decode_shared_space else MODULE_KEYS.Z_KEY
        # Return generative data
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[z_key],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: inference_outputs[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: inference_outputs[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
            MODULE_KEYS.CTX_PROJ_KEY: inference_outputs.get(MODULE_KEYS.CTX_PROJ_KEY),
        }
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        ctx_proj: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        from scvi.distributions import (
            NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
        # Add context embedding to decoder
        if self.decode_covariates:
            # Get inflated context embeddings for batch (either projected or raw)
            if self.decode_context_projection:
                ctx_emb = ctx_proj[batch_index.squeeze(-1)]
            else:
                ctx_emb = self.ctx_emb(batch_index).reshape(batch_index.shape[0], -1)
        else:
            ctx_emb = None
        # Perform decoder forward pass
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
            context_emb=ctx_emb
        )

        if self.dispersion == "gene-label":
            px_r = linear(
                F.one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    @auto_move_data
    def classify(
        self,
        x: torch.Tensor | None,
        batch_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool | None = None,
        inference_outputs: dict[str, torch.Tensor] | None = None,
        inference: bool = False,
        return_logits: bool = True,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        x
            Tensor of shape ``(n_obs, n_vars)``.
        batch_index
            Tensor of shape ``(n_obs,)`` denoting batch indices.
        cont_covs
            Tensor of shape ``(n_obs, n_continuous_covariates)``.
        cat_covs
            Tensor of shape ``(n_obs, n_categorical_covariates)``.
        use_posterior_mean
            Whether to use the posterior mean of the latent distribution for
            classification.
      
        Returns
        -------
        Tensor of shape ``(n_obs, n_labels)`` denoting logit scores per label.
        Before v1.1, this method by default returned probabilities per label,
        see #2301 for more details.
        """
        # Try caching inference and recalculate if missing
        if inference_outputs is None:
            inference_outputs = self.inference(
                x, 
                batch_index, 
                label, 
                g, 
                cont_covs, 
                cat_covs,
            )
        # Use model setting by default but overwrite if option is specified (e.g. during inference)
        use_posterior_mean = self.use_posterior_mean if use_posterior_mean is None else use_posterior_mean
        # Get inference outputs
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        # Aligner forward pass using either qz mean or sampled z
        _z = qz.loc if use_posterior_mean else z
        
        # Optionally use different embeddings, fall back to internals if none are given
        ctx_emb = ctx_emb if ctx_emb is not None else self.ctx_emb.weight
        cls_emb = cls_emb if cls_emb is not None else self.cls_emb.weight
        # Use regular alignment
        if not inference:
            align_out: dict[str, torch.Tensor] = self.aligner(
                _z, 
                ctx_emb=ctx_emb, cls_emb=cls_emb,
                ctx_idx=batch_index, cls_idx=label,
                return_logits=return_logits
            )
        else:
            align_out: dict[str, torch.Tensor] = self.aligner.classify(
                z=_z, ctx_emb=ctx_emb, ctx_idx=batch_index, cls_emb=cls_emb, return_logits=return_logits
            )
        # Add alignment output to inference
        inference_outputs.update(align_out)
        return inference_outputs
    
    def _reduce_loss(self, loss: torch.Tensor, reduction: str) -> torch.Tensor:
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'batchmean':
            return loss.sum(-1).mean()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        
    def _ce_semantic(
        self,
        logits: torch.Tensor,
        target_sim: torch.Tensor,
        T: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        logits: (B, N)
        targets: (B,) integer class labels
        target_sim: (N, N) semantic smoothing targets from embeddings
        """
        p = F.softmax(target_sim / T, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.sum(-p * log_probs, dim=-1)
    
    def _clip_loss(
            self, 
            z: torch.Tensor, 
            logits: torch.Tensor, 
            y: torch.Tensor, 
            emb: torch.Tensor, 
            T: torch.Tensor,
            weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
        # Normalize z and embedding
        z = F.normalize(z, dim=-1)
        emb = F.normalize(emb, dim=-1)

        # For class -> latent, restrict to the classes present in the batch
        chosen = emb[y]   # (batch, d)
        logits_c2z = (chosen @ z.T) / T # (batch, batch)
        # Each class embedding should match its corresponding latent (diagonal)
        labels_c2z = torch.arange(z.size(0), device=z.device)

        # ---- Calculate losses ----
        # Latent -> class loss, logits are already scaled by classifier
        if self.use_semantic_target_weights and weights is not None:
            # Subset weights
            target_sim = weights.to(y.device)[y]
            # Use embedding similarities as reference for ce
            loss_z2c = self._ce_semantic(logits, target_sim, T=T)
        else:
            # Use hard one-hot labels as reference for ce
            loss_z2c = F.cross_entropy(logits, y, reduction='none')
        # Class -> latent loss, can't use semantics here
        loss_c2z = F.cross_entropy(logits_c2z, labels_c2z, reduction='none')

        # Symmetric loss per sample
        return 0.5 * (loss_z2c + loss_c2z)
    
    def _joint_clip_loss(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate joint logits loss."""
        # Target should be diagonal indices
        labels = torch.arange(logits.shape[0], device=logits.device)
        # Align latent --> joint and joint --> latent
        loss_h2j = F.cross_entropy(logits, labels, reduction='none')
        loss_j2h = F.cross_entropy(logits.T, labels, reduction='none')
        # Return combined loss
        return 0.5 * (loss_h2j + loss_j2h)
    
    def _random_unseen_replacement(
        self,
        ctx_idx: torch.Tensor,
        cls_idx: torch.Tensor,
        p: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly replace some labels with unseen labels to keep zero-shot active.""" 
        if self.has_unseen_ctx:
            ctx_idx = emb_utils.replace_with_unseen_labels(ctx_idx, n_seen=self.n_batch, n_total=self.n_ctx, p=p)
        if self.has_unseen_cls:
            cls_idx = emb_utils.replace_with_unseen_labels(cls_idx, n_seen=self.n_labels, n_total=self.n_cls, p=p)
        return ctx_idx, cls_idx
    
    def _observable_bias_loss(self, random_logits: torch.Tensor, n_obs: int):
        """Punish random predictions that are biased towards observed classes."""
        # Get number of all available classes
        # Calculate probabilities
        probs = random_logits.softmax(dim=-1)
        preds = random_logits.argmax(dim=-1)
        # Get mean density in observed fraction
        frac_prob_obs = probs[:, :n_obs].sum(dim=-1).mean()
        frac_pred_obs = (preds < n_obs).float().mean()
        # Mean expected density
        expected_frac = n_obs / random_logits.size(-1)
        # Return bias loss
        soft_loss = (frac_prob_obs - expected_frac)**2
        hard_loss = (frac_pred_obs - expected_frac)**2
        return soft_loss + hard_loss
    
    def _logit_variance_loss(self, random_logits: torch.Tensor, m: float = 1e-2):
        """Prevent model from pushing down values."""
        var = random_logits.var(dim=-1)
        # penalize too-flat logits
        return torch.relu(m - var).mean()
    
    def get_ctx_buffer(self, reset: bool = False):
        """Return and optionally clear the stored pseudo argmax buffer for context predictions."""
        buffer_name = f"{self._ctx_key}_buffer"
        if not hasattr(self, buffer_name):
            return None

        buf = getattr(self, buffer_name)

        if reset:
            # reset the existing buffer without re-registering
            empty = torch.empty(0, dtype=buf.dtype, device=buf.device)
            setattr(self, buffer_name, empty)

        return buf


    def get_cls_buffer(self, reset: bool = False):
        """Return and optionally clear the stored pseudo argmax buffer for class predictions."""
        buffer_name = f"{self._cls_key}_buffer"
        if not hasattr(self, buffer_name):
            return None

        buf = getattr(self, buffer_name)

        if reset:
            empty = torch.empty(0, dtype=buf.dtype, device=buf.device)
            setattr(self, buffer_name, empty)

        return buf


    def _update_pseudo_buffer(self, random_logits: torch.Tensor, name: str, T: float = 0.1):
        """Append new argmax predictions to a named buffer."""
        preds = random_logits.argmax(dim=-1).detach()
        buf_name = f"{name}_buffer"

        # create or update the buffer safely
        if not hasattr(self, buf_name):
            # register once (persistent, moves with .to(device))
            self.register_buffer(buf_name, preds.clone())
        else:
            buf = getattr(self, buf_name)
            preds = preds.to(buf.device)
            new_buf = torch.cat((buf, preds), dim=0)
            setattr(self, buf_name, new_buf)

    def _grad_reversed_entropy_loss(self, logits: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Maximize entropy via gradient reversal."""
        logits_rev = grad_reverse(logits, lam)
        p = torch.softmax(logits_rev, dim=-1)
        return (p * p.log()).sum(dim=-1).mean()
    
    def _bias_bce(self, pseudo_rev_logits: torch.Tensor, n_obs: int, n: int) -> torch.Tensor:
        """Compute binary cross entropy between observed and unobserved predictions on a grad reversed random alignment."""
        B = pseudo_rev_logits.size(0)
        device = pseudo_rev_logits.device
        # Target: all predictions that are observable classes
        is_obs = (pseudo_rev_logits.argmax(-1) < n_obs).float().unsqueeze(1).detach()
        # Classify observed vs. not
        obs_mask = torch.ones((B, n_obs), device=device)
        no_obs_mask = torch.zeros((B, (n-n_obs)), device=device)
        target_mask = torch.cat((obs_mask, no_obs_mask), dim=1)
        target_one_hot = is_obs * target_mask
        # Gradient-reversed logit bce to observation
        return F.binary_cross_entropy_with_logits(pseudo_rev_logits, target_one_hot)

    def _pseudo_latent_loss(
        self,
        z: torch.Tensor,
        frac: float = 0.1,
        alpha: float = 0.1,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ensure that a random latent alignment still has an unbiased distribution over all possible classes."""
        B, D = z.shape
        B_pseudo = int(B * frac)
        device = z.device
        # Sample random latents
        z_pseudo = torch.randn(B_pseudo, D, device=device)
        # Revert gradient on pseudo z
        pseudo_z_rev = grad_reverse(z_pseudo)
        # Sample random context and class indices
        ctx_idx = torch.randint(0, self.n_ctx, (B_pseudo,), device=z.device)
        cls_idx = torch.randint(0, self.n_cls, (B_pseudo,), device=z.device)
        # Align pseudo-latent to embeddings
        align_out = self.aligner(
            pseudo_z_rev, 
            ctx_emb=self.ctx_emb.weight, cls_emb=self.cls_emb.weight,
            ctx_idx=ctx_idx, cls_idx=cls_idx,
            return_logits=True,
            T=T,
        )
        # Extract logits from pseudo-alignment
        z_pseudo_shared: torch.Tensor = align_out[MODULE_KEYS.Z_SHARED_KEY]
        p_ctx_logits: torch.Tensor = align_out[MODULE_KEYS.CTX_LOGITS_KEY]
        p_cls_logits: torch.Tensor = align_out[MODULE_KEYS.CLS_LOGITS_KEY]

        # Save buffers of argmax predictions for epoch-level loss and monitoring during training
        self._update_pseudo_buffer(p_ctx_logits, self._ctx_key)
        self._update_pseudo_buffer(p_cls_logits, self._cls_key)

        # Calculate BCE on prediciting observed class over unobserved
        ctx_loss = self._bias_bce(p_ctx_logits, n_obs=self.n_batch, n=self.n_ctx)
        cls_loss = self._bias_bce(p_cls_logits, n_obs=self.n_labels, n=self.n_cls)
        return alpha * ctx_loss + (1-alpha) * cls_loss
    
    def _manifold_regularization_loss(
        self,
        ctx_proj: torch.Tensor,
        cls_proj: torch.Tensor,
        frac: float = 0.1,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Calculate manifold regularization loss to ensure projected embedding keeps geometry intact."""
        ctx_reg_loss = torch.tensor(0.0)
        if self.has_unseen_ctx:
            n_ctx_sample = int(frac * self.n_ctx)
            ctx_reg_loss = emb_utils.manifold_regularization(ext_emb=self.ctx_emb.weight, proj_emb=ctx_proj, n_sample=n_ctx_sample)
        cls_reg_loss = torch.tensor(0.0)
        if self.has_unseen_cls:
            n_cls_sample = int(frac * self.n_cls)
            cls_reg_loss = emb_utils.manifold_regularization(ext_emb=self.cls_emb.weight, proj_emb=cls_proj, n_sample=n_cls_sample)
        # Compute final regularization loss
        return alpha * ctx_reg_loss + (1-alpha) * cls_reg_loss
    
    def _ce_cls_loss(self, logits: torch.Tensor, y: torch.Tensor, T: float | None = None) -> torch.Tensor:
        """Calculate additional simple classification loss on initial latent space."""
        target_sim = self.cls_targets
        # Return an CE loss
        if self.use_semantic_target_weights and target_sim is not None:
            # Subset weights
            target_sim = target_sim.to(y.device)[y]
            # Use embedding similarities as reference for ce
            loss = self._ce_semantic(logits, target_sim, T=T)
        else:
            # Use hard one-hot labels as reference for ce
            loss = F.cross_entropy(logits, y, reduction='none')
        return loss
    
    def _adv_ctx_loss(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Calculate adversial context classification loss on z with reversed gradients."""
        z_rev = grad_reverse(z)
        adv_ctx_logits = self.ctx_adv_classifier(z_rev)
        # Calculate loss
        return F.cross_entropy(adv_ctx_logits, ctx.squeeze(-1), reduction='none')
    
    def _center_z_on_ctrl(self, z: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Center z on mean of control cells."""
        if self.ctrl_class_idx is None:
            return z, y, b
        # Get control cell mask
        ctrl_mask = (y == self.ctrl_class_idx).squeeze(-1)
        # Center z on control cell mean TODO: optionally do this per context
        z_centered = z[~ctrl_mask] - z[ctrl_mask].mean(dim=0, keepdim=True)
        y = y[~ctrl_mask]
        b = b[~ctrl_mask]
        return z_centered, y, b


    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        cls_weight: float = 0.1,
        ctx_align_weight: float = 1.0,
        cls_align_weight: float = 1.0,
        joint_align_weight: float = 0.1,
        random_unseen_replacement_p: float = 0.05,
        pseudo_latent_frac: float = 0.05,
        manifold_regularization_frac: float = 0.0,
        pseudo_latent_weight: float = 1.0,
        manifold_regularization_weight: float = 1.0,
        align_temp_reg_weight: float = 1.0,
        T_align: float | None = None,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # Unpack input tensor
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        ctx: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]
        cls: torch.Tensor = tensors[REGISTRY_KEYS.LABELS_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z_mat = kl_divergence(qz, pz)
        # Calculate reconstruction loss over batch and all features
        reconst_loss_mat = -px.log_prob(x)
        # Clamp KL to a minimum if option is provided
        if self.min_kl is not None:
            kl_divergence_z_mat = torch.clamp(kl_divergence_z_mat, min=self.min_kl)
        # Aggregate elbo losses over latent dimensions / input features, will get batch normalized internally
        if self.reduction == 'batchmean':
            kl_divergence_z = kl_divergence_z_mat.sum(-1) # (b,)
            kl_div_per_latent = kl_divergence_z_mat.sum(0) # (l,)
            reconst_loss = reconst_loss_mat.sum(-1)
        elif self.reduction == 'mean':
            kl_divergence_z = kl_divergence_z_mat.mean(-1) # (b,)
            kl_div_per_latent = kl_divergence_z_mat.mean(0) # (l,)
            reconst_loss = reconst_loss_mat.mean(-1)
        else:
            raise ValueError(f'reduction has to be either "batchmean" or "mean", got {self.reduction}')

        # Use full batch by default
        n_obs_minibatch = x.shape[0]
        
        # Weighted reconstruction and KL
        weighted_reconst_loss = rl_weight * reconst_loss
        weighted_kl_local = kl_weight * kl_divergence_z

        # Save reconstruction losses
        kl_locals = {MODULE_KEYS.KL_Z_KEY: kl_divergence_z}
        # Batch normalize elbo loss of reconstruction + kl --> batchmean reduction
        total_loss = torch.mean(weighted_reconst_loss + weighted_kl_local)

        # Create extra metric container
        extra_metrics = {MODULE_KEYS.KL_Z_PER_LATENT_KEY: kl_div_per_latent}
        # Center z on control cell mean if control cells are encoded
        z, cls, ctx = self._center_z_on_ctrl(z, cls, ctx)

        # Collect inital loss and extra parameters
        lo_kwargs = {
            'reconstruction_loss': reconst_loss,
            'kl_local': kl_locals,
            'true_labels': cls,
            'n_obs_minibatch': n_obs_minibatch
        }

        # Re-format labels
        ctx = ctx.squeeze(-1)
        cls = cls.squeeze(-1)

        # Calculate classification loss on z
        if cls_weight > 0:
            z_cls_logits = self.classifier(z)
            z_ce_loss = self._ce_cls_loss(z_cls_logits, y=cls, T=0.1)
            z_ce_loss = self._reduce_loss(z_ce_loss, reduction=self.non_elbo_reduction)
            # Add classification loss details
            if torch.isnan(z_ce_loss):
                import pdb
                pdb.set_trace()
            lo_kwargs.update({
                'classification_loss': z_ce_loss,
                'logits': z_cls_logits,
            })
            # Add to total loss
            total_loss = total_loss + z_ce_loss * cls_weight

        # Only do alignment part if loss > 0
        if io.non_zero(ctx_align_weight) or io.non_zero(cls_align_weight):
            # Do alignment
            alignment_output = self.aligner(
                z, 
                ctx_idx=ctx, 
                ctx_emb=self.ctx_emb.weight,
                cls_idx=cls,
                cls_emb=self.cls_emb.weight,
                T=T_align
            )
            # Randomly set some labels to unseen contexts and classes to keep these embeddings active
            if random_unseen_replacement_p > 0:
                ctx, cls = self._random_unseen_replacement(ctx_idx=ctx.unsqueeze(0), cls_idx=cls.unsqueeze(0), p=random_unseen_replacement_p)
            # Extract shared latent space
            z_shared = alignment_output.get(MODULE_KEYS.Z_SHARED_KEY)
            # Extract the embedding projections to shared space
            ctx2z = alignment_output.get(MODULE_KEYS.CTX_PROJ_KEY)
            cls2z = alignment_output.get(MODULE_KEYS.CLS_PROJ_KEY)
            joint2z = alignment_output.get(MODULE_KEYS.JOINT_PROJ_KEY)
            # Extract context logits from aligner
            ctx_logits = alignment_output.get(MODULE_KEYS.CTX_LOGITS_KEY)
            if ctx_logits is None:
                # Calculate manually
                ctx_logits = self.aligner.get_ctx_logits(z_shared, ctx2z, T=T_align)
            # Extract class logits from aligner
            cls_logits = alignment_output.get(MODULE_KEYS.CLS_LOGITS_KEY)
            if cls_logits is None:
                # Calculate manually
                cls_logits = self.aligner.get_cls_logits(z_shared, cls2z, T=T_align)
            # Extract shared logits from aligner (if given)
            joint_logits = alignment_output.get(MODULE_KEYS.JOINT_LOGITS_KEY)
            if joint_logits is None:
                # Calculate manually
                joint_logits = self.aligner.get_joint_logits(z_shared, joint2z, T=T_align)

            # Calculate clip loss for context embedding
            ctx_loss_ps = self._clip_loss(
                z_shared, 
                logits=ctx_logits, 
                y=ctx, 
                emb=ctx2z, 
                weights=self.ctx_targets,
                T=self.aligner.ctx_temperature
            )
            # Calculate clip loss for class embedding
            cls_loss_ps = self._clip_loss(
                z_shared, 
                logits=cls_logits, 
                y=cls, 
                emb=cls2z, 
                weights=self.cls_targets,
                T=self.aligner.cls_temperature
            )
            # Apply reductions
            ctx_loss = self._reduce_loss(ctx_loss_ps, reduction=self.non_elbo_reduction)
            cls_loss = self._reduce_loss(cls_loss_ps, reduction=self.non_elbo_reduction)
            # Add to extra metrics
            extra_metrics[LOSS_KEYS.CLIP_CTX_CLS_LOSS] = ctx_loss
            extra_metrics[LOSS_KEYS.CLIP_CLS_LOSS] = cls_loss
            # Add predictions to loss output
            extra_metrics[PREDICTION_KEYS.PREDICTION_KEY] = torch.argmax(cls_logits, dim=-1).squeeze(-1).detach()
            
            # Combine alignment losses
            align_loss = ctx_loss * ctx_align_weight + cls_loss * cls_align_weight

            # Add joint embedding clip loss if available and enabled
            if joint_logits is not None and joint_align_weight > 0:
                joint_loss_ps = self._joint_clip_loss(joint_logits)
                joint_loss = self._reduce_loss(joint_loss_ps, reduction=self.non_elbo_reduction)
                # Add to extra metrics
                extra_metrics[LOSS_KEYS.JOINT_LOSS] = joint_loss
                # Add to overall alignment loss
                align_loss = align_loss + joint_loss * joint_align_weight
            
            # Add additional regularization losses
            if pseudo_latent_frac > 0 and pseudo_latent_weight > 0:
                # Enforce model to not be biased towards observed classes
                pseudo_latent_loss = self._pseudo_latent_loss(z, frac=pseudo_latent_frac, T=T_align)
                # Only add loss if not nan
                if not torch.isnan(pseudo_latent_loss):
                    # Add to overall alignment loss
                    align_loss = align_loss + pseudo_latent_loss * pseudo_latent_weight
                    # Add to extra metrics for logging
                    extra_metrics[LOSS_KEYS.PSEUDO_Z_LOSS] = pseudo_latent_loss
            # Add a manifold regularization loss on all embedding projections
            if manifold_regularization_frac > 0:
                mr_loss = self._manifold_regularization_loss(ctx_proj=ctx2z, cls_proj=cls2z, frac=manifold_regularization_frac, alpha=0.5)
                # Add to overall alignment loss
                align_loss = align_loss + mr_loss * manifold_regularization_weight
                # Add to extra metrics
                extra_metrics[LOSS_KEYS.MANIFOLD_REG_LOSS] = mr_loss
            # Add a small regularization on temperature
            if align_temp_reg_weight > 0:
                temp_reg_loss = self.aligner.get_temp_reg_loss()
                align_loss = align_loss + temp_reg_loss * align_temp_reg_weight
                extra_metrics[LOSS_KEYS.T_REG_LOSS] = temp_reg_loss
            # Add to extra metrics
            extra_metrics[LOSS_KEYS.ALIGN_LOSS] = align_loss
            # Add to total loss
            total_loss = total_loss + align_loss
        # Add extra metrics to loss output
        lo_kwargs['extra_metrics'] = extra_metrics

        # Set total loss
        lo_kwargs[LOSS_KEYS.LOSS] = total_loss
        return LossOutput(**lo_kwargs)

    def on_load(self, model: BaseModelClass):
        """Sync model adata manager on load"""
        manager = model.get_anndata_manager(model.adata, required=True)
        source_version = manager._source_registry[_constants._SCVI_VERSION_KEY]
        version_split = source_version.split('.')

        if int(version_split[0]) >= 1 and int(version_split[1]) >= 1:
            return

        # need this if <1.1 model is resaved with >=1.1 as new registry is
        # updated on setup
        manager.registry[_constants._SCVI_VERSION_KEY] = source_version

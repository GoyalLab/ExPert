from typing import Iterable

from src.modules._base import (
    Encoder, 
    DecoderSCVI, 
    ContextClassAligner, 
)
import src.utils.io as io
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS
from src.utils.distributions import rescale_targets
from src.utils.common import GradientReversalFn, batchmean

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
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.2,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        use_cpm: bool = False,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        decode_covariates: bool = False,
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
        use_learnable_temperature: bool = True,
        use_posterior_mean: bool = False,
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        use_feature_mask: bool = False,
        drop_prob: float = 1e-3,
        temperature: float = 1.0,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        extra_aligner_kwargs: dict | None = None,
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

        # Setup l-norm params
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l_mask = l_mask

        # Setup extra batch transformation params
        self.use_cpm = use_cpm

        # Setup extra basic vae args
        self.min_kl = min_kl
        self.use_posterior_mean = use_posterior_mean

        # Setup basic embedding params
        self.n_ctx_dim = ctx_emb.shape[-1]
        self.n_cls_dim = cls_emb.shape[-1]
        self.n_shared = n_shared

        # Set reduction metrics
        if reduction not in ['batchmean', 'mean']:
            raise ValueError(f'Invalid reduction for elbo loss metrics: {reduction}, choose either "batchmean", or "mean".')
        if non_elbo_reduction not in ['batchmean', 'mean', 'sum']:
            raise ValueError(f'Invalid reduction for extra loss metrics: {non_elbo_reduction}, choose either "batchmean", or "mean".')
        self.reduction = reduction
        self.non_elbo_reduction = non_elbo_reduction

        # Initialize learnable temperature scaling for logits
        self.use_learnable_temperature = use_learnable_temperature
        if use_learnable_temperature:
            self.contr_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
            self.clip_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
            self.proxy_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        
        # Setup external embeddings
        self.ctx_emb = torch.nn.Embedding.from_pretrained(ctx_emb, freeze=True)
        self.cls_emb = torch.nn.Embedding.from_pretrained(cls_emb, freeze=True)
        self.cls_sim = torch.nn.Embedding.from_pretrained(cls_sim, freeze=True)

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
        z_dim = n_shared if decode_shared_space else n_latent
        # Whether to decode covariates
        n_input_decoder = z_dim + n_continuous_cov * decode_covariates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        decoder_cat_list = cat_list if decode_covariates else None
        n_ctx_dim_decoder = self.n_ctx_dim if decode_covariates else None
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

        # ----- Setup external embedding aligner -----
       
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

        # ----- Debug -----
        self._step = 0

    def set_align_ext_emb_strategies(self, align_ext_emb_loss_strategy) -> None:
        """Set alignment strategies."""
        # Make sure the resulting type is a list
        self.align_ext_emb_loss_strategies: list[str] = align_ext_emb_loss_strategy if isinstance(align_ext_emb_loss_strategy, list) else [align_ext_emb_loss_strategy]
        # Check for allowed options
        for align_strategy in self.align_ext_emb_loss_strategies:
            if align_strategy not in self._align_ext_emb_loss_strategies:
                raise ValueError(f'Unrecognized alignment strategy: {align_strategy}, choose one of {self._align_ext_emb_loss_strategies}')

    @property
    def clip_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.clip_logit_scale.clamp(-1, 4.6).exp()
        else:
            return self.contrastive_temperature
        
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
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
        use_posterior_mean: bool | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
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
        final_out = {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }
        # Update initial inference outputs
        inference_out.update(final_out)
        
        # Use model setting by default but overwrite if option is specified (e.g. during inference)
        use_posterior_mean = self.use_posterior_mean if use_posterior_mean is None else use_posterior_mean
        # Aligner forward pass using either qz mean or sampled z
        _z = qz.loc if use_posterior_mean else z
        # Optionally use different embeddings, fall back to internals if none are given
        ctx_emb = ctx_emb if ctx_emb is not None else self.ctx_emb.weight
        cls_emb = cls_emb if cls_emb is not None else self.cls_emb.weight
        align_out: dict[str, torch.Tensor] = self.aligner(_z, ctx_emb=ctx_emb, cls_emb=cls_emb)
        # Add alignment output to inference
        inference_out.update(align_out)

        # Return inference outputs
        return inference_out
    
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
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
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
        # TODO: add film or cat for context embedding here
        if self.decode_covariates:
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
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        inference_outputs: dict[str, torch.Tensor] | None = None,
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
        # Try caching inference
        if inference_outputs is None:
            inference_outputs = self.inference(x, batch_index, g, cont_covs, cat_covs, use_posterior_mean=use_posterior_mean)
        # Re-do alignment if it was not cached or new embeddings are given
        if MODULE_KEYS.Z_SHARED_KEY not in inference_outputs or ctx_emb is not None or cls_emb is not None:
            # Replace embedding with new one if given
            ctx_emb = ctx_emb if ctx_emb is not None else self.ctx_emb.weight
            cls_emb = cls_emb if cls_emb is not None else self.cls_emb.weight
            # Calculate alignment
            alignment_out = self.aligner(
                z=inference_outputs[MODULE_KEYS.Z_KEY],
                ctx_emb=ctx_emb, cls_emb=cls_emb
            )
            # Add output to inference
            inference_outputs.update(alignment_out)
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
    
    def _clip_loss(self, z: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Get temperature from aligner
        T = self.aligner.cls_temperature
        # Normalize z and embedding
        z = F.normalize(z, dim=-1)
        emb = F.normalize(emb, dim=-1)

        # Latent -> class loss, logits are already scaled by classifier
        loss_z2c = F.cross_entropy(logits, y, reduction='none')

        # For class -> latent, restrict to the classes present in the batch
        chosen = emb[y]   # (batch, d)
        logits_c2z = (chosen @ z.T) / T # (batch, batch)
        # Each class embedding should match its corresponding latent (diagonal)
        labels_c2z = torch.arange(z.size(0), device=z.device)
        loss_c2z = F.cross_entropy(logits_c2z, labels_c2z, reduction='none')

        # Symmetric loss per sample
        return 0.5 * (loss_z2c + loss_c2z)

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        ctx_align_weight: float = 1.0,
        cls_align_weight: float = 1.0,
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

        # Weighted reconstruction and KL
        weighted_reconst_loss = rl_weight * reconst_loss
        weighted_kl_local = kl_weight * kl_divergence_z

        # Save reconstruction losses
        kl_locals = {MODULE_KEYS.KL_Z_KEY: kl_divergence_z}
        # Batch normalize elbo loss of reconstruction + kl --> batchmean reduction
        total_loss = torch.mean(weighted_reconst_loss + weighted_kl_local)

        # Collect inital loss and extra parameters
        lo_kwargs = {
            'reconstruction_loss': reconst_loss,
            'kl_local': kl_locals,
            'true_labels': cls,
        }
        # Create extra metric container
        extra_metrics = {MODULE_KEYS.KL_Z_PER_LATENT_KEY: kl_div_per_latent}

        # Only do alignment part if loss > 0
        if io.non_zero(ctx_align_weight) or io.non_zero(cls_align_weight):
            # Extract shared latent space
            z_shared = inference_outputs.get(MODULE_KEYS.Z_SHARED_KEY)
            ctx2z = inference_outputs.get(MODULE_KEYS.CTX_PROJ_KEY)
            cls2z = inference_outputs.get(MODULE_KEYS.CLS_PROJ_KEY)
            # Extract context logits from aligner
            ctx_logits = inference_outputs.get(MODULE_KEYS.CTX_LOGITS_KEY)
            # Extract class logits from aligner
            cls_logits = inference_outputs.get(MODULE_KEYS.CLS_LOGITS_KEY)

            # Get loss for context and class embedding for shared z
            ctx_loss_ps = self._clip_loss(z_shared, logits=ctx_logits, y=ctx.squeeze(-1), emb=ctx2z)
            cls_loss_ps = self._clip_loss(z_shared, logits=cls_logits, y=cls.squeeze(-1), emb=cls2z)
            # Apply reductions
            ctx_loss = self._reduce_loss(ctx_loss_ps, reduction=self.non_elbo_reduction)
            cls_loss = self._reduce_loss(cls_loss_ps, reduction=self.non_elbo_reduction)
            # Combine alignment losses
            align_loss = ctx_loss * ctx_align_weight + cls_loss * cls_align_weight
            # Add to extra metrics
            extra_metrics[LOSS_KEYS.ALIGN_LOSS] = align_loss
            extra_metrics[LOSS_KEYS.CTX_CLS_LOSS] = ctx_loss
            extra_metrics[LOSS_KEYS.CLS_LOSS] = cls_loss
            # Add classification loss details
            lo_kwargs.update({
                'classification_loss': cls_loss,
                'true_labels': cls,
                'logits': cls_logits,
            })
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

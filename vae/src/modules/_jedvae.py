from typing import Iterable

from src.modules._base import (
    Classifier, 
    EmbeddingAligner, 
    Encoder, 
    DecoderSCVI, 
    ExternalClassEmbedding, 
    ArcClassifier, 
    ContextEmbedding
)
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


class JEDVAE(VAE):
    """
    Adaption of scVI and scanVI models to predict Perturb-seq perturbations using class-embedding data.
    """
    _cls_module_types = ['standard', 'arc']
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
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        ctx_emb: torch.Tensor | None = None,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float | None = None,
        ext_class_embed_dim: int | None = None,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        linear_classifier: bool = False,
        cls_weights: torch.Tensor | None = None,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        encode_covariates: bool = False,
        decode_covariates: bool = False,
        decode_y: bool = False,                         # Should be disabled for classification tasks
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        l1_lambda: float | None = 1e-5,
        l2_lambda: float | None = 1e-3,
        l_mask: str | list[str] | None = None,
        focal_gamma: float = 2.0,
        min_kl: float | None = 1.0,
        classification_module_type: Literal['standard', 'arc'] = 'arc',              # Main classifier module (does not use external embeddings)
        classification_loss_strategy: Literal['ce', 'focal'] = 'focal',                             # Main classifier loss strategy
        align_ext_emb_loss_strategy: Literal['kl', 'clip'] | list[str] | None = None,            # Secondary classifier module (links external embedding)
        ctrl_class_idx: int | None = None,
        kl_class_temperature: float = 0.1,
        use_learnable_control_emb: bool = False,
        use_learnable_temperature: bool = True,
        use_adversial_context_cls: bool = True,
        use_reconstruction_control: bool = False,                           # Use control cells for reconstruction loss
        use_kl_control: bool = False,                                       # Use control cells for KL loss
        use_classification_control: bool = False,                           # Try to classify control cells
        use_contrastive_control: bool = True,                               # Use control cells as negatives in contrastive loss
        context_classifier_layers: int = 2,
        context_classifier_n_hidden: int = 256,
        contrastive_temperature: float = 0.1,
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        use_feature_mask: bool = False,
        drop_prob: float = 1e-3,
        classifier_parameters: dict = {},
        aligner_parameters: dict | None = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
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
            dropout_rate=dropout_rate_encoder,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=encode_covariates,
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
        
        if dropout_rate_decoder is not None:
            log.warning('Dropout rate for decoder currently unavailable. Will fall back to 0.')
        
        # Classifier parameters
        self.classifier_parameters = classifier_parameters
        self.linear_classifier = linear_classifier
        self.class_weights = cls_weights
        # Save external class embedding dimensions
        self.ext_class_embed_dim = ext_class_embed_dim
        self._update_cls_params()
        self.focal_gamma = focal_gamma
        self.kl_class_temperature = kl_class_temperature
        self.min_kl = min_kl
        # Contrastive parameters
        self.contrastive_temperature = contrastive_temperature
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
            self.contr_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / contrastive_temperature)))
            self.clip_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / contrastive_temperature)))
            self.proxy_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / contrastive_temperature)))
        # Setup an adversion context classifier
        self.use_adversial_context_cls = use_adversial_context_cls
        # Setup learnable control embedding
        self.use_reconstruction_control = use_reconstruction_control
        self.use_kl_control = use_kl_control
        self.use_classification_control = use_classification_control
        self.use_contrastive_control = use_contrastive_control
        # Set control class index
        self.ctrl_class_idx = ctrl_class_idx
        # Class embedding parameters
        self.use_learnable_control_emb = use_learnable_control_emb
        if cls_emb is not None:
            # Initialize external embedding
            self.class_embedding = ExternalClassEmbedding(
                cls_emb=cls_emb, 
                cls_sim=cls_sim, 
                ctrl_class_idx=ctrl_class_idx, 
                use_control=use_learnable_control_emb,
                device=self.device
            )
            log.info(f'Registered class embedding with shape: {self.class_embedding.shape}, using control: {use_learnable_control_emb}')
        else:
            log.info(f'No external class embedding registered with model.')
            self.class_embedding = None 

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # Check classification config
        self.set_classification_module_type(classification_module_type)
        # Check and set classification loss strategy
        self.set_cls_loss_strategy(classification_loss_strategy)
        # Check external class embedding alignments
        self.set_align_ext_emb_strategies(align_ext_emb_loss_strategy)

        # ----- Setup modules -----
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        n_input_decoder = n_latent + n_continuous_cov * decode_covariates
        # Setup default categorical covariates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.use_ext_emb = False
        # Add external context embedding
        if ctx_emb is not None:
            self.use_context_emb = True
            self.use_ext_emb = True
            n_batch = ctx_emb.shape[0]                  # Register number of contexts in embedding
            n_dim_context_emb = ctx_emb.shape[-1]       # Register number of dimensions in embedding
            # Create context embedding class
            _ctx_emb = ContextEmbedding(n_batch, n_dim_context_emb, add_unseen_buffer=False, ext_emb=ctx_emb)
            # Register embedding to both encoder and decoder
            self.e_context_emb = _ctx_emb
            self.d_context_emb = _ctx_emb
        # Add context embedding dimensions to encoder and or decoder dimensions
        elif self.batch_representation == 'embedding' and encode_covariates:
            self.use_context_emb = True
            n_dim_context_emb = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
            self.e_context_emb = ContextEmbedding(n_batch, n_dim_context_emb)
            self.d_context_emb = ContextEmbedding(n_batch, n_dim_context_emb)
        else:
            self.use_context_emb = False
            self.e_context_emb, self.d_context_emb = None, None
            n_dim_context_emb = 0
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # Setup batch index and other categorical co-variates
        self.cat_list = cat_list
        # Choose to encode covariates
        encoder_cat_list = cat_list if encode_covariates else None
        # Choose to decode covariates
        decoder_cat_list = cat_list if decode_covariates else None
        # Setup extra en- and decoder params
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

        # TODO: add option to use external context embedding

        # Re-Init z encoder
        self.z_encoder = Encoder(
            n_input=n_input_encoder, 
            n_output=n_latent, 
            n_hidden=n_hidden,
            n_cat_list=encoder_cat_list, 
            dropout_rate=dropout_rate_encoder, 
            use_batch_norm=use_batch_norm_encoder, 
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            use_feature_mask=use_feature_mask,
            drop_prob=drop_prob,
            n_dim_context_emb=n_dim_context_emb,
            use_context_inference=self.use_ext_emb,
            **_extra_encoder_kwargs,
            return_dist=True,
        )

        # Setup decoder
        self.decode_y = decode_y
        if self.decode_y:
            log.warning(f'Training a CVAE, decoder will be conditioned on class label.')
        # Setup decoder module
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=decoder_cat_list,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation='softmax',
            inject_covariates=deeply_inject_covariates,
            n_dim_context_emb=n_dim_context_emb,
            **_extra_decoder_kwargs,
        )

        # ----- Initialize classifiers ------
        
        # Setup main classifier module
        cls_module = self.get_classifier_module()
        log.info('Registered classifier.')
        n_cls_labels = n_labels if self.ctrl_class_idx is None else n_labels - 1
        self.classifier = cls_module(
            n_latent,
            n_labels=n_cls_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            use_learnable_temperature=use_learnable_temperature,
            **self.cls_parameters,
        )

        # Setup external embedding classifier
        if self.has_align_cls and self.class_embedding is not None:
            log.info('Registered external embedding aligner.')
            base_aligner_parameters = {
                'n_input': n_latent,
                'class_embed_dim': ext_class_embed_dim,
                'use_batch_norm': use_batch_norm_encoder,
                'use_layer_norm': use_layer_norm_encoder,
                'use_learnable_temperature': use_learnable_temperature,
                'logits': True,
                'return_latents': True
            }
            # Add hyperparameters and ensure base parameters stay constant
            aligner_parameters.update(base_aligner_parameters)
            # Save as class parameters
            self.aligner_parameters = aligner_parameters
            # Register aligner
            self.ext_emb_classifier = EmbeddingAligner(**self.aligner_parameters)

        # Initialize an adversial context (batch/cell type or line) classifier
        if self.use_adversial_context_cls:
            log.info('Registered adversial context classifier.')
            self.context_classifier = Classifier(
                n_input=n_latent,
                n_hidden=context_classifier_n_hidden,
                n_labels=n_batch,
                n_layers=context_classifier_layers,
                dropout_rate=0.1,
                logits=True,
            )
        # ----- Debug -----
        self._step = 0

    def _update_cls_params(self):
        """Update classifier parameters to fit model."""
        cls_parameters = {
            'n_layers': 0 if self.linear_classifier else self.classifier_parameters.get('n_layers', 1),
            'n_hidden': 0 if self.linear_classifier else self.classifier_parameters.get('n_hidden', 128),
            'dropout_rate': self.classifier_parameters.get('dropout_rate', 0.1),
            'logits': True,         # Logits are required for this model
            'return_latents': True
        }
        cls_parameters.update(self.classifier_parameters)
        self.cls_parameters = cls_parameters

    def has_classifier_latent(self) -> bool:
        return self.cls_module_type == 'arc'

    def get_classifier_module(self) -> Classifier | ArcClassifier:
        if self.cls_module_type == 'standard':
            # Return a basic classifier module
            return Classifier
        elif self.cls_module_type == 'arc':
            # Return an ArcClassifier
            return ArcClassifier
        else:
            raise ValueError(f'Unrecognized classification loss strategy: {self.cls_module_type}, choose one of {self._cls_module_types}')

    def set_classification_module_type(self, classification_module_type) -> None:
        """Set classification module type."""
        # Only set it if it's an allowed option
        if classification_module_type not in self._cls_module_types:
            raise ValueError(f'Unrecognized classification loss strategy: {classification_module_type}, choose one of {self._cls_module_types}')
        # Set option
        self.cls_module_type: str = classification_module_type

    def set_cls_loss_strategy(self, cls_loss_strategy) -> None:
        """Set classification loss strategy."""
        # Only set it if it's an allowed option
        if cls_loss_strategy not in self._cls_loss_strategies:
            raise ValueError(f'Unrecognized classification loss strategy: {cls_loss_strategy}, choose one of {self._cls_loss_strategies}')
        # Set option
        self.cls_loss_strategy: str = cls_loss_strategy

    def set_align_ext_emb_strategies(self, align_ext_emb_loss_strategy) -> None:
        """Set alignment strategies."""
        # Set to None to disable alignment strategies
        if align_ext_emb_loss_strategy is None:
            self.align_ext_emb_loss_strategies = None
            self.has_align_cls = False
            return
        # Make sure the resulting type is a list
        self.align_ext_emb_loss_strategies: list[str] = align_ext_emb_loss_strategy if isinstance(align_ext_emb_loss_strategy, list) else [align_ext_emb_loss_strategy]
        # Check for allowed options
        for align_strategy in self.align_ext_emb_loss_strategies:
            if align_strategy not in self._align_ext_emb_loss_strategies:
                raise ValueError(f'Unrecognized alignment strategy: {align_strategy}, choose one of {self._align_ext_emb_loss_strategies}')
        # Set alignment flag to true
        self.has_align_cls = True
    
    @property
    def contr_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.contr_logit_scale.clamp(-1, 4.6).exp()
        else:
            return self.contrastive_temperature
    
    @property
    def clip_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.clip_logit_scale.clamp(-1, 4.6).exp()
        else:
            return self.contrastive_temperature
        
    @property
    def proxy_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.proxy_logit_scale.clamp(-1, 4.6).exp()
        else:
            return self.contrastive_temperature
        
    def freeze_vae_base(self, modules: list[str] = ['z_encoder', 'decoder']) -> None:
        """Freeze modules."""
        for module_key in modules:
            self.freeze_module(module_key)
        
    def freeze_module(self, key: str) -> None:
        """Freeze a given module like encoder or decoder."""
        module = getattr(self, key)
        # Can't freeze what we don't have
        if module is None or not isinstance(module, torch.nn.Module):
            log.warning(f'{module} is not a valid module, skipped freeze.')
            return
        # Set module to eval mode
        module.eval()
        # Disable gradient flow for all module parameters
        for param in module.parameters():
            param.requires_grad = False
        log.info(f'Successfully froze {key} parameters.')
        # Label the module as frozen
        module.frozen = True

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
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
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

        # Inflate context embedding
        if self.use_context_emb:
            if self.use_ext_emb:
                # Pass full embedding to encoder
                context_emb = self.e_context_emb.weight
            else:
                context_emb = self.e_context_emb(batch_index).reshape(batch_index.shape[0], -1)
        else:
            context_emb = None

        # Perform forward pass through encoder
        qz, z = self.z_encoder(encoder_input, *categorical_input, g=g, context_emb=context_emb)

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

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
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

        # TODO: refactor forward function to not rely on y
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

        # Inflate context embedding
        if self.use_context_emb:
            context_emb = self.d_context_emb(batch_index).reshape(batch_index.shape[0], -1)
        else:
            context_emb = None

        # Perform decoder forward pass
        if self.decode_y:
            # Conditional VAE (CVAE)
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
                context_emb=context_emb
            )
        else:
            # Regular VAE
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                context_emb=context_emb
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
        labels: torch.Tensor | None = None,
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
        class_embeds
            Tensor of shape ``(n_labels, embed_dim)``. None to use internal embeddings, external are used if not None

        Returns
        -------
        Tensor of shape ``(n_obs, n_labels)`` denoting logit scores per label.
        Before v1.1, this method by default returned probabilities per label,
        see #2301 for more details.
        """
        # Try caching inference
        if inference_outputs is None:
            inference_outputs = self.inference(x, batch_index, g, cont_covs, cat_covs)
        qz = inference_outputs.get(MODULE_KEYS.QZ_KEY)
        z = inference_outputs.get(MODULE_KEYS.Z_KEY)
        # Reproduce the distribution object from individual mean and variance
        if qz is None:
            qzm = inference_outputs.get(MODULE_KEYS.QZM_KEY)
            qzv = inference_outputs.get(MODULE_KEYS.QZV_KEY)
            qz = torch.distributions.Normal(qzm, qzv.sqrt())
        # Sample from distribution to get z
        if z is None:
            z = self.z_encoder.z_transformation(qz.rsample())
        # Replace sampled latent space with distribution means
        z = qz.loc if use_posterior_mean else z

        # Classify based on x and class external embeddings if provided
        return self.classifier(z, labels=labels)
    
    def align(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        class_embeds: torch.Tensor | None = None,
        inference_outputs: dict[str, torch.Tensor] | None = None,
        noise_sigma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_align_cls:
            raise ValueError('Module is missing an alignment classifier.')
        # Do an additional forward pass if we can't cache it
        if inference_outputs is None:
            inference_outputs = self.inference(x, batch_index, g, cont_covs, cat_covs)
        # Extract latent distribution and sampled latent space
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        # Use z posterior distribution mean instead of sampled latent space
        z = qz.loc if use_posterior_mean else z

        # Forward pass through external embedding classifier
        return self.ext_emb_classifier(z, class_embeds=class_embeds, noise_sigma=noise_sigma)

    def focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        alpha: torch.Tensor | None = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        logits: Tensor of shape (batch_size, num_classes)
        targets: Tensor of shape (batch_size,) with class indices
        alpha: Optional weighting tensor of shape (num_classes,)
        gamma: Focusing parameter
        reduction: 'mean', 'sum', 'batchmean'
        """
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')  # per-sample loss
        pt = torch.exp(-ce_loss)  # pt = softmax prob of correct class
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        # Apply loss reduction
        if reduction is None or reduction == 'none':
            return focal_loss
        elif reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'batchmean':
            return focal_loss.sum(-1).mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f'Unexpected reduction strategy: {reduction}')
    
    def adversarial_context_loss(
            self,
            z: torch.Tensor,
            context_labels: torch.Tensor,
            lambda_adv: float = 1.0,
            reduction: str = 'mean',
        ) -> torch.Tensor:
        """
        Adversarial context loss: encourage encoder to remove context information.
        The gradient is reversed before passing through the context classifier.
        """
        if not getattr(self, 'context_classifier', False):
            return torch.tensor(1e-5, device=z.device)

        # Reverse gradient before classification
        z_rev = GradientReversalFn.apply(z, lambda_adv)

        # Predict context
        context_logits = self.context_classifier(z_rev)

        # Set labels
        if len(context_labels.shape) > 1:
            # Reshape labels to 1d array
            context_labels = context_labels.reshape(-1)

        # Normal cross-entropy loss for the context head
        loss = F.cross_entropy(context_logits, context_labels, reduction=reduction)
        return loss

    def _sym_contrastive_classification_loss(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        cls_emb: torch.Tensor,
        reduction: str = 'mean',
        lambda_proxy: float = 0.0,
        lambda_energy: float = 0.0,
        proxy_margin: float = 0.1,
        margin: float = 0.1,
        return_loss_dict: bool = True,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        CLIP-style symmetric contrastive class embedding loss (Radford et al., 2021)
        Parameters
        ----------
        z: (batch, d) latent vectors
        logits: (batch, num_classes) cosine similarity from z to class embeddings
        y: (batch,) true class indices
        cls_emb: (num_classes, d) fixed or learnable class embeddings
        reduction: (1,) loss reduction strategy
        
        Returns
        -------
        loss: (batch,) | (1,)
        """
        # Get model temperature
        T = self.clip_temp
        # Set reduction method
        if reduction == 'batchmean':
            red_fn = batchmean
        elif reduction == 'mean':
            red_fn = torch.mean
        else:
            red_fn = torch.sum
        # Normalize
        z = F.normalize(z, dim=-1)
        cls_emb = F.normalize(cls_emb, dim=-1)

        # Latent -> class loss, logits are already scaled by classifier
        loss_z2c = F.cross_entropy(logits, y, reduction='none')

        # For class -> latent, restrict to the classes present in the batch
        chosen = cls_emb[y]   # (batch, d)
        logits_c2z = (chosen @ z.T) / T # (batch, batch)
        # Each class embedding should match its corresponding latent (diagonal)
        labels_c2z = torch.arange(z.size(0), device=z.device)
        loss_c2z = F.cross_entropy(logits_c2z, labels_c2z, reduction='none')

        # Symmetric loss per sample
        clip_loss_per_sample = 0.5 * (loss_z2c + loss_c2z)
        clip_loss = red_fn(clip_loss_per_sample)
        loss_z2c_red = red_fn(loss_z2c)
        loss_c2z_red = red_fn(loss_c2z)

        # Use Proxy Anchor
        if lambda_proxy > 0:
            y_oh = F.one_hot(y, num_classes=cls_emb.size(0)).float()
            # Positive and negative anchors
            pos_term = torch.log1p(torch.exp(-self.proxy_temp * (logits[y_oh.bool()] - proxy_margin))).mean()
            neg_term = torch.log1p(torch.exp(self.proxy_temp * (logits[~y_oh.bool()] + proxy_margin))).mean()
            loss_proxy = pos_term + neg_term
        else:
            loss_proxy = torch.tensor(0.0)

        # Energy-based ranking TODO: debug assertion error in `cls_emb[~pos_mask]`
        if lambda_energy > 0:
            pos_sim = torch.sum(z * chosen, dim=-1)  # (B,), How similar is each sample to its embedding overall
            pos_mask = (y==torch.arange(cls_emb.shape[0], device=y.device).reshape(-1, 1)).sum(-1)
            neg_sim, _ = (z @ cls_emb[~pos_mask].T).max(dim=-1)     # Hardest negative per sample
            energy_loss_per_sample = F.softplus(margin - pos_sim + neg_sim).mean(-1)
            energy_loss = red_fn(energy_loss_per_sample)
        else:
            energy_loss = torch.tensor(0.0)

        # Combine final clip loss
        loss = clip_loss \
           + lambda_proxy * loss_proxy \
           + lambda_energy * energy_loss

        # Return full loss dict
        if return_loss_dict:
            return loss, {'clip_z2c': loss_z2c_red, 'clip_c2z': loss_c2z_red, 'proxy': loss_proxy, 'energy': energy_loss}
        else:
            return loss
        
    def _kl_classification_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        class_sim: torch.Tensor,
        target_scale: float = 2.0,
        teacher_T: float = 0.1,
        scale_by_temp: bool = True,
        observed_only: bool = False,
        reduction: str = 'batchmean',
        **kwargs
    ) -> torch.Tensor:
        # Select soft targets for each label
        cls_sim_weight = class_sim / class_sim.max()
        soft_targets = cls_sim_weight[y]
        # Focus on observed classes only
        if observed_only:
            y_labels = y.unique()
            soft_targets = soft_targets[:,y_labels]
            logits = logits[:,y_labels]
        # Re-scale class similarity to push secondary targets up
        if target_scale > 0:
            soft_targets = rescale_targets(soft_targets, scale=target_scale)
        # Apply softmax with temperature scaling to highlight top peaks
        soft_targets = F.softmax(soft_targets / teacher_T, dim=-1)
        soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-12)
        # Student logits with same temperature scaling
        sm_logits = F.log_softmax(logits, dim=-1)
        # KL divergence
        kl_loss = F.kl_div(sm_logits, soft_targets, reduction='none')
        # Scale by T^2 (distillation correction)
        if scale_by_temp:
            kl_loss = kl_loss / teacher_T**2
        # KL-loss is defined as a sum over probabilites
        if reduction is None or reduction == 'none':
            return kl_loss
        # Sum over classes, if batchmean the mean is taken afterwards
        elif reduction == 'sum' or reduction == 'batchmean':
            return kl_loss.sum(-1)
        else:
            return kl_loss.mean(-1)
        
    def _calc_alignment_loss(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        cls_emb: torch.Tensor,
        reduction: str = 'mean',
        **kwargs
    ) -> dict[str, torch.Tensor]:
        # Collect all alignment losses
        align_loss = torch.tensor(0.0)
        loss_dict = {}
        for align_strategy in self.align_ext_emb_loss_strategies:
            # Determine which loss to use
            if align_strategy == 'kl':
                # Calculate KL divergence from class embedding
                _cls_emb = F.normalize(cls_emb, dim=-1)
                cls_sim = _cls_emb @ _cls_emb.T
                _align_loss = self._kl_classification_loss(
                    logits=logits,
                    y=y, 
                    class_sim=cls_sim, 
                    reduction=reduction,
                    **kwargs
                )
            elif align_strategy == 'clip':
                # Calculate CLIP loss
                _align_loss, _loss_dict = self._sym_contrastive_classification_loss(
                    z=z, 
                    logits=logits,
                    y=y, 
                    cls_emb=cls_emb, 
                    reduction=reduction, 
                    return_loss_dict=True,
                    **kwargs
                )
                # Add individual loss parts to detailed log
                loss_dict.update(_loss_dict)
            else:
                raise ValueError(f'Invalid alignment strategy {align_strategy}')
            # Combine losses
            align_loss = align_loss + _align_loss
        # Take mean over classification losses if multiple are provided
        n_strategies = len(self.align_ext_emb_loss_strategies)
        if n_strategies > 1:
            align_loss = align_loss / n_strategies
        # Apply reduction over the batch to produce a single value
        if reduction == 'batchmean':
            align_loss = align_loss.sum(-1) / align_loss.shape[0]
        if reduction == 'mean':
            align_loss = align_loss.mean()
        elif reduction == 'sum':
            align_loss = align_loss.sum()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        # Add final loss
        loss_dict[LOSS_KEYS.ALIGN_LOSS] = align_loss
        return loss_dict
    
    def _calc_classification_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        # Get class weights
        cw = self.class_weights.to(device=y.device) if self.class_weights is not None else None
        # Collect classification loss
        if self.cls_loss_strategy == 'ce':
            ce_loss = self.focal_loss(logits, y, alpha=cw, gamma=1.0, reduction=reduction)
        else:
            ce_loss = self.focal_loss(logits, y, alpha=cw, gamma=self.focal_gamma, reduction=reduction)
        # Return classification loss
        return ce_loss

    @auto_move_data
    def classification_loss(
        self, 
        labelled_dataset: dict[str, torch.Tensor], 
        inference_outputs: dict[str, torch.Tensor] | None = None,
        use_posterior_mean: bool = True,
        cls_emb: torch.Tensor | None = None,
        alignment_loss_weight: float | None = None,
        reduction: str = 'mean',
        ctrl_mask: torch.Tensor | None = None,
        noise_sigma: float | None = 1e-2,
        **kwargs
    ) -> dict[str, dict[str, torch.Tensor]]:
        # Unpack batch
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        # Get gene embedding if given
        g = labelled_dataset.get(REGISTRY_KEYS.GENE_EMB_KEY)
        # Get continous covariates
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None
        # Get categorical covariates
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        
        # Remove control batch for classification loss
        if not self.use_classification_control and ctrl_mask is not None:
            # Classify non-control cells only
            _x = None
            _y = y[~ctrl_mask,:]
            _no_ctrl_idx = torch.arange(cls_emb.shape[0])!=self.ctrl_class_idx
            _cls_emb = cls_emb[_no_ctrl_idx,:]
            # Build subset of inference outputs as to not create second gradient branch
            _z = inference_outputs[MODULE_KEYS.Z_KEY][~ctrl_mask,:]
            qz: torch.distributions.Normal = inference_outputs[MODULE_KEYS.QZ_KEY]
            _qz = torch.distributions.Normal(qz.mean[~ctrl_mask], qz.stddev[~ctrl_mask])
            _inference_outputs = {
                MODULE_KEYS.Z_KEY: _z,
                MODULE_KEYS.QZ_KEY: _qz,
            }
            _batch_idx = batch_idx[~ctrl_mask]
            _cont_covs = cont_covs[~ctrl_mask,:] if cont_covs is not None else None
            _cat_covs = cat_covs[~ctrl_mask,:] if cat_covs is not None else None
        else:
            # Use pre-computed inference and default params
            _x = x
            _y = y
            _z = inference_outputs[MODULE_KEYS.Z_KEY]
            _cls_emb = cls_emb
            _inference_outputs = inference_outputs
            _batch_idx = batch_idx
            _cont_covs = cont_covs
            _cat_covs = cat_covs
        # Reshape y
        _y = _y.view(-1).long()

        # Main classification module forward pass
        cls_output = self.classify(
            _x, 
            batch_index=_batch_idx, 
            g=g,
            cat_covs=_cat_covs, 
            cont_covs=_cont_covs, 
            use_posterior_mean=use_posterior_mean,
            inference_outputs=_inference_outputs,
            labels=_y,
        )  # (n_obs, n_labels)

        # Handle outputs from different main classifiers
        if self.cls_module_type == 'arc':
            # Unpack embedding projections
            logits, cz, W = cls_output
        else:
            # Set embeddings to z, TODO: set to None or empty tensors?
            logits = cls_output
            cz = None
            W = None
        # Collect all classification and alignment losses
        loss_dict = {}
        # Collect all classification-based data results
        data_dict = {
            LOSS_KEYS.LOGITS: logits,       # Main classification logit output
            LOSS_KEYS.Y: _y,                    # Add true labels
            LOSS_KEYS.CZ: cz,               # z latent space, can be updated by arc
            LOSS_KEYS.W: W,                     # Arc learned class embedding weights
        }
        # Calculate classification loss from logits and true labels
        ce_loss = self._calc_classification_loss(
            logits=logits, 
            y=_y, 
            reduction=reduction, 
        )
        # Add classification loss to loss dict
        loss_dict[LOSS_KEYS.CLS_LOSS] = ce_loss

        # Calculate alignment between z and external class embedding space if it is available and enabled
        if alignment_loss_weight is not None and alignment_loss_weight > 0 and cls_emb is not None and self.has_align_cls:
            # Alignment module forward pass
            align_logits, z2c, c2z = self.align(
                x=_x, 
                g=g, 
                cont_covs=_cont_covs, 
                cat_covs=_cat_covs, 
                use_posterior_mean=use_posterior_mean, 
                class_embeds=_cls_emb, 
                inference_outputs=_inference_outputs, 
                noise_sigma=noise_sigma
            )
            # Add aligment latent spaces to output
            data_dict[LOSS_KEYS.Z2C] = z2c          # Latent to external class embedding space projection
            data_dict[LOSS_KEYS.C2Z] = c2z          # External class embedding space to latent projection
            # Calculate alignment loss
            align_loss_dict = self._calc_alignment_loss(
                z=z2c,
                logits=align_logits,
                y=_y,
                cls_emb=c2z,
                reduction=reduction,
                **kwargs
            )
            # Add alignment losses to loss dict
            loss_dict.update(align_loss_dict)
        # Return nested dictionary with losses and associated data
        return {
            LOSS_KEYS.LOSS: loss_dict, LOSS_KEYS.DATA: data_dict
        }

    # L regularization function
    def l_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        l1_norm, l2_norm = 0.0, 0.0
        masks = self.l_mask if isinstance(self.l_mask, (list, tuple)) else [self.l_mask]
        # Collect number of parameters
        for name, param in self.named_parameters():
            if 'bias' in name:
                continue
            if self.l_mask is None or any(mask in name for mask in masks):
                l1_norm += torch.sum(torch.abs(param))
                l2_norm += torch.sum(param ** 2)
        return l1_norm, l2_norm

    def _contrastive_loss_context(
        self,
        z: torch.Tensor,
        labelled_tensors: dict[str, torch.Tensor],
        reduction: str = 'mean',
        scale_by_temperature: bool = False,
        use_context_mask: bool = False,
        ctrl_mask: torch.Tensor | None = None,
        ctrl_scale: float | None = None,
        lambda_neg: float | None = 0.2,
        **kwargs,
    ) -> torch.Tensor:
        """
        Contrastive loss across contexts and classes to promote context invariance.
        Uses REGISTRY_KEYS.BATCH_KEY to identify context IDs.
        Uses REGISTRY_KEYS.LABELS_KEY to identify class IDs.

        Positive pairs: same class but from different contexts.
        Negative pairs: all cells from different classes.
        """
        # Get model temperature
        temperature = self.contr_temp
        # Get labels and contexts from batch
        labels = labelled_tensors[REGISTRY_KEYS.LABELS_KEY]
        contexts = labelled_tensors[REGISTRY_KEYS.BATCH_KEY]

        # Normalize embeddings
        z = F.normalize(z, dim=-1)

        # Cosine similarity matrix
        logits = torch.matmul(z, z.T) / temperature
        exp_logits = torch.exp(logits)

        # Class mask
        same_class = (labels.view(-1, 1) == labels.view(1, -1)).float()
        # Context mask
        same_context = (contexts.view(-1, 1) == contexts.view(1, -1)).float()
        # Self mask
        self_mask = torch.eye(len(labels), device=z.device)

        # Positives: same class but not same cells
        pos_mask = same_class * (1 - self_mask)
        # Positives: Only enforce different contexts if classes have multiple contexts available
        if use_context_mask:
            # Get number of contexts per class
            n_contexts_per_class = torch.zeros(labels.max() + 1, device=z.device)
            for label in labels.unique():
                n_contexts_per_class[label] = contexts[labels == label].unique().numel()
            # Only filter different contexts for classes with multiple contexts
            has_multiple_contexts = n_contexts_per_class[labels] > 1
            pos_mask = pos_mask * (1 - same_context * has_multiple_contexts.unsqueeze(1))
        
        # Use control cells as negatives only, don't try to pull control cells together
        if ctrl_mask is not None:
            pos_mask[ctrl_mask, :] = 0.0
            if ctrl_scale is not None:
                # Weight difference to control cells more heavily than others
                neg_weights = torch.ones_like(exp_logits)
                neg_weights[:, ctrl_mask] = ctrl_scale          # different contribution for control columns
                exp_logits = exp_logits * neg_weights

        # Denominator: all but self
        denom_mask = 1 - self_mask
        # Contrastive loss
        denom = (exp_logits * denom_mask).sum(dim=1, keepdim=True)
        log_probs = logits - torch.log(denom + 1e-12)
        mean_log_prob_pos = (log_probs * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-12)
        
        # Final loss is negative log likelihood
        loss = -mean_log_prob_pos
        # Remove control 0s
        loss = loss[loss>0]

        # --- Optional explicit negative term ---
        if lambda_neg is not None and lambda_neg > 0:
            neg_loss = 0.0
            neg_mask = (1 - same_class) * (1 - self_mask)
            # Leave out control cells for this loss
            if ctrl_mask is not None:
                neg_mask_ctrl = neg_mask * ctrl_mask.float().unsqueeze(0)
                sim_neg = logits * neg_mask_ctrl
            else:
                sim_neg = logits * neg_mask
            # Only keep valid pairs
            valid_neg = sim_neg[sim_neg != 0]
            if valid_neg.numel() > 0:
                neg_loss = F.logsigmoid(-valid_neg).mean()
            # Combine
            loss = loss + lambda_neg * (-neg_loss)

        # Optional gradient rescaling — counteracts ∂L/∂z ∝ 1/τ
        if scale_by_temperature:
            loss *= temperature
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'batchmean':
            return loss.sum(-1).mean()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        
    def _contrastive_loss(
            self,
            z: torch.Tensor,
            labelled_tensors: dict[str, torch.Tensor],
            temperature: float = 1,
            reduction: str = 'mean',
            scale_by_temperature: bool = False,
            eps: float = 1e-12,
            cls_sim: torch.Tensor | None = None,
            ctrl_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Calculate contrastive loss as defined by Khosla et al., 2020.

        This function computes the contrastive loss for a batch of embeddings 
        and their corresponding labels. The loss encourages embeddings of 
        similar labels to be closer in the latent space while pushing apart 
        embeddings of different labels.

        Args:
            labelled_tensors (dict[str, torch.Tensor]): 
                A dictionary containing:
                    - `MODULE_KEYS.Z_KEY`: A tensor of shape (n_obs, n_vars) 
                        representing the latent space embeddings.
                    - `REGISTRY_KEYS.LABELS_KEY`: A tensor of shape (n_obs,) 
                        containing the labels for each observation.
            temperature (float, optional): 
                A scaling factor for the cosine similarity logits. Default is 0.1.
            reduction (str, optional): 
                Specifies the reduction to apply to the output. 
                Options are 'sum' or 'mean'. Default is 'mean' since its not probability-based.
            scale_by_temperature (bool, optional):
                According to original paper, the gradient of this loss inversely scales with temperature, so this should counteract that.

        Returns:
            torch.Tensor: The computed contrastive loss. If `reduction` is 'sum', 
            the loss is a scalar. If `reduction` is 'mean', the loss is averaged 
            over the batch.
        """
        y = labelled_tensors[REGISTRY_KEYS.LABELS_KEY]  # (n_obs,)
        # Step 1: Normalize embeddings to unit hypersphere (‖z‖ = 1), required for cosine similarity
        z = F.normalize(z, dim=-1)
        # Step 2: Compute cosine similarities between all pairs: z_i • z_j
        # Then divide by temperature τ to control sharpness of softmax
        logits = torch.matmul(z, z.T) / temperature  # shape (n_obs, n_obs)
        # Step 3: Exponentiate similarities for numerator/denominator of softmax
        exp_logits = torch.exp(logits)
        # Step 4: Create positive pair mask P(i): entries where label_i == label_j, but i ≠ j
        y = y.view(-1, 1)  # shape (n_obs, 1) to allow broadcasting
        pos_mask = (y == y.T).float()  # 1 if same class, 0 otherwise
        self_mask = torch.eye(pos_mask.size(0), device=pos_mask.device)  # mask out i == j (diagonal)
        pos_mask = pos_mask * (1 - self_mask)  # now only i ≠ j positives are kept
        if ctrl_mask is not None:
            # Step 4b: Remove control cells from positive relationships
            # (they can't be positive anchors or positive partners)
            pos_mask[ctrl_mask, :] = 0.0  # control anchors --> no positives
            pos_mask[:, ctrl_mask] = 0.0  # no one can treat controls as positives
        # Step 5: Construct mask for denominator A(i): all indices ≠ i (exclude self)
        logits_mask = 1 - self_mask  # mask with 0s on diagonal, 1s elsewhere
        # Step 6: Weight negatives by class embedding distances if provided
        if cls_sim is not None:
            weights = cls_sim[y.squeeze()][:, y.squeeze()]  # (n, n)
            weights = weights / (weights.max() + eps)  # normalize to [0,1]
            # Apply weights to exp logits
            exp_logits = (exp_logits * weights)
        # Step 7: Compute softmax denominator: ∑_{a ∈ A(i)} exp(z_i • z_a / τ)
        denom = (exp_logits * logits_mask).sum(dim=1, keepdim=True)  # shape (n_obs, 1)
        # Step 8: Compute log-softmax for each pair: log( exp(z_i • z_p / τ) / denom )
        log_probs = logits - torch.log(denom + eps)  # shape (n_obs, n_obs)
        # Each row i now contains log-probability log(p_ij) for all j ≠ i
        # Step 9: For each anchor i, compute mean log-prob over all positives p ∈ P(i)
        # This is the key part of Eq. (2): average log-prob across multiple positives
        mean_log_prob_pos = (log_probs * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + eps)
        # Step 10: Compute final loss: negative mean log-probability
        loss = -mean_log_prob_pos  # shape (n_obs,)
        if ctrl_mask is not None:
            # Step 11: Zero out loss for control anchors (no positive term)
            loss = loss.masked_fill(ctrl_mask, 0.0)
        # Step 12: Optional gradient rescaling — counteracts ∂L/∂z ∝ 1/τ
        if scale_by_temperature:
            loss *= temperature
        # Step 13: Reduce across batch
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'batchmean':
            return loss.sum(-1).mean()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        contrastive_loss_weight: float | None = None,
        classification_ratio: float | None = None,
        alignment_loss_weight: float | None = None,
        target_scale: float = 4.0,
        adversarial_context_lambda: float = 0.1,
        use_posterior_mean: bool = True,
        use_ext_emb: bool = True,
        observed_only: bool = False,
        n_negatives: int | None = None,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # Unpack input tensor
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        b: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]
        l: torch.Tensor = tensors[REGISTRY_KEYS.LABELS_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]

        # Get external class embedding if available
        cls_emb = None
        if use_ext_emb and self.class_embedding is not None:
            # Filter class embedding for batch classes or not
            _l = l if observed_only else None
            # Get class embedding and similarities
            cls_emb = self.class_embedding(labels=_l, n_negatives=n_negatives, device=x.device)
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

        # Use full batch as default
        ctrl_mask = None
        n_obs_minibatch = x.shape[0]
        # Handle control cells for elbo loss
        if self.ctrl_class_idx is not None:
            # Calculate control mask
            ctrl_mask = (l == self.ctrl_class_idx).reshape(-1)
            # Check if there are any control cells in the batch
            if ctrl_mask.sum() > 0:
                # Calculate fraction of control cells in batch
                ctrl_frac = x.shape[0] / max(ctrl_mask.sum(), 1)    # avoid div/0
                # Divide elbo sum by class cells only
                n_obs_minibatch = (~ctrl_mask).sum()
                # Disregard elbo for control cells
                if not self.use_reconstruction_control and not self.use_kl_control:
                    reconst_loss[ctrl_mask] = 0.0
                    kl_divergence_z[ctrl_mask] = 0.0
                elif not self.use_reconstruction_control:
                    reconst_loss[ctrl_mask] = 0.0
                    kl_weight = kl_weight * ctrl_frac
                elif not self.use_kl_control:
                    kl_divergence_z[ctrl_mask] = 0.0
                    rl_weight = rl_weight * ctrl_frac
                else:
                    # Use full batch
                    n_obs_minibatch = x.shape[0]
        
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
            'true_labels': l,
            'n_obs_minibatch': n_obs_minibatch
        }
        # Create extra metric container
        extra_metrics = {MODULE_KEYS.KL_Z_PER_LATENT_KEY: kl_div_per_latent}

        # Add supervised contrastive loss on z if it is specified
        if contrastive_loss_weight is not None and contrastive_loss_weight > 0:
            contr_loss = self._contrastive_loss_context(
                z, 
                tensors,
                reduction=self.non_elbo_reduction,
                ctrl_mask=ctrl_mask,
                **kwargs
            )
            total_loss = total_loss + contr_loss * contrastive_loss_weight
            extra_metrics['contrastive_loss'] = contr_loss
        # Add classification loss on z if specified
        if classification_ratio is not None and classification_ratio > 0:
            cls_dict: dict[str, dict[str, torch.Tensor]] = self.classification_loss(
                labelled_dataset=tensors, 
                inference_outputs=inference_outputs,
                use_posterior_mean=use_posterior_mean,
                cls_emb=cls_emb,
                alignment_loss_weight=alignment_loss_weight,
                target_scale=target_scale,
                ctrl_mask=ctrl_mask,
                reduction=self.non_elbo_reduction,
                **kwargs
            )
            # Unpack classification loss results
            ce_loss_dict = cls_dict.pop(LOSS_KEYS.LOSS)
            ce_loss = ce_loss_dict.pop(LOSS_KEYS.CLS_LOSS)
            align_loss = ce_loss_dict.get(LOSS_KEYS.ALIGN_LOSS)
            # Unpack classification data results
            ce_data_dict = cls_dict.pop(LOSS_KEYS.DATA)
            true_labels = ce_data_dict.pop(LOSS_KEYS.Y)
            logits = ce_data_dict.pop(LOSS_KEYS.LOGITS)
            # Add z classification loss to overall loss
            total_loss = total_loss + ce_loss * classification_ratio
            
            # Add classification loss details
            lo_kwargs.update({
                'classification_loss': ce_loss,
                'true_labels': true_labels,
                'logits': logits,
            })
            # Add extra classification losses
            extra_metrics.update(ce_loss_dict)
            # Add alignment loss if weight is not 0
            if alignment_loss_weight is not None and alignment_loss_weight > 0 and align_loss is not None:
                total_loss = total_loss + align_loss * alignment_loss_weight

        # Add L regularizations (L1 and/or L2)
        l1, l2 = self.l_regularization()
        if self.l1_lambda is not None and self.l1_lambda > 0:
            extra_metrics['L1'] = l1
            total_loss = total_loss + l1 * self.l1_lambda
        if self.l2_lambda is not None and self.l2_lambda > 0:
            extra_metrics['L2'] = l2
            total_loss = total_loss + l2 * self.l2_lambda
        # Add adversional context loss if lambda > 0 (should be last to see z as it inverts the gradient flow)
        if adversarial_context_lambda > 0 and self.use_adversial_context_cls:
            _adv_loss = self.adversarial_context_loss(z, context_labels=b, lambda_adv=adversarial_context_lambda, reduction=self.non_elbo_reduction)
            total_loss = total_loss + _adv_loss
            extra_metrics[LOSS_KEYS.ADV_LOSS] = _adv_loss

        # Add to loss output
        lo_kwargs['extra_metrics'] = extra_metrics
 
        # Set total loss
        lo_kwargs[LOSS_KEYS.LOSS] = total_loss
        return LossOutput(**lo_kwargs)

    def on_load(self, model: BaseModelClass):
        manager = model.get_anndata_manager(model.adata, required=True)
        source_version = manager._source_registry[_constants._SCVI_VERSION_KEY]
        version_split = source_version.split('.')

        if int(version_split[0]) >= 1 and int(version_split[1]) >= 1:
            return

        # need this if <1.1 model is resaved with >=1.1 as new registry is
        # updated on setup
        manager.registry[_constants._SCVI_VERSION_KEY] = source_version

        # pre 1.1 logits fix
        model_kwargs = model.init_params_.get('model_kwargs', {})
        cls_params = model_kwargs.get('classifier_parameters', {})
        user_logits = cls_params.get('logits', False)

        if not user_logits:
            self.classifier.logits = False
            self.classifier.classifier.append(torch.nn.Softmax(dim=-1))

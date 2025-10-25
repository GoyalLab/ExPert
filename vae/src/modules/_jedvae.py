from typing import Iterable

from src.modules._base import Classifier, EmbeddingClassifier, Encoder, DecoderSCVI, ExternalClassEmbedding, MemoryQueue
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS
from src.utils.distributions import rescale_targets
from src.utils.common import pearson, Cache, GradientReversalFn

from typing import Iterable

import logging
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


class JEDVAE(VAE):
    """
    Adaption of scVI and scanVI models to predict Perturb-seq perturbations using class-embedding data.
    """
    
    def _update_cls_params(self):
        cls_parameters = {
            'n_layers': 0 if self.linear_classifier else self.classifier_parameters.get('n_layers', 1),
            'n_hidden': 0 if self.linear_classifier else self.classifier_parameters.get('n_hidden', 128),
            'dropout_rate': self.classifier_parameters.get('dropout_rate', 0.1),
            'logits': True,         # Logits are required for this model
            'return_latents': True
        }
        cls_parameters.update(self.classifier_parameters)
        self.cls_parameters = cls_parameters
        

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
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float | None = None,
        use_embedding_classifier: bool = True,
        class_embed_dim: int = 128,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        linear_classifier: bool = False,
        cls_weights: torch.Tensor | None = None,
        classifier_parameters: dict = {},
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        l1_lambda: float | None = 1e-5,
        l2_lambda: float | None = 1e-3,
        l_mask: str | list[str] | None = None,
        focal_gamma: float = 1.0,
        init_t: float = 0.1,
        classification_loss_strategy: Literal['similarity', 'kl', 'focal', 'symmetric_contrastive'] | list[str] = 'kl',
        ctrl_class_idx: int | None = None,
        kl_class_temperature: float = 0.1,
        use_learnable_control_emb: bool = False,
        use_learnable_temperature: bool = True,
        use_adversial_context_cls: bool = True,
        use_reconstruction_control: bool = False,                           # Use control cells for reconstruction loss
        use_kl_control: bool = False,                                       # Use control cells for KL loss
        use_classification_control: bool = False,                           # Try to classify control cells
        use_contrastive_control: bool = True,                               # Use control cells as negatives in contrastive loss
        clip_memory_n: int | None = 0,
        context_classifier_layers: int = 2,
        context_classifier_n_hidden: int = 256,
        contrastive_temperature: float = 0.1,
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        use_feature_mask: bool = False,
        drop_prob: float = 1e-3,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
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
        )

        # Setup l-norm params
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l_mask = l_mask
        
        if dropout_rate_decoder is not None:
            logging.warning('Dropout rate for decoder currently unavailable. Will fall back to 0.')
        
        # Classifier parameters
        self.classifier_parameters = classifier_parameters
        self.linear_classifier = linear_classifier
        self.use_embedding_classifier = use_embedding_classifier
        self.class_weights = cls_weights
        self.class_embed_dim = class_embed_dim
        self._update_cls_params()
        self.focal_gamma = focal_gamma
        self.classification_loss_strategy = classification_loss_strategy
        self.kl_class_temperature = kl_class_temperature
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
            self.contr_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
            self.clip_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
            self.proxy_logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1.0 / init_t)))
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
            logging.info(f'Registered class embedding with shape: {self.class_embedding.shape}, using control: {use_learnable_control_emb}')
        else:
            logging.info(f'No external class embedding registered with model.')
            self.class_embedding = None 

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # Check config
        self.cls_strategies = self.classification_loss_strategy if isinstance(self.classification_loss_strategy, list) else [self.classification_loss_strategy]
        for cls_strategy in self.cls_strategies:
            if cls_strategy not in ['similarity', 'kl', 'focal', 'symmetric_contrastive']:
                raise ValueError(f'Got invalid argument for `classification_loss_strategy` {cls_strategy}')

        # Setup encoder input dimensions
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        # Setup categorical co-variates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = cat_list
        encoder_cat_list = cat_list if encode_covariates else None
        # Setup extra en- and decoder params
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

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
            **_extra_encoder_kwargs,
            return_dist=True,
        )

        # Setup decoder
        self.decoder = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation='softmax',
            inject_covariates=deeply_inject_covariates,
            **_extra_decoder_kwargs,
        )

        # Initialize embedding classifier
        classifier_cls = EmbeddingClassifier if use_embedding_classifier else Classifier
        self.classifier = classifier_cls(
            n_latent,
            n_labels=n_labels,
            class_embed_dim=class_embed_dim,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            use_learnable_temperature=use_learnable_temperature,
            **self.cls_parameters,
        )

        # Initialize an adversial context (batch/cell type or line) classifier
        if self.use_adversial_context_cls:
            self.context_classifier = Classifier(
                n_input=n_latent,
                n_hidden=context_classifier_n_hidden,
                n_labels=n_batch,
                n_layers=context_classifier_layers,
                dropout_rate=0.1,
                logits=True,
            )
        # Initialize memory queue for clip loss
        self.clip_memory_n = clip_memory_n
        if clip_memory_n is not None and clip_memory_n > 0:
            self.clip_memory = MemoryQueue(dim=self.classifier.n_output, n=clip_memory_n)
        else:
            self.clip_memory = None
        # Debug
        self.use_cache = False
        self._step = 0
        self.cache = Cache(10)

    @property
    def contr_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.contr_logit_scale.clamp(0, 4.6).exp()
        else:
            return self.contrastive_temperature
    
    @property
    def clip_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.clip_logit_scale.clamp(0, 4.6).exp()
        else:
            return self.contrastive_temperature
        
    @property
    def proxy_temp(self) -> torch.Tensor:
        # Calculate logit scale
        if self.use_learnable_temperature:
            return 1.0 / self.proxy_logit_scale.clamp(0, 4.6).exp()
        else:
            return self.contrastive_temperature
        
    def freeze_vae_base(self) -> None:
        """Freeze encoder and decoder module"""
        self.freeze_module('z_encoder')
        self.freeze_module('decoder')
        
    def freeze_module(self, key: str) -> None:
        """Freeze a given module like encoder or decoder"""
        module = getattr(self, key)
        # Can't freeze what we don't have
        if module is None or not isinstance(module, torch.nn.Module):
            return
        # Set module to eval mode
        module.eval()
        # Disable gradient flow for all module parameters
        for param in module.parameters():
            param.requires_grad = False
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
        if self.batch_representation == 'embedding' and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input, g=g)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input, g=g)

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
    def classify(
        self,
        x: torch.Tensor | None,
        batch_index: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        class_embeds: torch.Tensor | None = None,
        inference_outputs: dict[str, torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
        noise_sigma: float | None = None,
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
        if inference_outputs is None:
            inference_outputs = self.inference(x, batch_index, g, cont_covs, cat_covs)
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        z = qz.loc if use_posterior_mean else z

        # Classify based on x and class external embeddings if provided
        return self.classifier(z, class_embeds=class_embeds, labels=labels, noise_sigma=noise_sigma)
    
    def _class_similarity_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str | None = 'mean',
        penalize_cls_uncertaintly: bool = True,
        soft: bool = False,
    ) -> torch.Tensor:
        # Get similarity scores for each target class
        cls_mask = targets == torch.arange(logits.shape[1], device=logits.device)
        cls_mask = torch.zeros_like(logits).masked_fill(cls_mask, 1.0).bool()
        cls_sims = logits[cls_mask]
        # Calculate cosine difference of cells and treat that as loss to minimize
        cs_scores = 1 - cls_sims
        # + overall similarity of cell to classes (classification uncertainty)
        if penalize_cls_uncertaintly:
            cs_scores += logits[~cls_mask].mean(dim=-1)
        if reduction is None or reduction == 'none':
            return cs_scores
        elif reduction == 'mean':
            return cs_scores.mean(-1)
        else:
            return cs_scores.sum(-1)

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
        if reduction is None or reduction == 'none':
            return focal_loss
        elif reduction == 'mean':
            return focal_loss.mean(-1)
        else:
            return focal_loss.sum(-1)

    def geom_alignment_loss(
            self, 
            cz: torch.Tensor,
            logits: torch.Tensor,
            cls_emb: torch.Tensor,
            cls_sim: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = 'mean',
            alpha: float = 0.1,
            beta: float = 1.0,
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        cz : (batch, d) - class projection space
        logits: (batch, n_classes) - class logits
        cls_emb : (n_classes, d) - class embeddings
        cls_sim: (n_classes, n_classes) - class embedding similarities
        targets : (batch,) - integer class labels
        reduction: str - loss aggregation strategy
        alpha: float - direct alignment weight
        beta: float - similarity alignment weight

        Returns
        -------
        alignment_loss : scalar Tensor
        """
        # Get target class embeddings
        y = targets.flatten()
        target_emb = cls_emb[y]
        # Compare to predicted class projection
        direct_align_score = F.mse_loss(cz, target_emb, reduction=reduction) if alpha > 0 else 0.0
        # Compare similarities from targets to other classes
        target_sim = cls_sim[y]
        # Calculate pearson correlation
        _cls_sim_corr = 1.0 - pearson(logits, target_sim, dim=-1)
        if reduction is None:
            cls_sim_corr = _cls_sim_corr
        if reduction == 'mean':
            cls_sim_corr = _cls_sim_corr.mean()
        elif reduction == 'sum':
            cls_sim_corr = _cls_sim_corr.sum()
        elif reduction == 'batchmean':
            cls_sim_corr = _cls_sim_corr.sum(-1).mean()
        else:
            raise ValueError(f'Unknown reduction: {reduction}')
        return alpha * direct_align_score + beta * cls_sim_corr
    
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
        
    def cosine_alignment_loss(
            self, 
            z_normalized: torch.Tensor, 
            class_embeds: torch.Tensor, 
            targets: torch.Tensor,
            reduction: str | None = 'sum'
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_normalized : (batch, d) - normalized latent vectors
        class_embeds : (n_classes, d) - normalized class embeddings
        targets : (batch,) - integer class labels
        reduction: str - loss aggregation strategy

        Returns
        -------
        alignment_loss : scalar Tensor
        """
        target_embeds = class_embeds[targets]  # (batch, d)
        target_embeds = F.normalize(target_embeds, dim=-1)
        cos_sim = F.cosine_similarity(z_normalized, target_embeds, dim=-1)  # (batch,)
        cos_diff = (1.0 - cos_sim)

        if reduction is None:
            return cos_diff
        if reduction == 'mean':
            return cos_diff.mean()
        elif reduction == 'sum':
            return cos_diff.sum()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')

    def _cosine_classification_loss(
        self,
        logits: torch.Tensor,             # latent vectors (batch, d)
        y: torch.Tensor,                  # true labels (batch,)
        class_sim: torch.Tensor,          # class embeddings (n_classes, d)
        reduction: str | None = 'mean',
        T: float = 0.1,                   # temperature scaling
    ) -> torch.Tensor:
        # Normalize latent vectors and class embeddings
        z = F.normalize(logits, dim=-1)           # (batch, d)
        c = F.normalize(class_sim, dim=-1)        # (n_classes, d)

        # Similarity logits: cosine similarity scaled by temperature
        sim_logits = torch.matmul(z, c.T) / T     # (batch, n_classes)

        # Cross-entropy with ground-truth labels
        if reduction is None:
            loss = F.cross_entropy(sim_logits, y, reduction='none')
        else:
            loss = F.cross_entropy(sim_logits, y, reduction=reduction)

        return loss
    
    def _clip_loss_z2c(
            self,
            z: torch.Tensor, 
            y: torch.Tensor, 
            logits: torch.Tensor,
        ):
        """
        z: (batch, d)
        y: (batch,)
        logits: (batch, num_classes)
        cls_emb: (num_classes, d)
        T: scalar
        """

        z = F.normalize(z, dim=-1)
        loss_z2c = F.cross_entropy(logits, y, reduction='none')

        # No memory registered, return default loss
        if self.clip_memory is None:
            return loss_z2c
        
        # Get classification temperature
        T_cls = self.classifier.temperature
        # Extend with similarities to queued latents (negatives)
        q_z, q_y = self.clip_memory.get()
        if q_z is not None:
            q_z = F.normalize(q_z, dim=-1)
            logits_queue = (z @ q_z.T) / T_cls
            # Mask same-class queue entries -> don't treat as negatives
            same_class_mask = (y.unsqueeze(1) == q_y.unsqueeze(0)).float()
            logits_queue = logits_queue - 1e9 * same_class_mask  # -inf for same-class queue items

            # Concatenate negatives
            logits = torch.cat([logits, logits_queue], dim=-1)
        # CE vs true class labels
        loss_z2c = F.cross_entropy(logits, y, reduction='none')

        # Update queue (no grad)
        with torch.no_grad():
            self.clip_memory.enqueue(z.detach(), y.detach())

        return loss_z2c

    def _sym_contrastive_classification_loss(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        cls_emb: torch.Tensor,
        reduction: str = 'mean',
        lambda_proxy: float = 0.5,
        lambda_energy: float = 0.3,
        margin: float = 0.1,
        proxy_margin: float = 0.1,
        return_loss_dict: bool = True,
        **kwargs
    ) -> torch.Tensor:
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
        if reduction == 'mean':
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
        y_oh = F.one_hot(y, num_classes=cls_emb.size(0)).float()
        # Positive and negative anchors
        pos_term = torch.log1p(torch.exp(-self.proxy_temp * (logits[y_oh.bool()] - proxy_margin))).mean()
        neg_term = torch.log1p(torch.exp(self.proxy_temp * (logits[~y_oh.bool()] + proxy_margin))).mean()
        loss_proxy = pos_term + neg_term

        # Energy-based ranking
        pos_sim = torch.sum(z * chosen, dim=-1)  # (B,), How similar is each sample to its embedding overall
        pos_mask = (y==torch.arange(cls_emb.shape[0], device=y.device).reshape(-1, 1)).sum(-1)
        neg_sim, _ = (z @ cls_emb[~pos_mask].T).max(dim=-1)     # Hardest negative per sample
        energy_loss_per_sample = F.softplus(margin - pos_sim + neg_sim).mean(-1)
        energy_loss = red_fn(energy_loss_per_sample)

        # Combine final loss
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
    
    def _calc_classification_loss(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        reduction: str = 'mean',
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        target_scale: float = 4.0,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        # Get class weights
        cw = self.class_weights.to(device=y.device) if self.class_weights is not None else None
        # Collect classification losses
        ce_loss = torch.tensor(0.0)
        full_loss_dict = {}
        # These functions all return losses for each cell (b,1)
        for cls_strategy in self.cls_strategies:
            # Determine which loss to use
            if cls_strategy == 'focal':
                _ce_loss = self.focal_loss(logits, y, alpha=cw, gamma=self.focal_gamma)
            elif cls_strategy == 'similarity':
                _ce_loss = self._class_similarity_loss(logits, y)
            elif cls_strategy == 'kl' and cls_emb is not None:
                _ce_loss = self._kl_classification_loss(logits, y, class_sim=cls_sim, target_scale=target_scale, **kwargs)
            elif cls_strategy == 'symmetric_contrastive':
                _ce_loss, _loss_dict = self._sym_contrastive_classification_loss(z=z, logits=logits, y=y, cls_emb=cls_emb, **kwargs)
                full_loss_dict.update(_loss_dict)
            else:
                raise ValueError(f'Invalid classification strategy {cls_strategy}')
            # Combine losses
            ce_loss = ce_loss + _ce_loss
        # Take mean over classification losses if multiple are provided
        if len(self.cls_strategies) > 1:
            ce_loss = ce_loss / len(self.cls_strategies)
        # Apply reduction over the batch to produce a single value
        if reduction == 'mean' or reduction == 'batchmean':
            loss = ce_loss.mean()
        elif reduction == 'sum':
            loss = ce_loss.sum()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        # Return loss dict
        full_loss_dict['ce_loss'] = loss
        return full_loss_dict

    @auto_move_data
    def classification_loss(
        self, 
        labelled_dataset: dict[str, torch.Tensor], 
        inference_outputs: dict[str, torch.Tensor] | None = None,
        use_posterior_mean: bool = True,
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        use_cls_scores: bool = True,
        alignment_loss_weight: float | None = None,
        reduction: str = 'mean',
        target_scale: float = 4.0,
        ctrl_mask: torch.Tensor | None = None,
        noise_sigma: float | None = 1e-2,
        **kwargs
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        # Get gene embedding if given
        g = labelled_dataset.get(REGISTRY_KEYS.GENE_EMB_KEY)

        # Try to get class scores
        cls_scores = labelled_dataset.get(REGISTRY_KEYS.CLS_CERT_KEY) if use_cls_scores else None

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        
        # Remove control batch for classification loss
        if not self.use_classification_control and ctrl_mask is not None:
            # Classify non-control cells only
            _x = None
            _y = y[~ctrl_mask,:]
            _no_ctrl_idx = torch.arange(cls_emb.shape[0])!=self.ctrl_class_idx
            _cls_emb = cls_emb[_no_ctrl_idx,:]
            _cls_sim = cls_sim[_no_ctrl_idx][:, _no_ctrl_idx]
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
            _cls_emb = cls_emb
            _cls_sim = cls_sim
            _inference_outputs = inference_outputs
            _batch_idx = batch_idx
            _cont_covs = cont_covs
            _cat_covs = cat_covs
        # Reshape y
        _y = _y.view(-1).long()

        # Classify
        cls_output = self.classify(
            _x, 
            batch_index=_batch_idx, 
            g=g,
            cat_covs=_cat_covs, 
            cont_covs=_cont_covs, 
            use_posterior_mean=use_posterior_mean,
            class_embeds=_cls_emb,
            inference_outputs=_inference_outputs,
            labels=_y,
            noise_sigma=noise_sigma
        )  # (n_obs, n_labels)
        if self.use_embedding_classifier:
            # Unpack embedding projections
            logits, cz, ez = cls_output
        else:
            # Set embeddings to z, TODO: set to None or empty tensors?
            logits, cz, ez = cls_output, x, x
        # Calculate classification loss from logits and true labels
        ce_loss_dict = self._calc_classification_loss(
            z=cz,
            logits=logits, 
            y=_y, 
            reduction=reduction, 
            cls_emb=ez, 
            cls_sim=_cls_sim,
            cls_scores=cls_scores, 
            target_scale=target_scale,
            ctrl_mask=ctrl_mask,
            **kwargs
        )
        # Calculate alignment between zx and class projection if external embedding is given
        if alignment_loss_weight is not None and alignment_loss_weight > 0 and cls_emb is not None:
            align_loss = self.geom_alignment_loss(
                cz=cz, 
                logits=logits,
                cls_emb=ez, 
                cls_sim=_cls_sim,
                targets=_y,
                reduction=reduction
            )
        else:
            align_loss = None
        return ce_loss_dict, _y, logits, align_loss

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
        classification_ratio: float | None = None,
        alignment_loss_weight: float | None = None,
        contrastive_loss_weight: float | None = None,
        target_scale: float = 4.0,
        adversarial_context_lambda: float = 0.1,
        use_posterior_mean: bool = True,
        use_ext_emb: bool = True,
        use_cls_scores: bool = True,
        observed_only: bool = False,
        n_negatives: int | None = None,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # X inference and generative
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        b: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]
        l: torch.Tensor = tensors[REGISTRY_KEYS.LABELS_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]


        # Get external class embedding if specified
        cls_emb, cls_sim = None, None
        if use_ext_emb and self.class_embedding is not None:
            # Filter class embedding for batch classes or not
            _l = l if observed_only else None
            # Get class embedding and similarities
            cls_emb, cls_sim = self.class_embedding(labels=_l, n_negatives=n_negatives, device=x.device)
        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z_mat = kl_divergence(qz, pz)
        # Calculate reconstruction loss over batch and all features
        reconst_loss_mat = -px.log_prob(x)
        # Aggregate elbo losses over latent dimensions / input features, will get batch normalized internally
        if self.reduction == 'batchmean':
            kl_divergence_z = kl_divergence_z_mat.sum(-1) # (b,1)
            reconst_loss = reconst_loss_mat.sum(-1)
        elif self.reduction == 'mean':
            kl_divergence_z = kl_divergence_z_mat.mean(-1) # (b, 1)
            reconst_loss = reconst_loss_mat.mean(-1)
        else:
            raise ValueError(f'reduction has to be either "batchmean" or "mean", got {self.reduction}')

        # Calculate control mask
        ctrl_mask = (l == self.ctrl_class_idx).reshape(-1)
        # Handle control cells for elbo loss
        if self.ctrl_class_idx is not None and ctrl_mask.sum() > 0:
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
        else:
            # Use full batch
            ctrl_mask = None
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
        extra_metrics = {}

        # Add contrastive loss if it is specified
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
        # Add classification based losses
        if classification_ratio is not None and classification_ratio > 0:
            ce_loss_dict, true_labels, logits, align_loss = self.classification_loss(
                labelled_dataset=tensors, 
                inference_outputs=inference_outputs,
                use_posterior_mean=use_posterior_mean,
                cls_emb=cls_emb,
                cls_sim=cls_sim,
                use_cls_scores=use_cls_scores,
                alignment_loss_weight=alignment_loss_weight,
                target_scale=target_scale,
                ctrl_mask=ctrl_mask,
                reduction=self.non_elbo_reduction,
                **kwargs
            )
            # Get main classification loss
            ce_loss = ce_loss_dict.pop('ce_loss')
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
            # Add alignment loss if it has been calculated
            if align_loss is not None:
                extra_metrics['alignment_loss'] = align_loss
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
            extra_metrics['adversial_context_loss'] = _adv_loss

        # Calculate unweighted sum of all losses
        unscaled_loss = reconst_loss.sum() / n_obs_minibatch + kl_divergence_z.sum() / n_obs_minibatch
        
        # Handle extra metrics if any are given
        if len(extra_metrics) > 0:
            # Add all extra metrics to the loss
            extra_loss_sum = sum(v.detach() for v in extra_metrics.values() if isinstance(v, torch.Tensor) and v.ndim<2)
            unscaled_loss = unscaled_loss + extra_loss_sum
        
        # Add unscaled loss to metrics
        extra_metrics['unscaled_loss'] = unscaled_loss
        # Add to loss output
        lo_kwargs['extra_metrics'] = extra_metrics
 
        # Set total loss
        lo_kwargs['loss'] = total_loss
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

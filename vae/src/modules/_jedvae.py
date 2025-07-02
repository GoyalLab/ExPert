from typing import TYPE_CHECKING, Iterable

from src.models.base import AttentionEncoder
from src.modules._base import EmbeddingClassifier, Encoder, DecoderSCVI
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS

from typing import Iterable

import logging
import torch
import torch.nn.functional as F

from scvi.data import _constants
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
            'n_layers': 0 if self.linear_classifier else self.classifier_parameters.get('n_layers', 10),
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
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float | None = None,
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
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        l1_lambda: float | None = 1e-5,
        use_attention_encoder: bool = False,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        reduction: Literal['mean', 'sum'] = 'sum',
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
            n_layers=n_layers_encoder,
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
            extra_encoder_kwargs=extra_encoder_kwargs,
            extra_decoder_kwargs=extra_decoder_kwargs,
        )

        # Setup scvi part of model
        self.l1_lambda = l1_lambda
        
        if dropout_rate_decoder is not None:
            logging.warning('Dropout rate for decoder currently unavailable. Will fall back to 0.')
        
        # Classifier parameters
        self.classifier_parameters = classifier_parameters
        self.linear_classifier = linear_classifier
        self.class_weights = cls_weights
        self.class_embed_dim = class_embed_dim
        self._update_cls_params()
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.reduction = reduction

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # Setup encoder input dimensions
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = cat_list
        encoder_cat_list = cat_list if encode_covariates else None
        
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

        # Select encoder class
        _encoder = AttentionEncoder if use_attention_encoder else Encoder

        # Re-Init encoder for X (rna-seq)
        self.z_encoder = _encoder(
            n_input=n_input_encoder, 
            n_output=n_latent, 
            n_hidden=n_hidden,
            n_layers=n_layers_encoder, 
            n_cat_list=encoder_cat_list, 
            dropout_rate=dropout_rate_encoder, 
            use_batch_norm=use_batch_norm_encoder, 
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            **_extra_encoder_kwargs,
            return_dist=True,
        )

        self.decoder = DecoderSCVI(
            n_input=n_latent,
            n_output=n_input,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
            **_extra_decoder_kwargs,
        )

        # Initialize embedding classifier
        self.classifier = EmbeddingClassifier(
            n_latent,
            n_labels=n_labels,
            class_embed_dim=class_embed_dim,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **self.cls_parameters,
        )

    @auto_move_data
    def classify(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        class_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        inference_outputs = self.inference(x, batch_index, cont_covs, cat_covs)
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        z = qz.loc if use_posterior_mean else z

        # Classify based on x and class external embeddings if provided
        return self.classifier(z, class_embeds)
    
    def focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        alpha: torch.Tensor | None = None, 
        gamma: float = 2.0, 
        reduction: str ='sum'
    ) -> torch.Tensor:
        """
        logits: Tensor of shape (batch_size, num_classes)
        targets: Tensor of shape (batch_size,) with class indices
        alpha: Optional weighting tensor of shape (num_classes,)
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')  # per-sample loss
        pt = torch.exp(-ce_loss)  # pt = softmax prob of correct class
        focal_loss = ((1 - pt) ** gamma) * ce_loss

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
    def cosine_alignment_loss(
            self, 
            z_normalized: torch.Tensor, 
            class_embeds: torch.Tensor, 
            targets: torch.Tensor,
            reduction: str = 'sum'
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
        if reduction == 'mean':
            return (1.0 - cos_sim).mean()
        elif reduction == 'sum':
            return (1.0 - cos_sim).sum()
        else:
            raise ValueError(f"Reduction has to be either 'sum', or 'mean'")
        
    @auto_move_data
    def classification_loss(
        self, 
        labelled_dataset: dict[str, torch.Tensor], 
        use_posterior_mean: bool = True,
        use_ext_emb: bool = True,
        alignment_loss_weight: float | None = None,
        reduction: str = 'mean',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        ext_emb = labelled_dataset.get(REGISTRY_KEYS.CLS_EMB_KEY) if use_ext_emb else None # (n_labels, n_emb)

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        # Classify
        logits, cz = self.classify(
            x, 
            batch_index=batch_idx, 
            cat_covs=cat_covs, 
            cont_covs=cont_covs, 
            use_posterior_mean=use_posterior_mean,
            class_embeds=ext_emb
        )  # (n_obs, n_labels)
        
        cw = self.class_weights.to(device=y.device) if self.class_weights is not None else None
        if self.use_focal_loss:
            ce_loss = self.focal_loss(
                logits,
                y.view(-1).long(),
                alpha=cw,
                gamma=self.focal_gamma,
                reduction=reduction
            )
        else:
            ce_loss = F.cross_entropy(
                logits,
                y.view(-1).long(),
                weight=cw,
                reduction=reduction
            )
        # Calculate alignment between zx and class projection if external embedding is given
        if alignment_loss_weight is not None and alignment_loss_weight > 0 and ext_emb is not None:
            align_loss = self.cosine_alignment_loss(
                z_normalized=cz, 
                class_embeds=ext_emb, 
                targets=y,
                reduction=reduction
            )
        else:
            align_loss = None
        return ce_loss, y, logits, align_loss

    # L1 regularization function
    def l1_regularization(self):
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm
    
    def _contrastive_loss_custom(
            self,
            labelled_tensors: dict[str, torch.Tensor],
            beta: float | None = None,
            margin: float = 20.0,
            activator: Literal['power2', 'exp', 'exp2'] = 'power2',
            reduction: str = 'sum'
        ) -> torch.Tensor:
        """Calculate contrastive based on pair-wise cell-cell distances"""
        x = labelled_tensors[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        y = labelled_tensors[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        def _activate(_d, act):
            if act == 'power2':
                return _d**2
            elif act == 'exp':
                return torch.exp(_d)
            elif act == 'exp2':
                return torch.exp2(_d)
            else:
                raise ValueError(f'Unkown argument for distance activation: {act}')
        # Calculate cell-wise distances for the batch
        D = torch.cdist(x, x)
        # Set label mask as integers
        mask = (y == y.reshape(-1)).float()
        # Distances of cells within same labels
        s = _activate(D * mask, activator)
        # Distances of cells across different labels
        d = (torch.clamp(margin - D, min=0.0)) * (1 - mask)
        d = _activate(d, activator)
        if beta is not None:
            # Introduce trade-off between similarity and dissimilarity
            loss = (1 - beta) * s + beta * d
        else:
            # Add similarity and dissimilarity one to one
            loss = s + d
        if reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
        
    def _contrastive_loss(
            self,
            z: torch.Tensor,
            labelled_tensors: dict[str, torch.Tensor],
            temperature: float = 0.1,
            reduction: str = 'mean',
            scale_by_temperature: bool = False,
            eps: float = 1e-12
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
                Options are 'sum' or 'mean'. Default is 'sum'.
            scale_by_temperature (bool, optional):
                According to original paper, the gradient of this loss inversely scales with temperature, so this should counteract that.

        Returns:
            torch.Tensor: The computed contrastive loss. If `reduction` is 'sum', 
            the loss is a scalar. If `reduction` is 'mean', the loss is averaged 
            over the batch.
        """
        y = labelled_tensors[REGISTRY_KEYS.LABELS_KEY]  # (n_obs,)
        # Step 1: Normalize embeddings to unit hypersphere (‖z‖ = 1), required for cosine similarity
        z = F.normalize(z, dim=1)
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
        # Step 5: Construct mask for denominator A(i): all indices ≠ i (exclude self)
        logits_mask = 1 - self_mask  # mask with 0s on diagonal, 1s elsewhere
        # Step 5.1: Take only the lower triangular part of the logits_mask
        logits_mask = torch.tril(logits_mask, diagonal=-1)
        # Step 6: Compute softmax denominator: ∑_{a ∈ A(i)} exp(z_i • z_a / τ)
        denom = (exp_logits * logits_mask).sum(dim=1, keepdim=True)  # shape (n_obs, 1)
        # Step 7: Compute log-softmax for each pair: log( exp(z_i • z_p / τ) / denom )
        log_probs = logits - torch.log(denom + eps)  # shape (n_obs, n_obs)
        # Each row i now contains log-probability log(p_ij) for all j ≠ i
        # Step 8: For each anchor i, compute mean log-prob over all positives p ∈ P(i)
        # This is the key part of Eq. (2): average log-prob across multiple positives
        mean_log_prob_pos = (log_probs * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + eps)
        # Step 9: Compute final loss: negative mean log-probability
        loss = -mean_log_prob_pos  # shape (n_obs,)
        # Step 10: Optional gradient rescaling — counteracts ∂L/∂z ∝ 1/τ
        if scale_by_temperature:
            loss *= temperature
        # Step 11: Reduce across batch
        return loss.sum() if reduction == 'sum' else loss.mean()
        
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
        contrastive_temperature: float = 0.1,
        use_posterior_mean: bool = True,
        use_ext_emb: bool = True,
        **kwargs,
    ) -> LossOutput:
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # X inference and generative
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        
        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z_mat = kl_divergence(qz, pz)
        # Calculate reconstruction loss over batch and all features
        reconst_loss_mat = -px.log_prob(x)
        # Aggregate elbo losses over latent dimensions / input features
        if self.reduction == 'sum':
            kl_divergence_z = kl_divergence_z_mat.sum(-1)
            reconst_loss = reconst_loss_mat.sum(-1) * rl_weight
        elif self.reduction == 'mean':
            kl_divergence_z = kl_divergence_z_mat.mean(-1)
            reconst_loss = reconst_loss_mat.mean(-1) * rl_weight
        else:
            raise ValueError(f'reduction has to be either "sum" or "mean", got {self.reduction}')
        
        # weighted KL
        weighted_kl_local = kl_weight * kl_divergence_z

        # Save reconstruction losses
        kl_locals = {MODULE_KEYS.KL_Z_KEY: kl_divergence_z}
        
        # Collect losses
        lo_kwargs = {
            'loss': torch.mean(reconst_loss + weighted_kl_local),
            'reconstruction_loss': reconst_loss,
            'kl_local': kl_locals,
            'true_labels': tensors[REGISTRY_KEYS.LABELS_KEY]
        }
        # Create extra metric container
        extra_metrics = {}

        # Add contrastive loss if it is specified
        if contrastive_loss_weight is not None and contrastive_loss_weight > 0:
            contr_loss = self._contrastive_loss(z, tensors, temperature=contrastive_temperature)
            lo_kwargs['loss'] += contr_loss * contrastive_loss_weight
            extra_metrics.update({'contrastive_loss': contr_loss})
        # Add classification based losses
        if classification_ratio is not None and classification_ratio > 0:
            ce_loss, true_labels, logits, align_loss = self.classification_loss(
                tensors, 
                use_posterior_mean, 
                use_ext_emb,
                alignment_loss_weight
            )
            # Add z classification loss to overall loss
            lo_kwargs['loss'] += ce_loss * classification_ratio
            
            # Add classification loss details
            lo_kwargs.update({
                'classification_loss': ce_loss,
                'true_labels': true_labels,
                'logits': logits,
            })
            # Add alignment loss if it has been calculated
            if align_loss is not None:
                extra_metrics.update({'alignment_loss': align_loss})
                lo_kwargs['loss'] += align_loss * alignment_loss_weight
        
        # Add L1 regularization
        if self.l1_lambda is not None and self.l1_lambda > 0:
            l1 = self.l1_regularization()
            extra_metrics.update({'L1': l1})
            lo_kwargs['loss'] += l1 * self.l1_lambda
        if len(extra_metrics) > 0:
            # Add extra metrics to loss output
            lo_kwargs['extra_metrics'] = extra_metrics
        return LossOutput(**lo_kwargs)

    def on_load(self, model: BaseModelClass):
        manager = model.get_anndata_manager(model.adata, required=True)
        source_version = manager._source_registry[_constants._SCVI_VERSION_KEY]
        version_split = source_version.split(".")

        if int(version_split[0]) >= 1 and int(version_split[1]) >= 1:
            return

        # need this if <1.1 model is resaved with >=1.1 as new registry is
        # updated on setup
        manager.registry[_constants._SCVI_VERSION_KEY] = source_version

        # pre 1.1 logits fix
        model_kwargs = model.init_params_.get("model_kwargs", {})
        cls_params = model_kwargs.get("classifier_parameters", {})
        user_logits = cls_params.get("logits", False)

        if not user_logits:
            self.classifier.logits = False
            self.classifier.classifier.append(torch.nn.Softmax(dim=-1))

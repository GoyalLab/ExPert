from math import e
from typing import TYPE_CHECKING, Iterable

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS

from typing import Iterable

import logging
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.functional import one_hot

from scvi.data import _constants
from scvi.module import Classifier
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)

from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution
from scvi.model.base import BaseModelClass

import pdb


class GPVAE(BaseModuleClass):
    """
    Multi-modal adaptation of scanVI and gene-embedding data to create Perturb-seq-based predictive models.
    
    """
    def _setup_dispersion(self):
        if self.dispersion == 'gene':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input_x))
        elif self.dispersion == 'gene-batch':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input_x, self.n_batch))
        elif self.dispersion == 'gene-label':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input_x, self.n_labels))
        elif self.dispersion == 'gene-cell':
            pass
        else:
            raise ValueError("`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'.")
        
    def _update_cls_params(self):
        cls_parameters = {
            'n_layers': 0 if self.linear_classifier else self.classifier_parameters.get('n_layers', 10),
            'n_hidden': 0 if self.linear_classifier else self.classifier_parameters.get('n_hidden', 128),
            'dropout_rate': self.classifier_parameters.get('dropout_rate', 0.1),
            'logits': True,
        }
        cls_parameters.update(self.classifier_parameters)
        self.cls_parameters = cls_parameters
        

    def __init__(
        self,
        n_input_x: int,                                           # Number of .var variables (genes)
        n_input_g: int,                                       # Number of embedding dimensions
        n_labels: int,                                          # Number of unique class labels
        n_batch: int,                                           # Number of different batches/datasets
        n_hidden_x: int = 256,
        n_hidden_g: int = 128,
        n_latent_x: int = 100,
        n_latent_g: int = 10,
        n_layers_encoder_x: int = 2,
        n_layers_encoder_g: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate_encoder_x: float = 0.2,
        dropout_rate_encoder_g: float = 0.2,
        dropout_rate_decoder: float | None = None,
        lambda_g: float = 1,
        mixup_lambda: float = 1,
        update_qz: bool = True,
        update_method: Literal['additive', 'encoder', 'concat'] = 'concat',
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        linear_classifier: bool = False,
        classifier_parameters: dict = {},
        use_posterior_mean: bool = False,
        recon_weight: float = 1,
        mse_weight: float = 0.1,
        corr_weight: float = 1,
        corr_var_weight: float = 0.1,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        log_variational_emb: bool = False,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,

    ):
        from scvi.nn import DecoderSCVI, Encoder
        # Initialize base model class
        super().__init__()

        # Setup scvi part of model
        self.n_input_x = n_input_x
        self.n_input_g = n_input_g
        self.n_latent_x = n_latent_x
        self.n_latent_g = n_latent_g
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.log_variational_emb = log_variational_emb
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.gene_likelihood = gene_likelihood
        if dropout_rate_decoder is not None:
            logging.warning('Dropout rate for decoder currently unavailable. Will fall back to 0.')

        # Save some parameters for G
        self.collapse_g = n_latent_x != n_latent_g
        if self.collapse_g and update_method == 'additive':
            logging.warning(f'Got two different latent dimensions for X ({n_latent_x}) and G ({n_latent_g}).\nThis will likely lead to poor additive performance. Consider using "encoder" as update_method instead.')
        self.update_method = update_method
        self.update_qz = update_qz
        self.lambda_g = lambda_g
        self.mixup_lambda = mixup_lambda
        self.recon_weight = recon_weight
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.corr_var_weight = corr_var_weight

        # Classifier parameters
        self.classifier_parameters = classifier_parameters
        self.linear_classifier = linear_classifier
        self.use_posterior_mean = use_posterior_mean
        self.class_weights = None
        self._update_cls_params()

        # Setup parameters
        self._setup_dispersion()

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # Setup encoder input dimensions
        n_input_encoder = n_input_x + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = cat_list
        encoder_cat_list = cat_list if encode_covariates else None
        
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

        # Change setup to work similar to CPA but for gene embedding instead of drugs
        # Init encoder for X (rna-seq)
        self.x_encoder = Encoder(
            n_input=n_input_encoder, 
            n_output=n_latent_x, 
            n_hidden=n_hidden_x,
            n_layers=n_layers_encoder_x, 
            n_cat_list=encoder_cat_list, 
            dropout_rate=dropout_rate_encoder_x, 
            use_batch_norm=use_batch_norm_encoder, 
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            **_extra_encoder_kwargs,
            return_dist=True,
        )
        # Init encoder for G (PS-Gene embedding)
        self.g_encoder = Encoder(
            n_input=n_input_g,
            n_output=n_latent_g,
            n_hidden=n_hidden_g,
            n_layers=n_layers_encoder_g,
            n_cat_list=encoder_cat_list,
            dropout_rate=dropout_rate_encoder_g,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            **_extra_encoder_kwargs,
            return_dist=True,
        )
        n_input_decoder = n_latent_x + n_latent_g if self.update_method == 'concat' else n_latent_x
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input_x,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden_x,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
            **_extra_decoder_kwargs,
        )

        # Initialize classifier
        self.classifier = Classifier(
            n_input_decoder,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **self.cls_parameters,
        )

    def _get_inference_input(
        self,
        tensors,
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        return {
            MODULE_KEYS.B_KEY: tensors[REGISTRY_KEYS.B_KEY],
            MODULE_KEYS.G_EMB_KEY: tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None),
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.ZG_KEY: inference_outputs[MODULE_KEYS.ZG_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
        }

    @auto_move_data
    def classify(
        self,
        b: torch.Tensor,
        g: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        b
            Tensor of shape ``(n_obs, n_vars)``.
        g
            Tensor of shape ``(n_obs, n_emb)``.
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
        
        b_ = b
        g_ = g
        if self.log_variational:
            b_ = torch.log1p(b_)
        if self.log_variational_emb:
            g_ = torch.log1p(g_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_b = torch.cat((b_, cont_covs), dim=-1)
            encoder_input_g = torch.cat((g_, cont_covs), dim=-1)
        else:
            encoder_input_b = b_
            encoder_input_g = g_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        
        # Encode B (basal)
        qz, z = self.x_encoder(encoder_input_b, batch_index, *categorical_input)
        # Encode G (gene embedding)
        _, z_g = self.g_encoder(encoder_input_g, batch_index, *categorical_input)
        # Combine latent spaces
        _, z_ = self._update_latent_x(z, z_g)
        # Whether to use posterior mean or latent space to classify
        z_ = qz.loc if self.use_posterior_mean else z_
        # Classify
        w_y = self.classifier(z_)
        return w_y
    
    @auto_move_data
    def classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = labelled_dataset[REGISTRY_KEYS.B_KEY]  # (n_obs, n_vars)
        g = labelled_dataset[REGISTRY_KEYS.GENE_EMB_KEY] # (n_obs, n_emb)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        # Classify
        logits = self.classify(
            b, g, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )  # (n_obs, n_labels)
        ce_loss = F.cross_entropy(
            logits,
            y.view(-1).long(),
            weight=self.class_weights
        )
        return ce_loss, y, logits

    def _update_latent_x_additive(
            self, 
            z: torch.Tensor, 
            z_g: torch.Tensor, 
            qz: Distribution | None = None,
            qz_g: Distribution | None = None,
        ) -> tuple[Distribution, torch.Tensor]:
        """Combine z_g with z"""
        # Same latent dimensions --> just add latents with weight
        z_g_weighted = z_g * self.lambda_g
        z_updated = z + (z_g_weighted.sum(dim=-1, keepdim=True) if self.collapse_g else z_g_weighted)

        if self.update_qz and qz is not None and qz_g is not None:
            qz_loc_updated = qz.loc + (qz_g.loc.sum(dim=-1, keepdim=True) if self.collapse_g else qz_g.loc * self.lambda_g)
            qz_scale_updated = torch.sqrt(
            qz.scale**2 + ((qz_g.scale.sum(dim=-1, keepdim=True) if self.collapse_g else qz_g.scale * self.lambda_g)**2)
            )
            qz_combined = Normal(loc=qz_loc_updated, scale=qz_scale_updated)
            return qz_combined, z_updated
        else:
            return qz, z_updated
        
    def _update_latent_x_encoder(
            self, 
            z: torch.Tensor, 
            z_g: torch.Tensor, 
            qz: Distribution | None = None,
            qz_g: Distribution | None = None,
        ) -> tuple[Distribution, torch.Tensor]:
        """Combine z and zg using another encoder"""
        if self.update_qz and qz is not None and qz_g is not None:
            _x = qz.loc
            _g = qz_g.loc
        else:
            _x = z
            _g = z_g
        return self.xg_encoder(_x, _g)
    
    def _update_latent_x(
            self, 
            z: torch.Tensor, 
            z_g: torch.Tensor, 
            qz: Distribution | None = None,
            qz_g: Distribution | None = None,
        ) -> tuple[Distribution, torch.Tensor]:
        if len(z.shape) > 2:
            z = torch.mean(z, dim=0)
        if len(z_g.shape) > 2:
            z_g = torch.mean(z_g, dim=0)
        if self.update_method == 'additive':
            return self._update_latent_x_additive(z, z_g, qz, qz_g)
        elif self.update_method == 'encoder':
            return self._update_latent_x_encoder(z, z_g, qz, qz_g)
        elif self.update_method == 'concat':
            return qz, torch.cat((z, z_g), dim=-1)
    
    @auto_move_data
    def inference(
        self,
        b: torch.Tensor,
        g: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        b_ = b
        g_ = g
        # Determine library size for cells in batch, look for absolute values if data is scaled, add pseudocount for empty cells (should not exist)
        library = torch.log(torch.abs(b).sum(1)+1e-9).unsqueeze(1)
        if self.log_variational:
            b_ = torch.log1p(b_)
        if self.log_variational_emb:
            g_ = torch.log1p(g_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_b = torch.cat((b_, cont_covs), dim=-1)
            encoder_input_g = torch.cat((g_, cont_covs), dim=-1)
        else:
            encoder_input_b = b_
            encoder_input_g = g_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        
        # Encode B (basal)
        qz, z = self.x_encoder(encoder_input_b, batch_index, *categorical_input)
        # Encode G (gene embedding)
        qz_g, z_g = self.g_encoder(encoder_input_g, batch_index, *categorical_input)
        # Mix g and re-do encoding to get a baseline for batch
        if self.mixup_lambda < 1.0:
            # Perform mixup on z_g
            rolled_g = torch.roll(g, shifts=1, dims=0)
            if cont_covs is not None and self.encode_covariates:
                mixed_input = torch.cat((rolled_g, cont_covs), dim=-1)
            else:
                mixed_input = rolled_g

            z_g_rolled = self.g_encoder(mixed_input, batch_index, *categorical_input)[1]
            z_g = self.mixup_lambda * z_g + (1 - self.mixup_lambda) * z_g_rolled

        # We use observed lib size
        ql = None

        # Draw more than one sample from distribution
        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.x_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
            z = torch.mean(z, dim=0)

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.ZG_KEY: z_g,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QZG_KEY: qz_g,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        zg: torch.Tensor,
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
            Poisson,
            ZeroInflatedNegativeBinomial,
        )
        # Create prior based on zx only
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        # Concatente latent spaces
        _, z_ = self._update_latent_x(z, zg)

        # Init decoder input
        if cont_covs is None:
            decoder_input = z_
        elif z_.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z_, cont_covs.unsqueeze(0).expand(z_.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z_, cont_covs], dim=-1)
        # Manage categorical covariates
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        # We use observed library size
        size_factor = library
        # Decode
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        # Determine likelihood distribution
        if self.dispersion == "gene-label":
            px_r = linear(
                one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
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
            px = Normal(px_rate, px_r)

        # Use observed library size
        pl = None

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    def _corr_loss(
        self,
        x,
        px: Distribution,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        from torchmetrics.functional import r2_score
        # Get actual labels
        y = logits.reshape(-1)
        # Split x by y and compute gene-gene correlation matrices
        unique_labels = torch.unique(logits)
 
        r2_mean, r2_var = 0.0, 0.0
        if self.gene_likelihood == 'normal':
            px_pred_mean = px.loc
            px_pred_var = px.scale ** 2
        elif self.gene_likelihood == 'poisson':
            raise NotImplementedError('Poisson distribution not compatible with correlation loss sorry.')
        else:
            px_pred_mean = px.mu
            px_pred_var = px.theta

        # Compute R2 for each predicted label
        for label in unique_labels:
            # Get indices for the current label
            label_indices = (y == label).nonzero(as_tuple=True)[0]

            # Subset x and x_pred for the current label
            x_label = x[label_indices]
            x_pred_label_mean = px_pred_mean[label_indices]
            x_pred_label_var = px_pred_var[label_indices]

            # Replace NaNs
            x_pred_label_mean = torch.nan_to_num(x_pred_label_mean, nan=0, posinf=1e3, neginf=-1e3)
            x_pred_label_var = torch.nan_to_num(x_pred_label_var, nan=0, posinf=1e3, neginf=-1e3)
            # Compute R2 score
            r2_m = torch.nan_to_num(r2_score(x_pred_label_mean.mean(0), x_label.mean(0)),
                                    nan=0.0)
            r2_v = torch.nan_to_num(r2_score(x_pred_label_var.mean(0), x_label.var(0)),
                                    nan=0.0)
            # Ensure loss can always be minimized
            r2_mean += r2_m.abs() * (-1 if r2_m > 0 else 1)
            r2_var += r2_v.abs() * (-1 if r2_v > 0 else 1)
            
        n_unique_indices = len(unique_labels)
        # Normalize by number of classes
        r2_mean /= n_unique_indices
        r2_var /= n_unique_indices

        # Weight the loss like this:
        # r2_mean + 0.5 * r2_var + math.e ** (disnt_after - disnt_basal)

        return {
            MODULE_KEYS.R2_MEAN_KEY: r2_mean, 
            MODULE_KEYS.R2_VAR_KEY: r2_var
        }
    
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ) -> LossOutput:
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # Perturbed expression
        x = tensors[REGISTRY_KEYS.X_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        px = generative_outputs[MODULE_KEYS.PX_KEY]
        # Base KL loss on z only
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY], pz
        ).sum(dim=-1)
        # We always use observed library size
        kl_divergence_l = torch.zeros_like(kl_divergence_z)
        # Build reconstruction loss
        reconst_loss = -px.log_prob(x).sum(-1) * self.recon_weight

        # Save different reconstruction losses in dict
        reconst_losses = {
            'reconst_loss': reconst_loss,
        }

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Combine losses
        loss = torch.mean(reconst_loss + weighted_kl_local)

        # Add classification based losses
        if labelled_tensors is not None and classification_ratio is not None and classification_ratio > 0:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)

            if self.corr_weight > 0:
                # Compute correlation loss and add to classification loss
                corr_loss_out = self._corr_loss(
                    x, generative_outputs[MODULE_KEYS.PX_KEY], true_labels
                )
                
                r2_mean = corr_loss_out[MODULE_KEYS.R2_MEAN_KEY]
                r2_var = corr_loss_out[MODULE_KEYS.R2_VAR_KEY]

                reconst_losses['r2_mean'] = r2_mean
                reconst_losses['r2_var'] = r2_var
                # Add correlation loss
                #loss += r2_mean * self.corr_weight + r2_var * self.corr_var_weight
            # Add classification loss
            loss += ce_loss * classification_ratio

            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_losses,
                kl_local=kl_divergence_z,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
            )
        return LossOutput(loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_divergence_z)

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

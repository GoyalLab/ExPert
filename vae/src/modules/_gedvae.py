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


class GEDVAE(BaseModuleClass):
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
        n_input_x: int,
        n_input_g: int,
        n_labels: int,
        n_batch: int,
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
        adjust_by_mean: bool = True,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        y_prior: torch.Tensor | None = None,
        linear_classifier: bool = False,
        classifier_parameters: dict = {},
        use_posterior_mean: bool = True,
        recon_weight: float = 1,
        normalize_recon_loss: Literal['cell', 'gene-cell'] | None = None,
        g_weight: float = 1,
        g_classification_weight: float = 1,
        contrastive_y: float = 0.25,
        contrastive_margin: float = 1,
        contrastive_loss_weight: float = 1,
        g_activation: Callable = torch.exp,
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
        from scvi.nn import DecoderSCVI, Encoder, Decoder, EncoderTOTALVI, DecoderTOTALVI
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
        self.ignore_g = self.n_input_g == 0
        self.adjust_by_mean = adjust_by_mean
        self.recon_weight = recon_weight
        self.normalize_recon_loss = normalize_recon_loss
        self.g_weight = g_weight
        self.g_classification_weight = g_classification_weight
        self.contrastive_y = contrastive_y
        self.contrastive_margin = contrastive_margin
        self.contrastive_loss_weight = contrastive_loss_weight
        if self.contrastive_loss_weight > 0 and n_latent_x != n_latent_g:
            logging.warning(f'Contrastive loss can only be used if the latent dimensions of X and G are equal. Disabling contrastive loss for training.')
            self.contrastive_loss_weight = 0
        self.g_activation = g_activation
        
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

        # Setup g encoder input
        n_input_encoder_g = n_input_encoder + n_input_g

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
            n_input=n_input_encoder_g,
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

        self.decoder = DecoderSCVI(
            n_input=n_latent_x,
            n_output=n_input_x,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden_x,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
            **_extra_decoder_kwargs,
        )

        self.g_decoder = DecoderSCVI(
            n_input=n_latent_g,
            n_output=n_input_encoder_g,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden_g,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
            **_extra_decoder_kwargs,
        )

        # Initialize classifier
        self.classifier = Classifier(
            n_latent_x,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **self.cls_parameters,
        )

        self.g_classifier = Classifier(
            n_latent_g,
            n_labels=n_labels,
            n_hidden=n_hidden_g,
            n_layers=n_layers_encoder_g,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            logits=True,
        )
        # Initialize prior probabilities for y
        self.y_prior = torch.nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )

        # Add scanvi latent vae
        self.encoder_z2_z1 = Encoder(
            n_latent_x,
            n_latent_x,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_encoder_x,
            n_hidden=n_hidden_x,
            dropout_rate=dropout_rate_encoder_x,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )
        self.decoder_z1_z2 = Decoder(
            n_latent_x,
            n_latent_x,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_decoder,
            n_hidden=n_hidden_x,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(
        self,
        tensors,
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
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
        x: torch.Tensor,
        g: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        x
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
        inference_outputs = self.inference(x, g, batch_index, cont_covs, cat_covs)
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        z = qz.loc if self.use_posterior_mean else z

        # Classify based on x alone
        w_y = self.classifier(z)
        class_out = {MODULE_KEYS.X_KEY: w_y}
        # Classify based on x + g
        if not self.ignore_g and g is not None:
            qzg = inference_outputs[MODULE_KEYS.QZG_KEY]
            zg = inference_outputs[MODULE_KEYS.ZG_KEY]
            zg = qzg.loc if self.use_posterior_mean else zg
            w_yg = self.g_classifier(zg)
            class_out[MODULE_KEYS.G_EMB_KEY] = w_yg
        return class_out
    
    @auto_move_data
    def classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        g = labelled_dataset[REGISTRY_KEYS.GENE_EMB_KEY] # (n_obs, n_emb)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        # Classify
        pred_output = self.classify(
            x, g, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )  # (n_obs, n_labels)
        logits_x = pred_output[MODULE_KEYS.X_KEY]
        ce_loss_x = F.cross_entropy(
            logits_x,
            y.view(-1).long(),
            weight=self.class_weights,
            reduction='sum'
        )
        class_out = {MODULE_KEYS.X_KEY: (ce_loss_x, y, logits_x)}
        logits_g = pred_output.get(MODULE_KEYS.G_EMB_KEY)
        if logits_g is not None:
            ce_loss_g = F.cross_entropy(
                logits_g,
                y.view(-1).long(),
                weight=self.class_weights,
                reduction='sum'
            )
            class_out[MODULE_KEYS.G_EMB_KEY] = (ce_loss_g, y, logits_g)
        return class_out
    
    def _zx_zg_distances(
        self,
        zx: torch.Tensor,
        zg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate cell-wise distances based on latent space"""
        dzx = torch.cdist(zx, zx)
        dzg = torch.cdist(zg, zg)
        return dzx, dzg
    
    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        g_ = g
        # Determine library size for cells in batch, look for absolute values if data is scaled, add pseudocount for empty cells (should not exist)
        library = torch.log(torch.abs(x_).sum(1)+1e-9).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_x = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input_x = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        
        # Encode B (basal)
        qz, z = self.x_encoder(encoder_input_x, batch_index, *categorical_input)
        # Encode G (if exists)
        if g is not None:
            if self.log_variational_emb:
                g_ = torch.log1p(g_)
            if self.g_activation is not None:
                g_ = self.g_activation(g_)
            if cont_covs is not None and self.encode_covariates is True:
                encoder_input_g = torch.cat((x_, g_, cont_covs), dim=-1)
            else:
                encoder_input_g = torch.cat((x_, g_), dim=-1)
            qz_g, z_g = self.g_encoder(encoder_input_g, batch_index, *categorical_input)
            if self.g_weight > 0:
                if self.adjust_by_mean:
                    xpwd, gpwd = self._zx_zg_distances(qz.loc, qz_g.loc)
                else:
                    xpwd, gpwd = self._zx_zg_distances(z, z_g)
            else:
                xpwd, gpwd = None, None
        else:
            qz_g, z_g, xpwd, gpwd = None, None, None, None

        # We use observed lib size
        ql = None

        # Draw more than one sample from distribution
        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.x_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.ZG_KEY: z_g,
            MODULE_KEYS.QZG_KEY: qz_g,
            MODULE_KEYS.PWDX_KEY: xpwd,
            MODULE_KEYS.PWDG_KEY: gpwd
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
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

        # Init decoder input
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
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
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    def contrastive_loss(self, z: torch.Tensor, zg: torch.Tensor):
        D = F.pairwise_distance(z, zg)
        y = self.contrastive_y
        margin = self.contrastive_margin
        loss = (1 - y) * 0.5 * D**2 + y * 0.5 * torch.clamp(margin - D, min=0.0)**2
        return loss.sum()
    
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

        # X inference and generative
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        # G inference
        xpwd: torch.Tensor | None = inference_outputs.get(MODULE_KEYS.PWDX_KEY)
        gpwd: torch.Tensor | None = inference_outputs.get(MODULE_KEYS.PWDG_KEY)
        zg: torch.Tensor | None = inference_outputs.get(MODULE_KEYS.ZG_KEY)
        
        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z = kl_divergence(qz, pz).sum(dim=-1)
        # We always use observed library size
        kl_divergence_l = torch.zeros_like(kl_divergence_z)
        # Calculate reconstruction loss
        reconst_loss = -px.log_prob(x).sum(-1) * self.recon_weight

        # KL local warmup
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        # weighted KL
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Save reconstruction losses
        reconst_losses = {
            'reconst_loss': reconst_loss,
        }
        # Combine losses
        if kl_weight == 0:
            loss = torch.mean(reconst_loss)
        else:    
            loss = torch.mean(reconst_loss + weighted_kl_local)
        # Normalize both reconstruction and KL
        if self.normalize_recon_loss == 'cell':
            loss /= (x.shape[0])
        if self.normalize_recon_loss == 'gene-cell':
            loss /= (x.shape[0] * self.n_input_x)

        # Calculate pair-wise cell distance loss
        if xpwd is not None and gpwd is not None and self.g_weight > 0:
            distance_loss = F.mse_loss(xpwd, gpwd, reduction='sum')
            reconst_losses['distance_loss'] = distance_loss
            loss += distance_loss * self.g_weight
        # Calculate contrastive loss between latent spaces
        if zg is not None and self.contrastive_loss_weight > 0:
            contrastive_loss = self.contrastive_loss(z, zg)
            reconst_losses['contrastive_loss'] = contrastive_loss
            loss += contrastive_loss * self.contrastive_loss_weight

        # Add classification based losses
        if labelled_tensors is not None:
            pred_loss_output = self.classification_loss(labelled_tensors)
            # Classification based on zx
            ce_loss, true_labels, logits = pred_loss_output[MODULE_KEYS.X_KEY]
            # Add classification loss to overall loss
            loss += ce_loss * classification_ratio

            if MODULE_KEYS.G_EMB_KEY in pred_loss_output:
                ce_loss_g, _, _ = pred_loss_output[MODULE_KEYS.G_EMB_KEY]
                loss += ce_loss_g * self.g_classification_weight
                reconst_losses[MODULE_KEYS.G_EMB_KEY] = ce_loss_g

            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_losses,
                kl_local=weighted_kl_local,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
            )
        return LossOutput(loss=loss, reconstruction_loss=reconst_losses, kl_local=weighted_kl_local)

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

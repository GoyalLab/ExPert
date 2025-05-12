from typing import TYPE_CHECKING, Iterable

from src.utils.common import FCParams
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS
from src.utils.distributions import NormalMixture

import logging
import pdb

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot

from scvi.data import _constants
from scvi.module._utils import broadcast_labels
from scvi.module import Classifier
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)

from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial
)

from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution
from scvi.model.base import BaseModelClass


class GENEVAE(BaseModuleClass):
    """
    Multi-modal adaptation of scanVI and gene-embedding data to create Perturb-seq-based predictive models.
    
    """


    def _setup_dispersion(self):
        if self.dispersion == 'gene':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input))
            self.pg_r = torch.nn.Parameter(2 * torch.rand(self.n_input_emb))
        elif self.dispersion == 'gene-batch':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input, self.n_batch))
            self.pg_r = torch.nn.Parameter(2 * torch.rand(self.n_input_emb, self.n_batch))
        elif self.dispersion == 'gene-label':
            self.px_r = torch.nn.Parameter(torch.randn(self.n_input, self.n_labels))
            self.pg_r = torch.nn.Parameter(2 * torch.rand(self.n_input_emb, self.n_labels))
        elif self.dispersion == 'gene-cell':
            pass
        else:
            raise ValueError("`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'.")

    def _setup_embedding_background(self, emb_background_prior_mean: np.ndarray | None, emb_background_prior_scale: np.ndarray | None):
        # parameters for prior on rate_back (background protein mean)
        if emb_background_prior_mean is None:
            if self.n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(self.n_input_emb, self.n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(self.n_input_emb, self.n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(torch.randn(self.n_input_emb))
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(self.n_input_emb), -10, 1)
                )
        else:
            if emb_background_prior_mean.shape[1] == 1 and self.n_batch != 1:
                init_mean = emb_background_prior_mean.ravel()
                init_scale = emb_background_prior_scale.ravel()
            else:
                init_mean = emb_background_prior_mean
                init_scale = emb_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        
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
        n_input: int,                                           # Number of .var variables (genes)
        n_input_emb: int,                                       # Number of embedding dimensions
        n_labels: int,                                          # Number of unique class labels
        n_batch: int,                                           # Number of different batches/datasets
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        y_prior: torch.Tensor | None = None,
        emb_background_prior_mean: np.ndarray | None = None,
        emb_background_prior_scale: np.ndarray | None = None,
        emb_batch_mask: dict[str | int, np.ndarray] | None = None,
        linear_classifier: bool = False,
        classifier_parameters: dict = {},
        use_posterior_mean: bool = True,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        log_variational_emb: bool = False,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        emb_likelihood: Literal['nb', 'normal'] = 'normal',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        scale_losses: bool = False,
    ):
        from scvi.nn import DecoderSCVI, Encoder, Decoder, EncoderTOTALVI, DecoderTOTALVI
        # Initialize base model class
        super().__init__()

        # Setup scvi part of model
        self.n_input = n_input
        self.n_input_emb = n_input_emb
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.log_variational_emb = log_variational_emb
        self.emb_batch_mask = emb_batch_mask
        self.gene_likelihood = gene_likelihood
        self.emb_likelihood = emb_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.scale_losses = scale_losses
        if (self.emb_likelihood == 'normal' or self.gene_likelihood == 'normal') and self.gene_likelihood != self.emb_likelihood:
            logging.warning(f'Detected that only one modality follows a normal distribution. This could lead to errors in reconstruction.')
        
        if self.scale_losses:
            self.max_losses = {
                'rl': -np.log(1/(n_labels))*n_input,
                'rl_g': -np.log(1/(n_labels))*n_input_emb,
                'kl': 1,
                'kl_l': 1,
                'kl_g': 1,
                'cls': -np.log(1/n_labels)
            }
        # Classifier parameters
        self.classifier_parameters = classifier_parameters
        self.linear_classifier = linear_classifier
        self.use_posterior_mean = use_posterior_mean
        self.class_weights = None
        self._update_cls_params()

        # Setup parameters
        self._setup_embedding_background(emb_background_prior_mean, emb_background_prior_scale)
        self._setup_dispersion()
        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # Setup encoder input dimensions
        n_input_total = n_input + n_input_emb
        n_input_encoder = n_input_total + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = cat_list
        encoder_cat_list = cat_list if encode_covariates else None
        
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

        # Change setup to work like TOTALVI but instead of protein expression use gene embeddings
        self.encoder = EncoderTOTALVI(
            n_input=n_input_encoder, 
            n_output=n_latent, 
            n_hidden=n_hidden,
            n_layers=n_layers_encoder, 
            n_cat_list=encoder_cat_list, 
            dropout_rate=dropout_rate_encoder, 
            use_batch_norm=use_batch_norm_encoder, 
            use_layer_norm=use_layer_norm_encoder,
            **_extra_encoder_kwargs,
        )

        self.decoder = DecoderTOTALVI(
            n_latent + n_continuous_cov,
            n_input,
            self.n_input_emb,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
            **_extra_decoder_kwargs,
        )

        # Initialize classifier
        self.classifier = Classifier(
            n_latent,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **self.cls_parameters,
        )
        # Initialize prior probabilities for y
        self.y_prior = torch.nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )

        # Add scanvi latent vae
        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )
        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    @auto_move_data
    def classify(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        x_ = x
        g_ = g
        # Determine library size for cells in batch
        # library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)
        if self.log_variational_emb:
            g_ = torch.log1p(g_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, g_, cont_covs), dim=-1)
        else:
            encoder_input = torch.cat((x_, g_), dim=-1)
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        # Encoder
        _, _, latent, _ = self.encoder(
            encoder_input, batch_index, *categorical_input
        )
        z = latent["z"]
        # Classify
        w_y = self.classifier(z)
        return w_y
    
    @auto_move_data
    def classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        g = labelled_dataset[REGISTRY_KEYS.GENE_EMB_KEY] # (n_obs, n_emb)
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        # Classify
        logits = self.classify(
            x, g, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )  # (n_obs, n_labels)
        ce_loss = F.cross_entropy(
            logits,
            y.view(-1).long(),
            weight=self.class_weights
        )
        return ce_loss, y, logits
    
    def _get_inference_input(
        self,
        tensors,
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
            MODULE_KEYS.G_EMB_KEY: tensors[REGISTRY_KEYS.GENE_EMB_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        library = inference_outputs[MODULE_KEYS.LIBRARY_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = tensors[size_factor_key] if size_factor_key in tensors.keys() else None

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.BATCH_INDEX_KEY: batch_index,
            MODULE_KEYS.LABEL_KEY: label,
            MODULE_KEYS.CAT_COVS_KEY: cat_covs,
            MODULE_KEYS.CONT_COVS_KEY: cont_covs,
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
        }
    
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
        # Determine library size for cells in batch
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)
        if self.log_variational_emb:
            g_ = torch.log1p(g_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, g_, cont_covs), dim=-1)
        else:
            encoder_input = torch.cat((x_, g_), dim=-1)
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        # Encode joint data
        qz, ql, latent, untran_latent = self.encoder(
            encoder_input, batch_index, *categorical_input
        )

        z = latent["z"]
        untran_z = untran_latent["z"]
        untran_l = untran_latent["l"]
        library_gene = latent["l"]

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.encoder.z_transformation(untran_z)

            untran_l = ql.sample((n_samples,))
            library_gene = self.encoder.l_transformation(untran_l)
        # Adjust for batch effects
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index.squeeze(-1), self.n_batch).float(),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library_gene,
            'untran_l': untran_l,
        }
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        label: torch.Tensor,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        transform_batch: int | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Run the generative step."""
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
        size_factor = library
        # Decode joint latent space
        px_, pg_, log_pro_back_mean = self.decoder(
            decoder_input, size_factor, batch_index, *categorical_input
        )
        # Act based on dispersion model
        if self.dispersion == 'gene-label':
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label.squeeze(-1), self.n_labels).float(), self.px_r)
            pg_r = F.linear(one_hot(label.squeeze(-1), self.n_labels).float(), self.pg_r)
        elif self.dispersion == 'gene-batch':
            px_r = F.linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
            pg_r = F.linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.pg_r)
        elif self.dispersion == 'gene':
            px_r = self.px_r
            pg_r = self.pg_r
        px_r = torch.exp(px_r)
        pg_r = torch.exp(pg_r)
        px_['r'] = px_r
        pg_['r'] = pg_r
        return {
            MODULE_KEYS.PX_KEY: px_,
            MODULE_KEYS.PG_KEY: pg_,
            'log_pro_back_mean': log_pro_back_mean,
        }
    
    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        px_dict: dict[str, torch.Tensor],
        pg_dict: dict[str, torch.Tensor],
        pro_batch_mask_minibatch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        px_ = px_dict
        pg_ = pg_dict
        # Reconstruction Loss (rna seq)
        if self.gene_likelihood == 'zinb':
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == 'nb':
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"]).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == 'normal':
            reconst_loss_gene = (
                -Normal(loc=px_["rate"], scale=px_["r"]).log_prob(x).sum(dim=-1)
            )

        # Reconstruction loss (gene embedding)
        if self.emb_likelihood == 'nb':
            pg_conditional = NegativeBinomialMixture(
                mu1=pg_["rate_back"],
                mu2=pg_["rate_fore"],
                theta1=pg_["r"],
                mixture_logits=pg_["mixing"],
            )
        elif self.emb_likelihood == 'normal':
            pg_conditional = NormalMixture(
                mu1=pg_["rate_back"],
                mu2=pg_["rate_fore"],
                sigma1=pg_["r"],
                mixture_logits=pg_["mixing"],
            )
        reconst_loss_emb_full = -pg_conditional.log_prob(g)
        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = pro_batch_mask_minibatch.bool() * reconst_loss_emb_full
            reconst_loss_emb = temp_pro_loss_full.sum(dim=-1)
        else:
            reconst_loss_emb = reconst_loss_emb_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_emb

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 0.25,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float = 100,
        emb_recon_weight: float = .1,
    ):
        """Compute the loss."""
        # Collect inference outputs
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        ql = inference_outputs[MODULE_KEYS.QL_KEY]
        # Collect generative outputs
        px = generative_outputs[MODULE_KEYS.PX_KEY]
        pg = generative_outputs[MODULE_KEYS.PG_KEY]
        # Extract data from tensors dict
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        g = tensors[REGISTRY_KEYS.GENE_EMB_KEY]

        if self.emb_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(g)
            for b in torch.unique(batch_index):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.emb_batch_mask[str(int(b.item()))].astype(np.float32),
                    device=g.device,
                )
        else:
            pro_batch_mask_minibatch = None
        
        # Calculate reconstruction loss based on both modalities
        reconst_loss_gene, reconst_loss_emb = self.get_reconstruction_loss(
            x, g, px, pg, pro_batch_mask_minibatch
        )

        # KL Divergence (gene expression)
        kl_div_z = kl(qz, Normal(0, 1)).sum(dim=-1)
        kl_div_l_gene = torch.zeros_like(kl_div_z)
        # KL Divergence (gene embedding)
        kl_div_back_emb_full = kl(
            Normal(pg["back_alpha"], pg["back_beta"]), self.back_mean_prior
        )
        if pro_batch_mask_minibatch is not None:
            kl_div_back_emb = pro_batch_mask_minibatch.bool() * kl_div_back_emb_full
            kl_div_back_emb = kl_div_back_emb.sum(dim=-1)
        else:
            kl_div_back_emb = kl_div_back_emb_full.sum(dim=-1)
        
        if self.scale_losses:
            # Anchor losses on maximum value
            reconst_loss_gene /= self.max_losses['rl']
            reconst_loss_emb /= self.max_losses['rl_g']
            kl_div_z /= self.max_losses['kl']
            kl_div_l_gene /= self.max_losses['kl_l']
            kl_div_back_emb /= self.max_losses['kl_g']
        # Combine final loss
        loss = torch.mean(
            reconst_loss_gene
            + emb_recon_weight * reconst_loss_emb
            + kl_weight * kl_div_z
            + kl_div_l_gene
            + kl_weight * kl_div_back_emb
        )

        reconst_losses = {
            "reconst_loss_gene": reconst_loss_gene,
            "reconst_loss_emb": reconst_loss_emb,
        }
        kl_local = {
            "kl_div_z": kl_div_z,
            "kl_div_l_gene": kl_div_l_gene,
            "kl_div_back_emb": kl_div_back_emb,
        }

        # Add classification loss
        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)

            loss += ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_losses,
                kl_local=kl_local,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
            )
        return LossOutput(loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local)


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

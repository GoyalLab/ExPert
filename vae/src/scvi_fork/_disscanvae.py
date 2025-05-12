from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F
from sklearn.utils.class_weight import compute_class_weight

from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.module.base import LossOutput, auto_move_data
from scvi.module._constants import MODULE_KEYS
from scvi.nn import Decoder, Encoder, DecoderSCVI, EncoderProjector

from ._classifier import Classifier
from ._utils import broadcast_labels
from ._scanvae import VAE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from torch.distributions import Distribution

    from scvi.model.base import BaseModelClass


class DisentangledSCANVAE(VAE):
    """Single-cell RNA-seq model with a disentangled latent space designed to learn genome-wide perturbations from multiple datasets.
    """

    def __init__(
        self,
        # base VAE parameters
        n_input: int,
        n_groups: int,
        n_labels_per_group: Iterable[int],
        n_batch: int = 0,
        n_hidden: int = 128,
        n_base_latent: int = 64,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "zinb",
        # disentanglement parameters
        n_latent_per_group: Iterable[int] | None = None,
        classifier_params_per_group: Iterable[dict] | None = None,
        # classifier parameters
        linear_classifier: bool = False,
        y_priors: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,          # TODO: integrate class weights
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        **vae_kwargs,
    ):
        super().__init__(
            n_input=n_input,                        # number of input genes
            n_batch=n_batch,                        # number of datasets to integrate
            n_latent=n_base_latent,                 # number of dimensions for Z_h, base latent space
            n_hidden=n_hidden,                      # number of hidden units in the hidden layers for the base encoder
            n_layers=n_layers,                      # number of hidden layers for the base encoder
            n_continuous_cov=n_continuous_cov,      # number of continuous covariates for base VAE
            n_cats_per_cov=n_cats_per_cov,          # number of categories for each continuous covariate
            dropout_rate=dropout_rate,              # dropout rate for the base encoder
            dispersion=dispersion,                  # dispersion parameter for the base VAE
            log_variational=log_variational,        # whether to log the variational parameters
            gene_likelihood=gene_likelihood,        # gene likelihood model
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs,
        )

        # check if group parameters are provided and match number of groups
        if n_groups == 0:
            raise ValueError("Number of groups must be greater than 0.")
        if len(n_labels_per_group) != n_groups:
            raise ValueError("Number of groups must match number of group labels.")
        if n_latent_per_group is None:
            n_latent_per_group = np.repeat(np.round(n_base_latent/2), n_groups)
        elif len(n_latent_per_group) != n_groups:
            raise ValueError("Number of groups must match number of group latent dimensions.")
        if classifier_params_per_group is None:
            classifier_params_per_group = np.repeat({}, n_groups)
        elif len(classifier_params_per_group) != n_groups:
            raise ValueError("Number of groups must match number of group classifier parameters.")
        else:
            cls_parameters = {
                "n_layers": 0 if linear_classifier else n_layers,
                "n_hidden": 0 if linear_classifier else n_hidden,
                "logits": True,
            }
            for cls_params in classifier_params_per_group:
                cls_params.update(cls_parameters)

        # check if encoder and decoder batch/layer norm are used
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # create Projector for each group from Z_h to Z_g, and create classifier C_g for each group
        self.projectors = torch.nn.ModuleList()
        self.classifiers = torch.nn.ModuleList()
        y_priors = []
        n_latent = 0                        # get sum of latent dimensions of group projections Z_t
        for i in np.arange(n_groups):
            n_labels_g, n_latent_g, cp = (
                n_labels_per_group[i],
                n_latent_per_group[i],
                classifier_params_per_group[i],
            )
            projector = EncoderProjector(
                encoder_latent_dim=n_base_latent,
                projector_latent_dim=n_latent_g,
            )
            self.projectors.append(projector)
            classifier = Classifier(
                n_latent_g,
                n_labels_g,
                **cp,
            )
            self.classifiers.append(classifier)
            # create y prior for each group
            y_prior = torch.nn.Parameter(
                y_priors[i] if y_priors[i] is not None else (1 / n_labels_g) * torch.ones(1, n_labels_g),
                requires_grad=False,
            )
            y_priors.append(y_prior)

            n_latent += n_latent_g
        # collect y priors for each group
        self.y_prior = y_priors

        # create base decoder for combined latent space Z_t
        self.zt_decoder = DecoderSCVI(
            n_latent,                           # decode based on concatenated latent space Z_t
            n_input,                            # project back up to input features (genes)
            n_cat_list=self.cat_list,           # include categorical covariates like dataset, batch index
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=self.deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if self.use_size_factor_key else "softmax",
            **vae_kwargs.get('_extra_decoder_kwargs', {}),
        )

        # create encoder and decoder for scanvi like loss
        self.encoder_z2_z1 = Encoder(
                n_latent,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                return_dist=True,
        )
        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | Iterable[Distribution] | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # encode to base latent space Z_h
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz_h, _ = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz_h, _ = self.z_encoder(encoder_input, batch_index, *categorical_input)
        # encode library size in Z_l
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded
        
        # compute Z_h to Z_t over Z_gs
        z_ts = []       # collect latent space for each group
        qz_t_ms, qz_t_vs = [], []      # collect distribution parameters for each group
        qz_ts = []      # collect distribution for each group
        for projector in self.projectors:
            qg_mu, qg_var = projector(qz_h)
            dist = Normal(qg_mu, qg_var.sqrt())
            latent = self.z_encoder.z_transformation(dist.rsample())
            qz_ts.append(dist)              # collect distribution
            qz_t_ms.append(dist.loc)         # collect mean of distribution
            qz_t_vs.append(dist.scale)       # collect variance of distribution
            z_ts.append(latent)             # collect latent space
        # concatenate all latent spaces
        z_t = torch.cat(z_ts, dim=-1)
        # get overall distribution for Z_t
        qz_t_m = torch.mean(qz_t_ms)
        qz_t_v = torch.mean(qz_t_vs)
        qz_t = Normal(qz_t_m, qz_t_v)

        # draw samples from qz_t
        if n_samples > 1:
            untran_z = qz_t.sample((n_samples,))
            z_t = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))

        return {
            MODULE_KEYS.Z_KEY: z_t,                 # concatenated latent spaces
            MODULE_KEYS.ZT_G_KEY: z_ts,             # latent spaces for each group
            MODULE_KEYS.QZ_KEY: qz_t,               # distribution for concatenated latent spaces
            MODULE_KEYS.QT_G_KEY: qz_ts,            # distribution for latent spaces for each group
            MODULE_KEYS.QL_KEY: ql,                 # library size distribution
            MODULE_KEYS.LIBRARY_KEY: library,       # library size
        }
    
    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,
        qzv: torch.Tensor,
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """Run the cached inference process."""
        raise NotImplementedError("Cached inference not implemented for disentangled model.")
        # TODO: implement cached inference for disentangled model
        from torch.distributions import Normal

        qz = Normal(qzm, qzv.sqrt())
        # use dist.sample() rather than rsample because we aren't optimizing the z here
        untran_z = qz.sample() if n_samples == 1 else qz.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: None,
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

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_scale, px_r, px_rate, px_dropout = self.zt_decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            px_scale, px_r, px_rate, px_dropout = self.zt_decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
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
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
    ) -> torch.Tensor:
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
        Tensor of shape ``(n_groups, n_obs, n_labels)`` denoting logit scores per label.
        Before v1.1, this method by default returned probabilities per label,
        see #2301 for more details.
        """
        if self.log_variational:
            x = torch.log1p(x)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        # get base latent space Z_h from encoder
        qz_h, z_h = self.z_encoder(encoder_input, batch_index, *categorical_input)
        z_h = qz_h.loc if use_posterior_mean else z_h
        # get projections from Z_h to Z_g for each group and classify for each group
        class_logits = []
        for i, projector in enumerate(self.projectors):
            qg_mu, qg_var = projector(z_h)                              # project Z_h to Z_g distribution
            dist = Normal(qg_mu, qg_var.sqrt())                         # distribution for Z_g
            latent = self.z_encoder.z_transformation(dist.rsample())    # transform Z_g to latent space
            c = self.classifiers[i](latent)                             # classify group latent space (n_obs, n_labels)
            class_logits.append(c)                                      # collect logits for each group
        return torch.tensor(class_logits)
    
    @auto_move_data
    def classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]  # (n_obs, n_vars)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        # NOTE: prior to v1.1, this method returned probabilities per label by
        # default, see #2301 for more details
        logits_per_group = self.classify(
            x, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )  # (n_groups, n_obs, n_labels)
        # compute cross entropy loss for each group
        ce_losses = []
        ys = []
        for i, logits in enumerate(logits_per_group):
            KEY = REGISTRY_KEYS.GROUP_BASE_KEY + str(i)
            y = labelled_dataset[KEY]       # (n_obs, 1)
            ce_loss = F.cross_entropy(
                logits,
                y.view(-1).long(),
                weight=self.class_weights[i] if self.class_weights is not None else None,
            )
            ce_losses.append(ce_loss)
        return torch.tensor(ce_losses), torch.tensor(ys), logits_per_group

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_ouputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ):
        """Compute the loss."""
        px: Distribution = generative_ouputs[MODULE_KEYS.PX_KEY]
        qz1: torch.Tensor = inference_outputs[MODULE_KEYS.QZ_KEY]
        z1: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        batch_index: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]

        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        reconst_loss = -px.log_prob(x).sum(-1)

        # KL Divergence
        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)

        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        # Classification loss for each group
        logits_per_group = self.classify(z1)
        probs_per_group = [F.softmax(logits, dim=-1) for logits in logits_per_group]

        if z1.ndim == 2:
            loss_z1_unweight_ = loss_z1_unweight.view(self.n_labels, -1).t()
            kl_divergence_z2_ = kl_divergence_z2.view(self.n_labels, -1).t()
        else:
            loss_z1_unweight_ = torch.transpose(
                loss_z1_unweight.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
            kl_divergence_z2_ = torch.transpose(
                kl_divergence_z2.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
        for i, probs in enumerate(probs_per_group):
            reconst_loss += loss_z1_weight + (loss_z1_unweight_ * probs).sum(dim=-1)
            kl_divergence = (kl_divergence_z2_ * probs).sum(dim=-1)
            kl_divergence += kl(
                Categorical(probs=probs),
                Categorical(
                    probs=self.y_prior[i].repeat(probs.size(0), probs.size(1), 1)
                    if len(probs.size()) == 3
                    else self.y_prior[i].repeat(probs.size(0), 1)
                ),
            )

        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence)

        kl_divergence += kl_divergence_l
        # Normalize reconstruction loss to match scale of classifaication loss
        #reconst_loss /= 1000
        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        if labelled_tensors is not None:
            ce_loss_pg, _, _ = self.classification_loss(labelled_tensors)
            ce_loss = torch.sum(ce_loss_pg)
            loss += ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local=kl_divergence,
                classification_loss=ce_loss,
                extra_metrics={REGISTRY_KEYS.GROUP_BASE_KEY + str(i): c for i, c in enumerate(ce_loss_pg)},
            )
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence)


from src.modules._gpvae import GPVAE
from src.utils.constants import REGISTRY_KEYS
from src._train.plan import SemiSupervisedTrainingPlan
from src.utils.preprocess import _prep_adata

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from typing import Literal, Any
import numpy.typing as npt
from collections.abc import Iterator

from anndata import AnnData

# scvi imports
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.data import AnnDataManager
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.data._utils import get_anndata_attribute
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from scvi.train._callbacks import SubSampleLabels
from scvi.model.base import (
    BaseModelClass,
    ArchesMixin,
    VAEMixin,
    RNASeqMixin
)
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    ObsmField,
    NumericalJointObsField,
    NumericalObsField,
)

logger = logging.getLogger(__name__)


class GPVI(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    _module_cls = GPVAE
    _training_plan_cls = SemiSupervisedTrainingPlan
    _LATENT_QZM_KEY = "gpvi_latent_qzm"
    _LATENT_QZV_KEY = "gpvi_latent_qzv"


    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float | None = None,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        linear_classifier: bool = False,
        use_class_weights: bool = False,
        n_latent_g: int | None = None,
        n_hidden_g: int | None = None,
        n_layers_encoder_g: int | None = None,
        dropout_rate_encoder_g: float | None = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        gpvae_model_kwargs = dict(model_kwargs)
        # Initialize indices and labels for this VAE
        self._set_indices_and_labels()

        # Ignores unlabeled catgegory
        n_labels = self.summary_stats.n_labels - 1
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # Add classification weights
        class_weights = None
        if use_class_weights:
            labels_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
            labels = self.adata.obs[labels_key].values
            class_weights = compute_class_weight('balanced', classes=self._label_mapping[:-1], y=labels)
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch
        n_dims_emb = self.summary_stats.n_gene_embedding if 'n_gene_embedding' in self.summary_stats.keys() else 0
        # Look for special parameters for G, fall back to X parameters
        n_latent_g = n_latent_g if n_latent_g else n_latent
        n_hidden_g = n_hidden_g if n_hidden_g else n_hidden
        n_layers_encoder_g = n_layers_encoder_g if n_layers_encoder_g else n_layers_encoder
        dropout_rate_encoder_g = dropout_rate_encoder_g if dropout_rate_encoder_g else dropout_rate_encoder

        # Initialize genevae
        self.module = self._module_cls(
            n_input_x=self.summary_stats.n_vars,
            n_input_g=n_dims_emb,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent_x=n_latent,
            n_latent_g=n_latent_g,
            n_hidden_x=n_hidden,
            n_hidden_g=n_hidden_g,
            n_layers_encoder_x=n_layers_encoder,
            n_layers_encoder_g=n_layers_encoder_g,
            n_layers_decoder=n_layers_decoder,
            dropout_rate_encoder_x=dropout_rate_encoder,
            dropout_rate_encoder_g=dropout_rate_encoder_g,
            dropout_rate_decoder=dropout_rate_decoder,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            **gpvae_model_kwargs,
        )

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels
        # TODO: give more detailed summary
        self._model_summary_string = (
            f"GPVI Model with the following params: \n"
            f"unlabeled_category: {self.unlabeled_category_}, n_classes: {self.n_labels}, "
            f"dispersion: {dispersion}, gene_likelihood: {gene_likelihood}"
        )
    
    

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells. Same as used in scanVI.x"""
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = None,
        ignore_embedding: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        indices
            Return probabilities for each class label.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        use_posterior_mean
            If ``True``, uses the mean of the posterior distribution to predict celltype
            labels. Otherwise, uses a sample from the posterior distribution - this
            means that the predictions will be stochastic.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )
        y_pred = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            g = None if ignore_embedding else tensors[REGISTRY_KEYS.GENE_EMB_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred = self.module.classify(
                x,
                g,
                batch_index=batch,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
            )
            if self.module.classifier.logits:
                pred = torch.nn.functional.softmax(pred, dim=-1)
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred).numpy()
        if not soft:
            predictions = []
            for p in y_pred:
                predictions.append(self._code_to_label[p])

            return np.array(predictions)
        else:
            n_labels = len(pred[0])
            pred = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return pred
        
    @devices_dsp.dedent
    def train(
        self,
        data_params: dict[str, Any]={}, 
        model_params: dict[str, Any]={}, 
        train_params: dict[str, Any]={},
        return_runner: bool = False
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset for semisupervised training.
        n_samples_per_label
            Number of subsamples for each label class to sample per epoch. By default, there
            is no label subsampling.
        check_val_every_n_epoch
            Frequency with which metrics are computed on the data for validation set for both
            the unsupervised and semisupervised trainers. If you'd like a different frequency for
            the semisupervised trainer, set check_val_every_n_epoch in semisupervised_train_kwargs.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        %(param_accelerator)s
        %(param_devices)s
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.SemiSupervisedDataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.SemiSupervisedTrainingPlan`. Keyword arguments
            passed to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        epochs = train_params.get('max_epochs')
        # determine number of epochs needed for complete training
        max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        if epochs is None:
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
            epochs = max_epochs
        logging.info(f'Epochs suggested: {max_epochs}, training for {epochs} epochs.')

        train_size: int = data_params.pop('train_size', 0.9)
        if not train_size < 1.0 and train_size > 0:
            raise ValueError(f'Parameter train_size should be between 0 and 1, got {train_size}')

        # Create data splitter
        data_splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            **data_params,
        )

        plan_kwargs = train_params.pop('plan_kwargs', {})
        # create training plan
        training_plan = self._training_plan_cls(self.module, self.n_labels, **plan_kwargs)
        check_val_every_n_epoch = train_params.pop('check_val_every_n_epoch', 10)
        # if we have labeled cells, we want to subsample labels each epoch
        sampler_callback = [SubSampleLabels()] if len(self._labeled_indices) != 0 else []
        if "callbacks" in train_params.keys():
            train_params["callbacks"] += [sampler_callback]
        else:
            train_params["callbacks"] = sampler_callback
        # create training runner
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            accelerator='auto',
            devices='auto',
            check_val_every_n_epoch=check_val_every_n_epoch,
            **train_params
        )
        if 'logger' in train_params.keys():
            # save hyper-parameters to lightning logs
            hparams = {
                'data_params': data_params,
                'scanvi_params': model_params,
                'plan_params': plan_kwargs,
                'train_params': train_params
            }
            runner.trainer.logger.log_hyperparams(hparams)
        # Train model
        if return_runner:
            return runner
        else:
            return runner()
        
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        ignore_embedding: bool = False,
        dataloader: Iterator[dict[str, torch.Tensor | None]] = None,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with :attr:`~anndata.AnnData.var_names` in the same
            order as the ones used to train the model. If ``None`` and ``dataloader`` is also
            ``None``, it defaults to the object used to initialize the model.
        indices
            Indices of observations in ``adata`` to use. If ``None``, defaults to all observations.
            Ignored if ``dataloader`` is not ``None``
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns an
            estimate of the mean using ``mc_samples`` Monte Carlo samples.
        mc_samples
            Number of Monte Carlo samples to use for the estimator for distributions with no
            closed-form mean (e.g., the logistic normal distribution). Not used if ``give_mean`` is
            ``True`` or if ``return_dist`` is ``True``.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``. Ignored if ``dataloader`` is not ``None``
        return_dist
            If ``True``, returns the mean and variance of the latent distribution. Otherwise,
            returns the mean of the latent distribution.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

        Returns
        -------
        An array of shape ``(n_obs, n_latent)`` if ``return_dist`` is ``False``. Otherwise, returns
        a tuple of arrays ``(n_obs, n_latent)`` with the mean and variance of the latent
        distribution.
        """
        from torch.distributions import Distribution, Normal
        from torch.nn.functional import softmax
        from torch import Tensor

        from scvi.module._constants import MODULE_KEYS

        self._check_if_trained(warn=False)
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        zs: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []
        for tensors in dataloader:
            if ignore_embedding:
                tensors[REGISTRY_KEYS.GENE_EMB_KEY] = None
            outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                **self.module._get_inference_input(tensors)
            )

            if MODULE_KEYS.QZ_KEY in outputs:
                qz: Distribution = outputs.get(MODULE_KEYS.QZ_KEY)
                qzm: Tensor = qz.loc
                qzv: Tensor = qz.scale.square()
            else:
                qzm: Tensor = outputs.get(MODULE_KEYS.QZM_KEY)
                qzv: Tensor = outputs.get(MODULE_KEYS.QZV_KEY)
                qz: Distribution = Normal(qzm, qzv.sqrt())

            if return_dist:
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                continue

            z: Tensor = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)

            if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                samples = qz.sample([mc_samples])
                z = softmax(samples, dim=-1).mean(dim=0)

            zs.append(z.cpu())

        if return_dist:
            return torch.cat(qz_means).numpy(), torch.cat(qz_vars).numpy()
        else:
            return torch.cat(zs).numpy()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str,
        cls_labels: list[str],
        ctrl_key: str = 'control',
        layer: str | None = None,
        basal_layer: str | None = None,
        gene_emb_obsm_key: str | None = 'gene_embedding',
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        cast_to_csr: bool = True,
        raw_counts: bool = False,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        copy: bool = False,
        **kwargs,
    ):
        """
        Setup AnnData object for training a GPVI model.
        """
        if adata.is_view or copy:
            logging.info(f'Copying adata.')
            adata = adata.copy()
        if not isinstance(adata.X, sp.csr_matrix) and cast_to_csr:
            logging.info('Casting adata.X to csr matrix to boost training efficiency.')
            adata.X = sp.csr_matrix(adata.X)
        setup_method_args = cls._get_setup_method_args(**locals())

        # Split adata into control (basal), perturbed, and gene embedding data
        # .X = perturbed, .layers['basal'] = basal, .obsm['gene_embedding'] = gene embedding
        if REGISTRY_KEYS.B_KEY not in adata.layers:
            _prep_adata(adata, cls_labels, ctrl_key, p_pos=-1)
        
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=raw_counts),
            LayerField(REGISTRY_KEYS.B_KEY, basal_layer, is_count_data=raw_counts),
            ObsmField(REGISTRY_KEYS.GENE_EMB_KEY, gene_emb_obsm_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        # Create AnnData manager
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
